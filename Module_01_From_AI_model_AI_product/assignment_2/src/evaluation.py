from __future__ import annotations

import asyncio
import ast
import json
from typing import Any, Literal

import pandas as pd
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm

from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from .config import settings


tqdm.pandas()


class SingleCriterionScore(BaseModel):
    verdict: Literal["correct", "incorrect"]
    justification: str


JUDGE_PROMPT = """You are a financial QA judge.

Return your answer in json only.

Compare the model answer to the ground truth.
Return exactly this JSON schema:
{
  "verdict": "correct" or "incorrect",
  "justification": "one short sentence"
}

Rules:
- Mark correct only if the answer matches the ground truth in meaning.
- Minor wording differences are allowed.
- Numeric mistakes, wrong entities, wrong direction, or unsupported claims are incorrect.
- Be concise and strict."""


def extract_json_obj(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Could not parse JSON from: {text[:300]}")
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object.")
    return obj


def judge_correctness(
    question: str,
    ground_truth: str,
    answer: str,
    client: OpenAI,
    judge_model: str,
) -> dict[str, Any]:
    user_prompt = f"""Question:
{question}

Ground truth:
{ground_truth}

Model answer:
{answer}
"""

    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content.strip()
    try:
        parsed = extract_json_obj(raw_text)
        result = SingleCriterionScore.model_validate(parsed)
        return result.model_dump()
    except (ValidationError, ValueError, json.JSONDecodeError):
        return {
            "verdict": "incorrect",
            "justification": f"Judge output could not be parsed: {raw_text[:150]}",
        }


def chunk_texts(chunks: Any) -> list[str]:
    if not isinstance(chunks, list):
        return []
    out: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("content") or chunk.get("chunk")
            if text is not None:
                out.append(str(text))
    return out


def page_hit_at_k(row: pd.Series, k: int) -> int:
    evidence_pages = set(row["evidence_pages"] if isinstance(row["evidence_pages"], list) else [])

    retrieved_chunks = row["retrieved_chunks"]
    if isinstance(retrieved_chunks, str):
        retrieved_chunks = retrieved_chunks.strip()
        if retrieved_chunks:
            try:
                retrieved_chunks = ast.literal_eval(retrieved_chunks)
            except (ValueError, SyntaxError):
                retrieved_chunks = []
        else:
            retrieved_chunks = []

    if not isinstance(retrieved_chunks, list):
        retrieved_chunks = []

    retrieved_pages: set[int] = set()
    for chunk in retrieved_chunks[:k]:
        if isinstance(chunk, dict):
            page_value = chunk.get("page_number")
            if page_value is None:
                continue
            try:
                retrieved_pages.add(int(str(page_value).strip()))
            except (TypeError, ValueError):
                continue

    return int(bool(evidence_pages & retrieved_pages))


async def run_faithfulness_scores_async(
    df: pd.DataFrame,
    answer_column: str = "RAG_answer",
    retrieved_chunks_column: str = "retrieved_chunks",
    question_column: str = "question",
    limit: int | None = None,
) -> list[Any]:
    async_client = AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.nebuis_api_key,
    )
    ragas_llm = llm_factory(model=settings.judge_model, client=async_client, max_tokens=4096)
    scorer = Faithfulness(llm=ragas_llm)

    faith_scores: list[Any] = [pd.NA] * len(df)
    rows = list(df.iterrows())
    if limit is not None:
        rows = rows[:limit]

    for idx, (_, row) in enumerate(rows):
        answer_value = row[answer_column]
        if isinstance(answer_value, dict):
            answer_value = answer_value.get("answer", "")
        contexts = chunk_texts(row[retrieved_chunks_column])
        result = await scorer.ascore(
            user_input=row[question_column],
            response=str(answer_value),
            retrieved_contexts=contexts,
        )
        faith_scores[idx] = result.value

    return faith_scores


def run_faithfulness_scores(
    df: pd.DataFrame,
    answer_column: str = "RAG_answer",
    retrieved_chunks_column: str = "retrieved_chunks",
    question_column: str = "question",
    limit: int | None = None,
) -> list[Any]:
    return asyncio.run(
        run_faithfulness_scores_async(
            df,
            answer_column=answer_column,
            retrieved_chunks_column=retrieved_chunks_column,
            question_column=question_column,
            limit=limit,
        )
    )


def get_avg_metrics(df: pd.DataFrame) -> pd.DataFrame:
    export_df = df[
        [
            "financebench_id",
            "question",
            "correctness",
            "faithfulness",
            "page_hit_at_1",
            "page_hit_at_3",
            "page_hit_at_5",
        ]
    ].copy()

    for column in ["faithfulness", "page_hit_at_1", "page_hit_at_3", "page_hit_at_5"]:
        export_df[column] = pd.to_numeric(export_df[column], errors="coerce")

    correctness_num = export_df["correctness"].map({"correct": 1, "incorrect": 0})
    avg_correctness = correctness_num.mean()
    avg_faithfulness = export_df["faithfulness"].dropna().mean()
    page_hit_at_1 = export_df["page_hit_at_1"].mean()
    page_hit_at_3 = export_df["page_hit_at_3"].mean()
    page_hit_at_5 = export_df["page_hit_at_5"].mean()

    print("Average correctness:", avg_correctness)
    print("Average faithfulness:", avg_faithfulness)
    print("Page-hit@1:", page_hit_at_1)
    print("Page-hit@3:", page_hit_at_3)
    print("Page-hit@5:", page_hit_at_5)

    return export_df
