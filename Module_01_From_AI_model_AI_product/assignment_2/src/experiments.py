from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import settings
from .data import build_chunks, build_doc_info, extract_evidence_pages, get_embeddings, get_pdf_files, load_financebench_data, load_or_build_vectorstore, sample_task1_rows
from .evaluation import get_avg_metrics, judge_correctness, page_hit_at_k, run_faithfulness_scores
from .rag import SYSTEM_PROMPT, SYSTEM_PROMPT_STRICT, answer_with_rag


@dataclass
class ProjectArtifacts:
    dataframe: pd.DataFrame
    vectorstore: Any


def build_project_context(project_root: str | Path) -> dict[str, Any]:
    project_root = Path(project_root)
    data_path = project_root / "financebench_merged.jsonl"
    pdf_dir = project_root / "pdfs"
    index_dir = project_root / "faiss_financebench"

    df = load_financebench_data(data_path)
    doc_info = build_doc_info(df)
    pdf_files = get_pdf_files(pdf_dir)
    embeddings = get_embeddings()
    chunks = build_chunks(doc_info, pdf_files)
    vectorstore = load_or_build_vectorstore(embeddings, index_dir=index_dir, chunks=chunks)

    return {
        "df": df,
        "doc_info": doc_info,
        "pdf_files": pdf_files,
        "embeddings": embeddings,
        "chunks": chunks,
        "vectorstore": vectorstore,
    }


def run_naive_generation(df: pd.DataFrame, client, model: str = settings.model, limit: int | None = None) -> pd.DataFrame:
    task_rows = sample_task1_rows(df)
    if limit is not None:
        task_rows = task_rows.head(limit)

    results = []
    for _, row in task_rows.iterrows():
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": row["question"]}],
            temperature=0.0,
            max_tokens=1024,
        )
        results.append(
            {
                "financebench_id": row["financebench_id"],
                "question_type": row["question_type"],
                "question": row["question"],
                "naive_answer": response.choices[0].message.content.strip(),
                "ground_truth": row["answer"],
                "verdict": "",
            }
        )

    return pd.DataFrame(results)


def run_rag_answers(
    df: pd.DataFrame,
    client,
    vectorstore,
    model: str = settings.model,
    system_prompt: str = SYSTEM_PROMPT,
    k: int = 4,
    limit: int | None = None,
) -> pd.DataFrame:
    if limit is not None:
        df = df.head(limit)

    rows = []
    for _, row in df.iterrows():
        rag_result = answer_with_rag(client, model, vectorstore, system_prompt, row["question"], k=k)
        rows.append(
            {
                "financebench_id": row["financebench_id"],
                "question_type": row.get("question_type"),
                "question": row["question"],
                "ground_truth": row.get("answer"),
                "evidence_pages": extract_evidence_pages(row.get("evidence")),
                "RAG_answer": rag_result["answer"],
                "retrieved_chunks": rag_result["retrieved_chunks"],
            }
        )
    return pd.DataFrame(rows)


def score_rag_dataframe(
    df: pd.DataFrame,
    client,
    judge_model: str = settings.judge_model,
    faithfulness_limit: int | None = None,
) -> pd.DataFrame:
    scored = df.copy()
    scored["correctness_raw"] = scored.progress_apply(
        lambda row: judge_correctness(
            row["question"],
            row["ground_truth"],
            row["RAG_answer"],
            client,
            judge_model,
        ),
        axis=1,
    )
    scored["correctness"] = scored["correctness_raw"].apply(lambda value: value["verdict"])
    scored["correctness_justification"] = scored["correctness_raw"].apply(lambda value: value["justification"])

    for k in [1, 3, 5]:
        scored[f"page_hit_at_{k}"] = scored.apply(lambda row, kk=k: page_hit_at_k(row, kk), axis=1)

    scored["faithfulness"] = run_faithfulness_scores(scored, limit=faithfulness_limit)
    return scored


def build_metrics_export(df: pd.DataFrame) -> pd.DataFrame:
    return get_avg_metrics(df)


def build_cycle_summary(base_df: pd.DataFrame, prompt_df: pd.DataFrame, model_df: pd.DataFrame, topk3_df: pd.DataFrame, topk8_df: pd.DataFrame) -> pd.DataFrame:
    def metrics_frame(frame: pd.DataFrame) -> dict[str, float]:
        export_df = get_avg_metrics(frame)
        correctness_num = export_df["correctness"].map({"correct": 1, "incorrect": 0})
        return {
            "correctness": correctness_num.mean(),
            "faithfulness": export_df["faithfulness"].dropna().mean(),
            "page_hit_at_1": export_df["page_hit_at_1"].mean(),
            "page_hit_at_3": export_df["page_hit_at_3"].mean(),
            "page_hit_at_5": export_df["page_hit_at_5"].mean(),
        }

    baseline = metrics_frame(base_df)
    prompt = metrics_frame(prompt_df)
    model = metrics_frame(model_df)
    topk3 = metrics_frame(topk3_df)
    topk8 = metrics_frame(topk8_df)

    return pd.DataFrame(
        {
            "experiment": ["baseline", "prompt", "model", "top-k_3", "top-k_8"],
            "change": [
                "Baseline: naive generation with meta-llama/Llama-3.3-70B-Instruct without retrieval.",
                "Added system prompt to guide the model to answer only from provided context and cite sources.",
                "Switched generation model to a stronger model.",
                "Decreased retrieved chunks from 5 to 3.",
                "Increased retrieved chunks from 5 to 8.",
            ],
            "correctness": [baseline["correctness"], prompt["correctness"], model["correctness"], topk3["correctness"], topk8["correctness"]],
            "faithfulness": [baseline["faithfulness"], prompt["faithfulness"], model["faithfulness"], topk3["faithfulness"], topk8["faithfulness"]],
            "page_hit_at_1": [baseline["page_hit_at_1"], prompt["page_hit_at_1"], model["page_hit_at_1"], topk3["page_hit_at_1"], topk8["page_hit_at_1"]],
            "page_hit_at_3": [baseline["page_hit_at_3"], prompt["page_hit_at_3"], model["page_hit_at_3"], topk3["page_hit_at_3"], topk8["page_hit_at_3"]],
            "page_hit_at_5": [baseline["page_hit_at_5"], prompt["page_hit_at_5"], model["page_hit_at_5"], topk3["page_hit_at_5"], topk8["page_hit_at_5"]],
        }
    )
