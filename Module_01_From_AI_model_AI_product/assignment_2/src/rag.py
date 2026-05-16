from __future__ import annotations

from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """You are a financial question-answering assistant.
Answer only from the provided context.
If the context does not contain the answer, say that the context does not contain enough information.
Be concise.
Cite the document name for each factual claim using [doc_name]. Do not invent citations."""

SYSTEM_PROMPT_STRICT = """You are a financial question-answering assistant.

You must answer using only the provided context.
If the context does not explicitly support the answer, respond exactly:
Not enough information in the retrieved context.

Rules:
- Do not use outside knowledge.
- Do not infer missing facts.
- Do not add any claim unless it is directly supported by the context.
- Cite every factual claim using the exact document name in brackets, like [doc_name].
- If the question cannot be answered from the context, give the refusal message only.
- Be concise and factual."""


def format_chunks(chunks: list[Any]) -> str:
    if not chunks:
        return "No relevant context was retrieved."

    parts: list[str] = []
    for index, doc in enumerate(chunks, 1):
        doc_name = doc.metadata.get("doc_name", "unknown")
        page_number = doc.metadata.get("page_number", "unknown")
        parts.append(
            f"### Chunk {index}\n"
            f"doc_name: {doc_name}\n"
            f"page_number: {page_number}\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def answer_with_rag(
    client: OpenAI,
    model: str,
    vectorstore: Any,
    system_prompt: str,
    query: str,
    k: int = 4,
) -> dict[str, Any]:
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    context = format_chunks(retrieved_docs)

    user_prompt = f"""Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    answer = response.choices[0].message.content.strip()
    return {
        "answer": answer,
        "retrieved_chunks": [
            {
                "doc_name": doc.metadata.get("doc_name"),
                "page_number": doc.metadata.get("page_number"),
                "text": doc.page_content,
            }
            for doc in retrieved_docs
        ],
    }
