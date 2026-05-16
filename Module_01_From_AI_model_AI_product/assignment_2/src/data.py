from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .config import settings


def load_financebench_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_json(Path(path), lines=True)
    df = df[df["question_type"] != "metrics-generated"].copy()
    df = df.drop(columns=["dataset_subset_label"], errors="ignore")
    return df


def sample_task1_rows(df: pd.DataFrame, per_type: int = 5) -> pd.DataFrame:
    task1_rows = []
    for question_type in ["domain-relevant", "novel-generated"]:
        subset = df[df["question_type"] == question_type].sort_values("financebench_id").head(per_type)
        task1_rows.append(subset)
    return pd.concat(task1_rows, ignore_index=True).sort_values("financebench_id")


def get_pdf_files(pdf_dir: str | Path) -> dict[str, Path]:
    pdf_dir = Path(pdf_dir)
    return {path.stem: path for path in pdf_dir.glob("*.pdf")}


def load_pdf_pages(doc_name: str, pdf_files: dict[str, Path]) -> list[Any]:
    pdf_path = pdf_files.get(doc_name)
    if pdf_path is None:
        return []
    return PyPDFLoader(str(pdf_path)).load()


def add_metadata(pages: list[Any], doc_name: str, company: str, doc_period: str) -> list[Any]:
    for index, page in enumerate(pages):
        page.metadata["doc_name"] = doc_name
        page.metadata["company"] = company
        page.metadata["doc_period"] = doc_period
        page.metadata["page_number"] = index
    return pages


def _estimate_tokens(text: str) -> int:
    # A light heuristic for token count when model tokenizer is unavailable.
    return max(1, len(text) // 4)


def _normalize_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line.strip())
    line = re.sub(r"\d", "#", line)
    return line.lower()


def _looks_like_page_number(line: str) -> bool:
    raw = line.strip().lower()
    if not raw:
        return False
    if re.fullmatch(r"\d+", raw):
        return True
    if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", raw):
        return True
    return False


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > 140:
        return False
    if _looks_like_page_number(stripped):
        return False
    if stripped.endswith(":"):
        return True
    if re.fullmatch(r"\d+(\.\d+)*\s+.+", stripped):
        return True
    alpha_chars = sum(ch.isalpha() for ch in stripped)
    if alpha_chars == 0:
        return False
    upper_ratio = sum(ch.isupper() for ch in stripped if ch.isalpha()) / alpha_chars
    if upper_ratio >= 0.8 and len(stripped.split()) <= 12:
        return True
    title_like = stripped[:1].isupper() and len(stripped.split()) <= 12
    return title_like and not stripped.endswith(".")


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 4:
        return False
    if "|" in stripped:
        return True
    if "\t" in stripped:
        return True
    if len(re.findall(r"\s{2,}", stripped)) >= 2:
        return True
    if len(re.findall(r"\d", stripped)) >= 4 and len(re.findall(r"\s{2,}", stripped)) >= 1:
        return True
    return False


def _split_large_text_with_minimal_overlap(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = _estimate_tokens(sentence)
        if current and current_tokens + sent_tokens > max_tokens:
            chunk_text = " ".join(current).strip()
            chunks.append(chunk_text)

            if overlap_tokens > 0:
                overlap: list[str] = []
                overlap_count = 0
                for existing in reversed(current):
                    overlap.insert(0, existing)
                    overlap_count += _estimate_tokens(existing)
                    if overlap_count >= overlap_tokens:
                        break
                current = overlap
                current_tokens = sum(_estimate_tokens(part) for part in current)
            else:
                current = []
                current_tokens = 0

        current.append(sentence)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _collect_repeated_margin_lines(pages: list[Any]) -> set[str]:
    candidate_lines: list[str] = []
    for page in pages:
        lines = [line for line in page.page_content.splitlines() if line.strip()]
        if not lines:
            continue
        margin = lines[:3] + lines[-3:]
        for line in margin:
            norm = _normalize_line(line)
            if norm:
                candidate_lines.append(norm)

    if not pages:
        return set()

    counts = Counter(candidate_lines)
    threshold = max(2, int(len(pages) * 0.2))
    return {
        line
        for line, count in counts.items()
        if count >= threshold and len(line) <= 120
    }


def _extract_structural_blocks(page_text: str, repeated_margin_lines: set[str]) -> list[dict[str, str]]:
    lines = [line.rstrip() for line in page_text.splitlines()]
    blocks: list[dict[str, str]] = []

    current_type: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_type, current_lines
        if not current_lines or current_type is None:
            current_type = None
            current_lines = []
            return
        text = "\n".join(line for line in current_lines if line.strip()).strip()
        if text:
            blocks.append({"type": current_type, "text": text})
        current_type = None
        current_lines = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush()
            continue

        norm = _normalize_line(line)
        if norm in repeated_margin_lines:
            continue
        if _looks_like_page_number(line):
            continue

        if _is_heading_line(line):
            flush()
            blocks.append({"type": "heading", "text": line})
            continue

        line_type = "table" if _is_table_line(line) else "narrative"
        if current_type is None:
            current_type = line_type
            current_lines = [line]
            continue

        if line_type != current_type:
            flush()
            current_type = line_type
            current_lines = [line]
            continue

        current_lines.append(line)

    flush()
    return blocks


def _split_table_into_row_groups(table_text: str, max_tokens: int) -> list[str]:
    lines = [line for line in table_text.splitlines() if line.strip()]
    if not lines:
        return []

    header = lines[0]
    rows = lines[1:] if len(lines) > 1 else []

    if _estimate_tokens(table_text) <= max_tokens:
        return [table_text]

    if not rows:
        return _split_large_text_with_minimal_overlap(table_text, max_tokens=max_tokens, overlap_tokens=0)

    chunks: list[str] = []
    current_rows: list[str] = []
    current_tokens = _estimate_tokens(header)

    for row in rows:
        row_tokens = _estimate_tokens(row)
        if current_rows and current_tokens + row_tokens > max_tokens:
            chunks.append("\n".join([header] + current_rows))
            current_rows = []
            current_tokens = _estimate_tokens(header)
        current_rows.append(row)
        current_tokens += row_tokens

    if current_rows:
        chunks.append("\n".join([header] + current_rows))

    return chunks


def _split_narrative_with_langchain(
    text: str,
    base_metadata: dict[str, Any],
    section_title: str,
    md_splitter: MarkdownHeaderTextSplitter,
    child_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    if not text.strip():
        return []

    section_header = section_title.strip() if section_title else "Section"
    markdown_text = f"## {section_header}\n\n{text.strip()}"
    md_chunks = md_splitter.split_text(markdown_text)

    final_docs: list[Document] = []
    for md_chunk in md_chunks:
        parent_meta = {
            **base_metadata,
            **md_chunk.metadata,
            "chunk_type": "narrative",
            "section_title": md_chunk.metadata.get("h3")
            or md_chunk.metadata.get("h2")
            or md_chunk.metadata.get("h1")
            or section_title,
        }
        temp_doc = Document(page_content=md_chunk.page_content, metadata=parent_meta)
        final_docs.extend(child_splitter.split_documents([temp_doc]))

    return final_docs


def _append_structural_chunks(
    all_chunks: list[Document],
    page_metadata: dict[str, Any],
    blocks: list[dict[str, str]],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    md_splitter: MarkdownHeaderTextSplitter,
    child_splitter: RecursiveCharacterTextSplitter,
) -> None:
    min_tokens = int(target_tokens * 0.8)
    current_section = ""
    heading_pending_for_next_content = ""
    narrative_buffer: list[str] = []
    narrative_tokens = 0

    def flush_narrative() -> None:
        nonlocal narrative_buffer, narrative_tokens, heading_pending_for_next_content
        if not narrative_buffer:
            return

        text = "\n\n".join(narrative_buffer).strip()
        if not text:
            narrative_buffer = []
            narrative_tokens = 0
            return

        if heading_pending_for_next_content and current_section:
            text = f"{heading_pending_for_next_content}\n\n{text}"
            heading_pending_for_next_content = ""

        docs = _split_narrative_with_langchain(
            text=text,
            base_metadata=page_metadata,
            section_title=current_section,
            md_splitter=md_splitter,
            child_splitter=child_splitter,
        )
        for idx, doc in enumerate(docs):
            doc.metadata["section_chunk_index"] = idx
            all_chunks.append(doc)

        narrative_buffer = []
        narrative_tokens = 0

    for block in blocks:
        block_type = block["type"]
        block_text = block["text"].strip()
        if not block_text:
            continue

        if block_type == "heading":
            flush_narrative()
            current_section = block_text
            heading_pending_for_next_content = block_text
            continue

        if block_type == "table":
            flush_narrative()
            table_chunks = _split_table_into_row_groups(block_text, max_tokens=max_tokens)
            for idx, table_chunk in enumerate(table_chunks):
                content = table_chunk
                if heading_pending_for_next_content:
                    content = f"{heading_pending_for_next_content}\n\n{content}"
                    heading_pending_for_next_content = ""
                metadata = {
                    **page_metadata,
                    "chunk_type": "table",
                    "section_title": current_section,
                    "section_chunk_index": idx,
                }
                all_chunks.append(Document(page_content=content, metadata=metadata))
            continue

        block_tokens = _estimate_tokens(block_text)
        if narrative_buffer and narrative_tokens >= min_tokens and narrative_tokens + block_tokens > max_tokens:
            flush_narrative()

        narrative_buffer.append(block_text)
        narrative_tokens += block_tokens

        if narrative_tokens >= max_tokens:
            flush_narrative()

    flush_narrative()


def build_chunks(
    doc_info: pd.DataFrame,
    pdf_files: dict[str, Path],
    chunk_size: int = 1000,
    chunk_overlap: int = 32,
) -> list[Any]:
    all_chunks: list[Document] = []
    target_tokens = max(128, chunk_size)
    max_tokens = int(target_tokens * 1.2)

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,
        chunk_overlap=max(0, chunk_overlap) * 4,
    )

    for _, row in doc_info.iterrows():
        pages = load_pdf_pages(row["doc_name"], pdf_files)
        if not pages:
            continue
        pages = add_metadata(pages, row["doc_name"], row["company"], row["doc_period"])
        repeated_margin_lines = _collect_repeated_margin_lines(pages)

        for page in pages:
            page_metadata = dict(page.metadata)
            blocks = _extract_structural_blocks(page.page_content, repeated_margin_lines)
            _append_structural_chunks(
                all_chunks=all_chunks,
                page_metadata=page_metadata,
                blocks=blocks,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
                overlap_tokens=max(0, chunk_overlap),
                md_splitter=md_splitter,
                child_splitter=child_splitter,
            )

    return all_chunks


def build_doc_info(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["doc_name", "company", "doc_period"]]
        .dropna()
        .drop_duplicates(subset=["doc_name"])
        .reset_index(drop=True)
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"token": settings.hf_token},
    )


def load_or_build_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    index_dir: str | Path = "faiss_financebench",
    chunks: list[Any] | None = None,
    save_if_built: bool = False,
) -> FAISS:
    index_path = Path(index_dir)
    if index_path.exists():
        return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    if chunks is None:
        raise FileNotFoundError(f"No FAISS index found at {index_path} and no chunks were provided.")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    if save_if_built:
        vectorstore.save_local(str(index_path))
    return vectorstore


def extract_evidence_pages(evidence: Any) -> list[int]:
    if evidence is None or (isinstance(evidence, float) and pd.isna(evidence)):
        return []

    if isinstance(evidence, str):
        evidence = evidence.strip()
        if not evidence:
            return []
        try:
            evidence = ast.literal_eval(evidence)
        except (ValueError, SyntaxError):
            return []

    if not isinstance(evidence, list):
        return []

    pages: list[int] = []
    for item in evidence:
        if isinstance(item, dict):
            page_value = item.get("evidence_page_num") or item.get("page_number") or item.get("page") or item.get("page_num")
            if page_value is None:
                continue
            try:
                pages.append(int(str(page_value).strip()))
            except (TypeError, ValueError):
                continue
        else:
            try:
                pages.append(int(str(item).strip()))
            except (TypeError, ValueError):
                continue
    return pages
