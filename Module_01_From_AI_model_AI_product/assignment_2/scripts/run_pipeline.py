from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from assignment_2_project.config import get_client, settings
from assignment_2_project.data import build_chunks, build_doc_info, get_embeddings, get_pdf_files, load_financebench_data, load_or_build_vectorstore, sample_task1_rows
from assignment_2_project.evaluation import get_avg_metrics, judge_correctness, page_hit_at_k, run_faithfulness_scores
from assignment_2_project.experiments import build_metrics_export, run_naive_generation, run_rag_answers
from assignment_2_project.rag import SYSTEM_PROMPT, SYSTEM_PROMPT_STRICT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the assignment 2 FinanceBench project.")
    parser.add_argument("--mode", choices=["task1", "task5", "eval", "cycles", "all"], default="all")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    client = get_client()

    df = load_financebench_data(project_root / "financebench_merged.jsonl")
    pdf_files = get_pdf_files(project_root / "pdfs")
    doc_info = build_doc_info(df)
    embeddings = get_embeddings()
    chunks = build_chunks(doc_info, pdf_files)
    vectorstore = load_or_build_vectorstore(embeddings, index_dir=project_root / "faiss_financebench", chunks=chunks)

    if args.mode in {"task1", "all"}:
        task1_df = run_naive_generation(df, client, model=settings.model, limit=args.limit)
        task1_path = project_root / "assignment2_naive_generation.xlsx"
        task1_df.to_excel(task1_path, index=False)
        print(f"Saved task 1 output to {task1_path}")

    if args.mode in {"task5", "all"}:
        base_df = sample_task1_rows(df)
        if args.limit is not None:
            base_df = base_df.head(args.limit)
        rag_df = run_rag_answers(df=base_df, client=client, vectorstore=vectorstore, model=settings.model, system_prompt=SYSTEM_PROMPT, k=args.k)
        rag_path = project_root / "assignment2_run_and_compare.xlsx"
        rag_df.to_excel(rag_path, index=False)
        print(f"Saved RAG comparison output to {rag_path}")

    if args.mode in {"eval", "all"}:
        rag_df = run_rag_answers(df=df, client=client, vectorstore=vectorstore, model=settings.model, system_prompt=SYSTEM_PROMPT, k=args.k, limit=args.limit)
        rag_df["correctness_raw"] = rag_df.apply(
            lambda row: judge_correctness(row["question"], row["ground_truth"], row["RAG_answer"], client, settings.judge_model),
            axis=1,
        )
        rag_df["correctness"] = rag_df["correctness_raw"].apply(lambda value: value["verdict"])
        rag_df["correctness_justification"] = rag_df["correctness_raw"].apply(lambda value: value["justification"])
        for k in [1, 3, 5]:
            rag_df[f"page_hit_at_{k}"] = rag_df.apply(lambda row, kk=k: page_hit_at_k(row, kk), axis=1)
        rag_df["faithfulness"] = run_faithfulness_scores(rag_df, limit=args.limit)
        export_df = build_metrics_export(rag_df)
        eval_path = project_root / "assignment2_evaluation.xlsx"
        export_df.to_excel(eval_path, index=False)
        print(f"Saved evaluation output to {eval_path}")

    if args.mode == "cycles":
        print("Cycle experiments are exposed in the package modules. Run them by composing the helpers in assignment_2_project.experiments.")


if __name__ == "__main__":
    main()
