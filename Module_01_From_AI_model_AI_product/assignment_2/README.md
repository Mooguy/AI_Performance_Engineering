# Assignment 2 Project

A clean, scriptable version of the `assignment_2.ipynb` workflow.

## What is included

- FinanceBench data loading and preprocessing
- PDF chunking and FAISS retrieval
- Naive generation baseline
- RAG answer generation
- Judge-based correctness scoring
- RAGAS faithfulness scoring
- Retrieval page-hit metrics
- Improvement cycle experiments for prompt, model, and top-k variants

## Layout

- `src/assignment_2_project/` contains the reusable code
- `scripts/run_pipeline.py` is the main entry point
- `requirements.txt` lists the notebook dependencies
- `.env.example` shows the expected environment variables

## Environment variables

Copy `.env.example` to `.env` and set the values you need:

- `NEBIUS_API_KEY`
- `HF_TOKEN`

## Inputs expected in the project root

- `financebench_merged.jsonl`
- `pdfs/`
- `faiss_financebench/` if you want to load the prebuilt FAISS index

## Typical usage

```bash
python scripts/run_pipeline.py --mode all
```

Other useful modes:

```bash
python scripts/run_pipeline.py --mode task1
python scripts/run_pipeline.py --mode task5
python scripts/run_pipeline.py --mode eval
python scripts/run_pipeline.py --mode cycles
```

## Notes

- The project leaves the original notebook untouched.
- The code is structured so you can reuse individual steps from a normal Python workflow instead of running everything inside Jupyter.
