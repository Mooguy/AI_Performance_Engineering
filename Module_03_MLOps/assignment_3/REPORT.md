# Evaluation Pipeline Report

## Architecture

Airflow DAG (`dags/evaluate_agent.py`) with 4 tasks:

```
prepare_run -> run_agent -> run_eval -> summarize_and_log
```

- **prepare_run**: Creates `runs/<run-id>/config.json` from Airflow params
- **run_agent**: Runs `mini-extra swebench` via DockerOperator, writes trajectories + `preds.json` to `runs/<run-id>/run-agent/`
- **run_eval**: Runs `swebench.harness.run_evaluation` via DockerOperator, writes report to `runs/<run-id>/run-eval/`
- **summarize_and_log**: Parses report, writes `metrics.json` + `manifest.json`, logs to MLflow

## Artifact Layout

```
runs/<run-id>/
  config.json        # run parameters
  manifest.json      # paths to all artifacts
  metrics.json       # resolved, total, resolve_rate
  run-agent/
    preds.json
    <instance-id>/*.traj.json
  run-eval/
    *.json           # swebench evaluation report
```

## How to Trigger (Standalone)

```bash
bash run-airflow-standalone.sh
# Open http://localhost:8080
# Trigger evaluate_agent DAG with params
```

Params:
- `split`: test
- `subset`: verified
- `workers`: 2
- `model`: nebius/NousResearch/Hermes-4-70B
- `task_slice`: 0:2
- `custom_run_id`: run-001

## How to Trigger (Docker Compose)

```bash
docker compose up -d
# Open http://localhost:8080
```

## How to Rerun by run-id

Set `custom_run_id` param to the desired run-id. All artifacts write to `runs/<run-id>/`.

## MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Open http://localhost:5000
```

Logs per run: params, metrics (resolved, total, resolve_rate), config.json, metrics.json.

## Completed Runs

| Run ID  | Model | Instances | Resolved |
|---------|-------|-----------|----------|
| run-001 | Llama-3.3-70B | 2 | 0/2 |
| run-002 | Hermes-4-70B (Docker) | 2 | 0/2 |

Resolved=0 reflects model capability, not pipeline failure. Both runs completed end-to-end.

## Screenshots

- `screenshots/airflow_dag.png` - Airflow UI (Docker Compose)
- `screenshots/mlflow_runs.png` - MLflow logged run

## Notes

- S3 upload skipped (extra credit)
- Docker Compose Airflow requires `apache/airflow:2.9.0` + custom `Dockerfile.airflow` with mlflow, python-dotenv, apache-airflow-providers-docker
