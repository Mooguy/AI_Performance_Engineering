import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTAINER_ROOT = "/mlops-assignment"

load_dotenv(PROJECT_ROOT / ".env")


@dag(
    dag_id="evaluate_agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "split": Param("test", type="string"),
        "subset": Param("verified", type="string"),
        "workers": Param(2, type="integer"),
        "model": Param("nebius/NousResearch/Hermes-4-70B", type="string"),
        "task_slice": Param("0:2", type="string"),
        "custom_run_id": Param("run-001", type="string"),
    },
)
def evaluate_agent():

    @task
    def prepare_run(**context):
        params = context["params"]
        dag_run_id = params["custom_run_id"] or f"run-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        run_dir = PROJECT_ROOT / "runs" / dag_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run-agent").mkdir(exist_ok=True)
        (run_dir / "run-eval").mkdir(exist_ok=True)

        config = {
            "run_id": dag_run_id,
            "split": params["split"],
            "subset": params["subset"],
            "workers": params["workers"],
            "model": params["model"],
            "task_slice": params["task_slice"],
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
        return dag_run_id

    run_agent = DockerOperator(
        task_id="run_agent",
        image="mlops-agent",
        command=(
            "uv run mini-extra swebench"
            " --subset {{ params.subset }}"
            " --split {{ params.split }}"
            " --model {{ params.model }}"
            " --slice {{ params.task_slice }}"
            " --workers {{ params.workers }}"
            f" -o {CONTAINER_ROOT}/runs/{{{{ ti.xcom_pull(task_ids='prepare_run') }}}}/run-agent"
        ),
        mounts=[
            Mount(source=str(PROJECT_ROOT), target=CONTAINER_ROOT, type="bind"),
            Mount(source="/var/run/docker.sock", target="/var/run/docker.sock", type="bind"),
        ],
        environment={
            "NEBIUS_API_KEY": os.environ.get("NEBIUS_API_KEY", ""),
            "MSWEA_COST_TRACKING": "ignore_errors",
        },
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mount_tmp_dir=False,
    )

    run_eval = DockerOperator(
        task_id="run_eval",
        image="mlops-agent",
        command=(
            "bash -c '"
            f"mkdir -p {CONTAINER_ROOT}/runs/{{{{ ti.xcom_pull(task_ids='prepare_run') }}}}/run-eval && "
            f"cd {CONTAINER_ROOT}/runs/{{{{ ti.xcom_pull(task_ids='prepare_run') }}}}/run-eval && "
            f"{CONTAINER_ROOT}/.venv/bin/python -m swebench.harness.run_evaluation"
            " --dataset_name princeton-nlp/SWE-bench_{{ params.subset | capitalize }}"
            f" --predictions_path {CONTAINER_ROOT}/runs/{{{{ ti.xcom_pull(task_ids='prepare_run') }}}}/run-agent/preds.json"
            " --max_workers {{ params.workers }}"
            " --run_id {{ ti.xcom_pull(task_ids='prepare_run') }}"
            f" --report_dir {CONTAINER_ROOT}/runs/{{{{ ti.xcom_pull(task_ids='prepare_run') }}}}/run-eval"
            "'"
        ),
        mounts=[
            Mount(source=str(PROJECT_ROOT), target=CONTAINER_ROOT, type="bind"),
            Mount(source="/var/run/docker.sock", target="/var/run/docker.sock", type="bind"),
        ],
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mount_tmp_dir=False,
    )

    @task
    def summarize_and_log(**context):
        params = context["params"]
        dag_run_id = params["custom_run_id"] or f"run-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        run_dir = PROJECT_ROOT / "runs" / dag_run_id
        eval_out = run_dir / "run-eval"

        metrics = {"resolved": 0, "total": 0, "resolve_rate": 0.0}
        for search_dir in [eval_out, PROJECT_ROOT]:
            for f in search_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    if isinstance(data, dict) and "resolved_instances" in data:
                        metrics["resolved"] = data.get("resolved_instances", 0)
                        metrics["total"] = data.get("submitted_instances", 0)
                        if metrics["total"] > 0:
                            metrics["resolve_rate"] = metrics["resolved"] / metrics["total"]
                        break
                except Exception:
                    continue

        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (run_dir / "manifest.json").write_text(json.dumps({
            "run_id": dag_run_id,
            "config": str(run_dir / "config.json"),
            "agent_output": str(run_dir / "run-agent"),
            "preds": str(run_dir / "run-agent" / "preds.json"),
            "trajectories": [str(f) for f in (run_dir / "run-agent").rglob("*.traj.json")],
            "eval_output": str(run_dir / "run-eval"),
            "eval_report": next((str(f) for f in eval_out.glob("*.json")), None),
            "metrics": str(run_dir / "metrics.json"),
        }, indent=2))

        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
        mlflow.set_experiment("evaluate_agent")
        with mlflow.start_run(run_name=dag_run_id):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(run_dir / "config.json"))
            mlflow.log_artifact(str(run_dir / "metrics.json"))

    r = prepare_run()
    r >> run_agent >> run_eval >> summarize_and_log()


evaluate_agent()
