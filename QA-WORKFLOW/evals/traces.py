import json
import os
from pathlib import Path
from time import sleep

from langsmith import Client

from paths import analysis_config_dir, load_workflow_dotenv


RETRY_DELAY_SECONDS = 5


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def metadata_for(run):
    return (run.get("extra") or {}).get("metadata") or {}


def existing_trace_path(config_id: str, run_id: str):
    traces_dir = analysis_config_dir(config_id) / "traces"
    if not traces_dir.exists():
        return None

    for path in traces_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        root_metadata = metadata_for(data.get("root_run") or {})
        if root_metadata.get("run_id") == run_id:
            return path

    return None


def root_run_matches(root_run, config_id: str, run_id: str) -> bool:
    metadata = root_run.extra.get("metadata") if root_run.extra else {}
    return metadata.get("config_id") == config_id and metadata.get("run_id") == run_id


def download_trace_once(client, project_name: str, config_id: str, run_id: str):
    existing_path = existing_trace_path(config_id, run_id)
    if existing_path:
        print(f"Trace already downloaded at {existing_path}")
        return existing_path

    filter_query = f'and(eq(metadata_key, "run_id"), eq(metadata_value, "{run_id}"))'
    root_runs = list(
        client.list_runs(
            filter=filter_query,
            is_root=True,
            limit=1,
            project_name=project_name,
        )
    )

    if not root_runs:
        return None

    root_run = root_runs[0]
    if not root_run_matches(root_run, config_id, run_id):
        return None
    if root_run.status == "pending" or not root_run.end_time:
        return None

    trace_id = root_run.trace_id
    output_path = analysis_config_dir(config_id) / "traces" / f"{trace_id}.json"
    if output_path.exists():
        print(f"Trace already downloaded at {output_path}")
        return output_path

    trace_runs = list(client.list_runs(project_name=project_name, trace_id=trace_id))
    trace_runs.sort(key=lambda run: run.dotted_order)

    write_json(
        output_path,
        {
            "trace_id": str(trace_id),
            "root_run_id": str(root_run.id),
            "root_run": root_run.model_dump(mode="json"),
            "runs": [run.model_dump(mode="json") for run in trace_runs],
        },
    )
    print(f"Saved trace to {output_path}")
    return output_path


def download_trace(config_id: str, run_id: str):
    load_workflow_dotenv(override=True)
    project_name = os.getenv("LANGSMITH_PROJECT")
    last_error = None

    for attempt in range(2):
        try:
            output_path = download_trace_once(
                Client(),
                project_name,
                config_id,
                run_id,
            )
            if output_path:
                return output_path
        except Exception as error:
            last_error = error

        if attempt == 0:
            sleep(RETRY_DELAY_SECONDS)

    message = f"Traces are not downloading for config_id={config_id}, run_id={run_id}"
    if last_error:
        raise RuntimeError(message) from last_error
    raise RuntimeError(message)
