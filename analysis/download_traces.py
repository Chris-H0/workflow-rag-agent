import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client


CONFIG_ID = "v0-test-ollama-llama3.2"
LIMIT = 75


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def download_traces(config_id: str, limit: int):
    load_dotenv(".env", override=True)
    project_name = os.getenv("LANGSMITH_PROJECT")

    filter_query = (
        f'and(eq(metadata_key, "config_id"), eq(metadata_value, "{config_id}"))'
    )

    try:
        client = Client()
        analysis_dir = Path(__file__).resolve().parent
        output_dir = analysis_dir / config_id / "traces"
        output_dir.mkdir(parents=True, exist_ok=True)

        root_run_kwargs = {
            "filter": filter_query,
            "is_root": True,
            "limit": limit,
            "project_name": project_name,
        }

        count = 0
        for root_run in client.list_runs(**root_run_kwargs):
            trace_id = root_run.trace_id
            output_path = output_dir / f"{trace_id}.json"

            if output_path.exists():
                continue

            trace_runs = list(
                client.list_runs(project_name=project_name, trace_id=trace_id)
            )
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
            count += 1

        print(f"Downloaded {count} traces for config ID: {config_id} to {output_dir}.")
    except Exception as error:
        raise RuntimeError(f"Failed to download traces for config ID: {config_id}") from error


if __name__ == "__main__":
    download_traces(CONFIG_ID, LIMIT)
