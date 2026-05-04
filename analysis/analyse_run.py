import json
import shutil
from datetime import datetime
from html import escape
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import pandas as pd


CONFIG_ID = "v0-tests-haiku-4-5"
ANALYSIS_ROOT = "analysis"
OVERWRITE = True


RUNS_CSV = "runs.csv"
COMPONENTS_CSV = "components.csv"
SUMMARY_JSON = "summary.json"
REPORT_HTML = "report.html"

GRAPH_FILES = [
    "latency_by_run.png",
    "latency_breakdown_by_run.png",
    "latency_breakdown_by_type.png",
    "latency_breakdown_by_node.png",
    "correctness_vs_latency.png",
    "tokens_vs_latency.png",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def parse_time(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def duration_seconds(start_time, end_time):
    start = parse_time(start_time)
    end = parse_time(end_time)
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def to_float(value):
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def metadata_for(run):
    return (run.get("extra") or {}).get("metadata") or {}


def percentile(values, pct):
    clean_values = sorted(value for value in values if pd.notna(value))
    if not clean_values:
        return None
    index = (len(clean_values) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(clean_values) - 1)
    weight = index - lower
    return clean_values[lower] * (1 - weight) + clean_values[upper] * weight


def metric_average(values):
    clean_values = [value for value in values if pd.notna(value)]
    if not clean_values:
        return None
    return float(mean(clean_values))


def prepare_output_dir(output_dir: Path):
    if output_dir.exists() and OVERWRITE:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def validate_input_dirs(config_dir: Path):
    traces_dir = config_dir / "traces"
    evals_dir = config_dir / "evals"

    if not traces_dir.exists():
        raise FileNotFoundError(f"Missing traces directory: {traces_dir}")
    if not evals_dir.exists():
        raise FileNotFoundError(f"Missing evals directory: {evals_dir}")

    trace_files = sorted(traces_dir.glob("*.json"))
    eval_files = sorted(evals_dir.glob("*.json"))

    if not trace_files:
        raise FileNotFoundError(f"No trace JSON files found in {traces_dir}")
    if not eval_files:
        raise FileNotFoundError(f"No eval JSON files found in {evals_dir}")

    return trace_files, eval_files


def load_evals(eval_files):
    eval_rows = []
    eval_by_run_id = {}
    warnings = []

    for path in eval_files:
        data = read_json(path)
        run_id = data.get("run_id")
        if not run_id:
            warnings.append(f"Eval file has no run_id: {path}")
            continue
        if run_id in eval_by_run_id:
            warnings.append(f"Duplicate eval run_id found: {run_id}")

        metrics = data.get("metrics") or {}
        row = {
            "run_id": run_id,
            "config_id": data.get("config_id"),
            "example_id": data.get("example_id"),
            "repeat": data.get("repeat"),
            "question": data.get("question"),
            "gold_answer": data.get("gold_answer"),
            "agent_answer": data.get("agent_answer"),
            "supporting_titles": "; ".join(data.get("supporting_titles") or []),
            "retrieved_titles": "; ".join(data.get("retrieved_titles") or []),
            "retrieval_rounds": data.get("retrieval_rounds"),
            "nodes": " -> ".join(data.get("nodes") or []),
            "model_config": json.dumps(data.get("model_config") or {}, sort_keys=True),
            "eval_file": str(path),
        }
        for metric_name, metric_value in metrics.items():
            row[metric_name] = metric_value

        eval_rows.append(row)
        eval_by_run_id[run_id] = row

    return eval_by_run_id, pd.DataFrame(eval_rows), warnings


def flatten_traces(trace_files):
    trace_rows = []
    component_rows = []
    trace_by_run_id = {}
    warnings = []

    for path in trace_files:
        data = read_json(path)
        root_run = data.get("root_run") or {}
        runs = data.get("runs") or []
        root_metadata = metadata_for(root_run)
        run_id = root_metadata.get("run_id")
        trace_id = data.get("trace_id") or root_run.get("trace_id")

        if not run_id:
            warnings.append(f"Trace file has no root metadata.run_id: {path}")
            continue
        if run_id in trace_by_run_id:
            warnings.append(f"Duplicate trace run_id found: {run_id}")

        root_duration = duration_seconds(root_run.get("start_time"), root_run.get("end_time"))
        if root_duration is None:
            warnings.append(f"Root trace is missing start/end time for run_id={run_id}")

        root_status = root_run.get("status")
        root_error = root_run.get("error")
        if root_status != "success" or root_error:
            warnings.append(
                f"Root trace has non-success status for run_id={run_id}: "
                f"status={root_status}, error={root_error}"
            )

        trace_row = {
            "run_id": run_id,
            "trace_id": trace_id,
            "root_run_id": data.get("root_run_id") or root_run.get("id"),
            "root_name": root_run.get("name"),
            "root_status": root_status,
            "root_error": root_error,
            "root_start_time": root_run.get("start_time"),
            "root_end_time": root_run.get("end_time"),
            "total_latency_s": root_duration,
            "prompt_tokens": root_run.get("prompt_tokens") or 0,
            "completion_tokens": root_run.get("completion_tokens") or 0,
            "total_tokens": root_run.get("total_tokens") or 0,
            "total_cost": to_float(root_run.get("total_cost")),
            "trace_file": str(path),
        }
        trace_rows.append(trace_row)
        trace_by_run_id[run_id] = trace_row

        for run in runs:
            metadata = metadata_for(run)
            component_duration = duration_seconds(run.get("start_time"), run.get("end_time"))
            if component_duration is None:
                warnings.append(
                    f"Component is missing start/end time for run_id={run_id}: "
                    f"{run.get('run_type')} {run.get('name')} {run.get('id')}"
                )

            status = run.get("status")
            error = run.get("error")
            if status and status != "success" or error:
                warnings.append(
                    f"Component has non-success status for run_id={run_id}: "
                    f"{run.get('run_type')} {run.get('name')} status={status}, error={error}"
                )

            component_rows.append(
                {
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "component_id": run.get("id"),
                    "parent_run_id": run.get("parent_run_id"),
                    "name": run.get("name"),
                    "run_type": run.get("run_type"),
                    "langgraph_node": metadata.get("langgraph_node"),
                    "langgraph_step": metadata.get("langgraph_step"),
                    "start_time": run.get("start_time"),
                    "end_time": run.get("end_time"),
                    "duration_s": component_duration,
                    "status": status,
                    "error": error,
                    "prompt_tokens": run.get("prompt_tokens") or 0,
                    "completion_tokens": run.get("completion_tokens") or 0,
                    "total_tokens": run.get("total_tokens") or 0,
                    "total_cost": to_float(run.get("total_cost")),
                    "model_name": metadata.get("ls_model_name") or metadata.get("model_name"),
                    "model_provider": metadata.get("ls_provider"),
                    "dotted_order": run.get("dotted_order"),
                }
            )

    return (
        trace_by_run_id,
        pd.DataFrame(trace_rows),
        pd.DataFrame(component_rows),
        warnings,
    )


def join_runs(eval_df: pd.DataFrame, trace_df: pd.DataFrame, warnings):
    if eval_df.empty:
        raise ValueError("No eval rows were loaded.")
    if trace_df.empty:
        raise ValueError("No trace rows were loaded.")

    eval_ids = set(eval_df["run_id"])
    trace_ids = set(trace_df["run_id"])
    missing_traces = sorted(eval_ids - trace_ids)
    missing_evals = sorted(trace_ids - eval_ids)

    if missing_traces:
        warnings.append(f"Eval rows without matching trace: {missing_traces}")
    if missing_evals:
        warnings.append(f"Trace rows without matching eval: {missing_evals}")

    runs_df = eval_df.merge(trace_df, on="run_id", how="outer", suffixes=("_eval", "_trace"))
    correctness_columns = [
        "exact_match",
        "contains_gold_answer",
        "contains_partial_gold_answer",
        "supporting_title_hit",
        "supporting_title_recall",
    ]
    for column in correctness_columns:
        if column not in runs_df:
            runs_df[column] = pd.NA

    runs_df["is_correct"] = runs_df["exact_match"].fillna(False).astype(bool)
    runs_df["latency_rank"] = runs_df["total_latency_s"].rank(method="min", ascending=False)
    return runs_df


def top_level_node_components(components_df: pd.DataFrame):
    if components_df.empty:
        return components_df
    root_ids = set(
        components_df.loc[
            components_df["parent_run_id"].isna() | (components_df["name"] == "LangGraph"),
            "component_id",
        ]
    )
    return components_df[
        (components_df["run_type"] == "chain")
        & (components_df["parent_run_id"].isin(root_ids))
        & (components_df["name"] != "LangGraph")
    ].copy()


def summarise(runs_df: pd.DataFrame, components_df: pd.DataFrame, warnings):
    latencies = runs_df["total_latency_s"].dropna().tolist()
    metric_columns = [
        "exact_match",
        "contains_gold_answer",
        "contains_partial_gold_answer",
        "supporting_title_hit",
        "supporting_title_recall",
    ]
    metrics = {}
    for column in metric_columns:
        if column in runs_df:
            metrics[column] = metric_average(pd.to_numeric(runs_df[column], errors="coerce"))

    status_counts = (
        runs_df["root_status"].fillna("missing").value_counts().to_dict()
        if "root_status" in runs_df
        else {}
    )
    run_type_counts = (
        components_df["run_type"].fillna("unknown").value_counts().to_dict()
        if not components_df.empty
        else {}
    )
    model_configs = (
        runs_df["model_config"].dropna().value_counts().head(5).to_dict()
        if "model_config" in runs_df
        else {}
    )

    return {
        "config_id": CONFIG_ID,
        "run_count": int(len(runs_df)),
        "eval_count": int(runs_df["eval_file"].notna().sum()) if "eval_file" in runs_df else 0,
        "trace_count": int(runs_df["trace_file"].notna().sum()) if "trace_file" in runs_df else 0,
        "component_count": int(len(components_df)),
        "latency_s": {
            "mean": mean(latencies) if latencies else None,
            "median": median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
        },
        "correctness": metrics,
        "root_status_counts": status_counts,
        "component_run_type_counts": run_type_counts,
        "model_config_counts": model_configs,
        "warnings": warnings,
    }


def save_figure(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def short_run_labels(runs_df: pd.DataFrame):
    labels = []
    for _, row in runs_df.iterrows():
        example_id = str(row.get("example_id") or "unknown")
        repeat = row.get("repeat")
        labels.append(f"{example_id[-6:]} r{repeat}")
    return labels


def plot_latency_by_run(runs_df: pd.DataFrame, output_dir: Path):
    data = runs_df.sort_values("total_latency_s", ascending=False).reset_index(drop=True)
    colours = data["is_correct"].map({True: "#2ca02c", False: "#d62728"}).tolist()
    plt.figure(figsize=(11, 5))
    plt.bar(short_run_labels(data), data["total_latency_s"], color=colours)
    plt.ylabel("Latency (s)")
    plt.xlabel("Example / repeat")
    plt.title("Total latency by run")
    plt.xticks(rotation=45, ha="right")
    save_figure(output_dir / "latency_by_run.png")


def plot_latency_breakdown_by_run(runs_df: pd.DataFrame, top_nodes_df: pd.DataFrame, output_dir: Path):
    if top_nodes_df.empty:
        return
    node_order = [
        "generate_query_or_respond",
        "retrieve",
        "rewrite_question",
        "generate_followup_query",
        "generate_answer",
    ]
    pivot = top_nodes_df.pivot_table(
        index="run_id",
        columns="name",
        values="duration_s",
        aggfunc="sum",
        fill_value=0,
    )
    ordered_runs = runs_df.sort_values("total_latency_s", ascending=False)["run_id"].tolist()
    pivot = pivot.reindex(ordered_runs).fillna(0)
    columns = [column for column in node_order if column in pivot.columns]
    columns += [column for column in pivot.columns if column not in columns]
    labels = short_run_labels(runs_df.set_index("run_id").loc[pivot.index].reset_index())

    plt.figure(figsize=(12, 6))
    bottom = pd.Series([0.0] * len(pivot), index=pivot.index)
    for column in columns:
        plt.bar(labels, pivot[column], bottom=bottom, label=column)
        bottom += pivot[column]
    plt.ylabel("Latency (s)")
    plt.xlabel("Example / repeat")
    plt.title("Top-level LangGraph node latency by run")
    plt.xticks(rotation=45, ha="right")
    plt.legend(fontsize=8)
    save_figure(output_dir / "latency_breakdown_by_run.png")


def plot_latency_breakdown_by_type(components_df: pd.DataFrame, output_dir: Path):
    nested = components_df[components_df["name"] != "LangGraph"].copy()
    summary = nested.groupby("run_type", dropna=False)["duration_s"].sum().sort_values()
    plt.figure(figsize=(8, 4.8))
    plt.barh(summary.index.fillna("unknown"), summary.values, color="#1f77b4")
    plt.xlabel("Summed component duration (s)")
    plt.title("Nested component duration by run type")
    save_figure(output_dir / "latency_breakdown_by_type.png")


def plot_latency_breakdown_by_node(components_df: pd.DataFrame, output_dir: Path):
    nested = components_df[components_df["name"] != "LangGraph"].copy()
    nested["langgraph_node"] = nested["langgraph_node"].fillna("unknown")
    summary = nested.groupby("langgraph_node")["duration_s"].sum().sort_values()
    plt.figure(figsize=(9, 5))
    plt.barh(summary.index, summary.values, color="#ff7f0e")
    plt.xlabel("Summed component duration (s)")
    plt.title("Nested component duration by LangGraph node")
    save_figure(output_dir / "latency_breakdown_by_node.png")


def plot_correctness_vs_latency(runs_df: pd.DataFrame, output_dir: Path):
    colours = runs_df["is_correct"].map({True: "#2ca02c", False: "#d62728"}).tolist()
    plt.figure(figsize=(8, 5))
    plt.scatter(runs_df["total_latency_s"], runs_df["supporting_title_recall"], c=colours, s=80)
    plt.xlabel("Total latency (s)")
    plt.ylabel("Supporting title recall")
    plt.ylim(-0.05, 1.05)
    plt.title("Correctness vs latency")
    save_figure(output_dir / "correctness_vs_latency.png")


def plot_tokens_vs_latency(runs_df: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.scatter(runs_df["total_tokens"], runs_df["total_latency_s"], c="#1f77b4", s=80)
    plt.xlabel("Total tokens")
    plt.ylabel("Total latency (s)")
    plt.title("Tokens vs latency")
    save_figure(output_dir / "tokens_vs_latency.png")


def create_plots(runs_df: pd.DataFrame, components_df: pd.DataFrame, output_dir: Path):
    top_nodes_df = top_level_node_components(components_df)
    plot_latency_by_run(runs_df, output_dir)
    plot_latency_breakdown_by_run(runs_df, top_nodes_df, output_dir)
    plot_latency_breakdown_by_type(components_df, output_dir)
    plot_latency_breakdown_by_node(components_df, output_dir)
    plot_correctness_vs_latency(runs_df, output_dir)
    plot_tokens_vs_latency(runs_df, output_dir)


def dataframe_table(df: pd.DataFrame, columns, max_rows=20):
    available_columns = [column for column in columns if column in df.columns]
    if not available_columns:
        return "<p>No data available.</p>"
    return df[available_columns].head(max_rows).to_html(index=False, escape=True)


def key_value_table(values):
    rows = []
    for key, value in values.items():
        rows.append(f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>")
    return f"<table>{''.join(rows)}</table>"


def model_config_table(runs_df: pd.DataFrame):
    if "model_config" not in runs_df:
        return "<p>No model configuration found in eval data.</p>"

    configs = [value for value in runs_df["model_config"].dropna().unique() if value]
    if not configs:
        return "<p>No model configuration found in eval data.</p>"

    rows = []
    seen = set()
    for config_json in configs:
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError:
            continue

        for node_name, node_config in sorted(config.items()):
            key = json.dumps({"node": node_name, **node_config}, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "node": node_name,
                    "provider": node_config.get("provider"),
                    "model": node_config.get("model"),
                    "temperature": node_config.get("temperature"),
                }
            )

    if not rows:
        return "<p>No parseable model configuration found in eval data.</p>"

    return pd.DataFrame(rows).to_html(index=False, escape=True)


def observed_model_table(components_df: pd.DataFrame):
    if components_df.empty:
        return "<p>No traced model calls found.</p>"

    observed = components_df[components_df["run_type"] == "llm"].copy()
    if observed.empty:
        return "<p>No traced model calls found.</p>"

    observed["langgraph_node"] = observed["langgraph_node"].fillna("unknown")
    observed["model_provider"] = observed["model_provider"].fillna("unknown")
    observed["model_name"] = observed["model_name"].fillna("unknown")
    return (
        observed.groupby(["langgraph_node", "model_provider", "model_name"], as_index=False)
        .agg(
            calls=("component_id", "count"),
            duration_s=("duration_s", "sum"),
            total_tokens=("total_tokens", "sum"),
            total_cost=("total_cost", "sum"),
        )
        .sort_values(["langgraph_node", "calls"], ascending=[True, False])
        .to_html(index=False, escape=True)
    )


def graph_html(output_dir: Path):
    html = []
    for filename in GRAPH_FILES:
        if (output_dir / filename).exists():
            title = filename.removesuffix(".png").replace("_", " ").title()
            html.append(
                f"<section><h2>{escape(title)}</h2>"
                f'<img src="{escape(filename)}" alt="{escape(title)}"></section>'
            )
    return "\n".join(html)


def create_report(
    output_dir: Path,
    runs_df: pd.DataFrame,
    components_df: pd.DataFrame,
    summary,
):
    correct_runs = runs_df[runs_df["exact_match"] == True].sort_values(
        "total_latency_s", ascending=False
    )
    partially_correct_runs = runs_df[
        (runs_df["exact_match"] == False)
        & (
            (runs_df["contains_gold_answer"] == True)
            | (runs_df["contains_partial_gold_answer"] == True)
        )
    ].sort_values("total_latency_s", ascending=False)
    fully_incorrect_runs = runs_df[
        (runs_df["exact_match"] == False)
        & (runs_df["contains_gold_answer"] == False)
        & (runs_df["contains_partial_gold_answer"] == False)
    ].sort_values("total_latency_s", ascending=False)
    run_result_columns = [
        "example_id",
        "repeat",
        "total_latency_s",
        "exact_match",
        "contains_gold_answer",
        "contains_partial_gold_answer",
        "supporting_title_recall",
        "question",
        "agent_answer",
        "gold_answer",
    ]
    llm_by_node = (
        components_df[components_df["run_type"] == "llm"]
        .assign(langgraph_node=lambda df: df["langgraph_node"].fillna("unknown"))
        .groupby("langgraph_node", as_index=False)
        .agg(duration_s=("duration_s", "sum"), total_tokens=("total_tokens", "sum"), total_cost=("total_cost", "sum"))
        .sort_values("duration_s", ascending=False)
    )
    retriever_tool = components_df[
        components_df["run_type"].isin(["tool", "retriever"])
    ].sort_values("duration_s", ascending=False)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(CONFIG_ID)} analysis</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 0.35rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; margin: 8px 0 28px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #d9e2ec; padding: 14px; border-radius: 6px; background: #fbfcfd; }}
    .warning {{ color: #9f580a; }}
    code {{ background: #f0f4f8; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>{escape(CONFIG_ID)} Analysis</h1>
  <p>Generated from <code>traces/*.json</code> and <code>evals/*.json</code>. Root latency is end-to-end LangGraph wall time; nested component duration charts intentionally keep summed child durations separate from root latency.</p>

  <div class="grid">
    <div class="card"><h2>Run Counts</h2>{key_value_table({
        "runs": summary["run_count"],
        "evals": summary["eval_count"],
        "traces": summary["trace_count"],
        "components": summary["component_count"],
    })}</div>
    <div class="card"><h2>Latency Seconds</h2>{key_value_table({
        key: round(value, 3) if value is not None else None
        for key, value in summary["latency_s"].items()
    })}</div>
    <div class="card"><h2>Correctness</h2>{key_value_table({
        key: round(value, 3) if value is not None else None
        for key, value in summary["correctness"].items()
    })}</div>
  </div>

  <h2>Configured Models By Node</h2>
  {model_config_table(runs_df)}

  <h2>Observed LLM Models In Traces</h2>
  {observed_model_table(components_df)}

  <h2>Warnings</h2>
  {"<p>No warnings.</p>" if not summary["warnings"] else "<ul>" + "".join(f'<li class="warning">{escape(warning)}</li>' for warning in summary["warnings"]) + "</ul>"}

  {graph_html(output_dir)}

  <h2>Correct Runs</h2>
  {dataframe_table(correct_runs, run_result_columns, max_rows=50)}

  <h2>Partially Correct Runs</h2>
  {dataframe_table(partially_correct_runs, run_result_columns, max_rows=50)}

  <h2>Fully Incorrect Runs</h2>
  {dataframe_table(fully_incorrect_runs, run_result_columns, max_rows=50)}

  <h2>LLM Latency By Node</h2>
  {dataframe_table(llm_by_node, ["langgraph_node", "duration_s", "total_tokens", "total_cost"])}

  <h2>Tool And Retriever Latency</h2>
  {dataframe_table(retriever_tool, ["run_id", "run_type", "name", "langgraph_node", "duration_s", "status", "error"], max_rows=30)}

  <h2>All Runs</h2>
  {dataframe_table(runs_df.sort_values("total_latency_s", ascending=False), ["example_id", "repeat", "total_latency_s", "total_tokens", "total_cost", "exact_match", "contains_gold_answer", "contains_partial_gold_answer", "supporting_title_recall", "retrieval_rounds", "nodes"], max_rows=50)}
</body>
</html>
"""
    (output_dir / REPORT_HTML).write_text(html, encoding="utf-8")


def main():
    root_dir = Path(__file__).resolve().parents[1]
    analysis_root = root_dir / ANALYSIS_ROOT
    config_dir = analysis_root / CONFIG_ID
    output_dir = config_dir / "analysis"

    trace_files, eval_files = validate_input_dirs(config_dir)
    prepare_output_dir(output_dir)

    eval_by_run_id, eval_df, eval_warnings = load_evals(eval_files)
    trace_by_run_id, trace_df, components_df, trace_warnings = flatten_traces(trace_files)
    warnings = eval_warnings + trace_warnings

    runs_df = join_runs(eval_df, trace_df, warnings)
    summary = summarise(runs_df, components_df, warnings)

    runs_df.to_csv(output_dir / RUNS_CSV, index=False)
    components_df.to_csv(output_dir / COMPONENTS_CSV, index=False)
    write_json(output_dir / SUMMARY_JSON, summary)

    create_plots(runs_df, components_df, output_dir)
    create_report(output_dir, runs_df, components_df, summary)

    print(f"Analysed config ID: {CONFIG_ID}")
    print(f"Eval files: {len(eval_by_run_id)}")
    print(f"Trace files: {len(trace_by_run_id)}")
    print(f"Output: {output_dir}")
    if warnings:
        print(f"Warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
