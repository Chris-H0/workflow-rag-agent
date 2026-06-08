import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from analysis.analyse_run import analyse_run
from agent.graph import build_graph
from agent.model_router import ModelRouter
from agent.placement import (
    build_boundary_io_model_config,
    build_branching_terminal_model_config,
    build_decision_terminal_model_config,
    build_entry_strong_terminal_model_config,
    build_fast_terminal_only_model_config,
    build_internal_control_strong_terminal_model_config,
    build_recovery_terminal_model_config,
    build_tiered_control_terminal_model_config,
    build_terminal_only_model_config,
)
from agent.tools import build_retriever_tool
from evals.runner import run_download_traces, run_eval_loop
from rag.pipeline import build_retriever_from_documents
from rag.sources import build_hotpotqa_documents, load_hotpotqa_examples


HOTPOTQA_LOAD_LIMIT = 100
HOTPOTQA_LEVEL = "hard"
QUESTIONS = 75
REPEATS = 1
PRINT_UPDATES = False

BASELINE_CONFIGS = [
    {
        "config_id": "v3-gpt-5.5",
        "label": "Cloud only",
        "description": "All LLM nodes use GPT-5.5.",
    },
    {
        "config_id": "v3-qwen3.5-2b",
        "label": "Local only",
        "description": "All LLM nodes use qwen3.5:2b locally.",
    },
    {
        "config_id": "v3-gpt-4.1-mini",
        "label": "Cloud only mini",
        "description": "All LLM nodes use GPT-4.1-mini.",
    },
]

GPT_5_5_PROFILES = {
    "local": {
        "provider": "ollama",
        "model": "qwen3.5:2b",
        "temperature": 0,
        "thinking": False,
        "reasoning": False,
    },
    "cloud": {
        "provider": "openai",
        "model": "gpt-5.5",
        "temperature": 0,
    },
}

GPT_4_1_MINI_PROFILES = {
    "local": GPT_5_5_PROFILES["local"],
    "cloud": {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "temperature": 0,
    },
}

TIERED_PROFILES = {
    "local": GPT_5_5_PROFILES["local"],
    "cloud_fast": GPT_4_1_MINI_PROFILES["cloud"],
    "cloud_strong": GPT_5_5_PROFILES["cloud"],
}

POLICY_EXPERIMENTS = [
    {
        "config_id": "v3-policy-terminal-only-qwen3.5-2b-gpt-5.5",
        "label": "Terminal only",
        "description": (
            "Use GPT-5.5 only for user-facing terminal synthesis; run all support "
            "LLM nodes locally."
        ),
        "builder": build_terminal_only_model_config,
        "model_profiles": GPT_5_5_PROFILES,
    },
    {
        "config_id": "v3-policy-decision-terminal-qwen3.5-2b-gpt-5.5",
        "label": "Decision + terminal",
        "description": (
            "Use GPT-5.5 for retrieval decision-making and final synthesis; run "
            "query generation and rewriting locally."
        ),
        "builder": build_decision_terminal_model_config,
        "model_profiles": GPT_5_5_PROFILES,
    },
    {
        "config_id": "v3-policy-branching-terminal-qwen3.5-2b-gpt-5.5",
        "label": "Branching + terminal",
        "description": (
            "Use GPT-5.5 for LLM nodes that control branching plus final synthesis; "
            "run non-branching rewrite support locally."
        ),
        "builder": build_branching_terminal_model_config,
        "model_profiles": GPT_5_5_PROFILES,
    },
    {
        "config_id": "v3-policy-boundary-io-qwen3.5-2b-gpt-4.1-mini",
        "label": "Boundary I/O",
        "description": (
            "Use GPT-4.1-mini for workflow boundary LLM nodes connected to START "
            "or END; run internal support nodes locally."
        ),
        "builder": build_boundary_io_model_config,
        "model_profiles": GPT_4_1_MINI_PROFILES,
    },
    {
        "config_id": "v3-policy-recovery-terminal-qwen3.5-2b-gpt-5.5",
        "label": "Recovery + terminal",
        "description": (
            "Use GPT-5.5 for recovery/rewrite paths and terminal synthesis; run "
            "normal query generation and decision support locally."
        ),
        "builder": build_recovery_terminal_model_config,
        "model_profiles": GPT_5_5_PROFILES,
    },
    {
        "config_id": "v3-policy-tiered-control-terminal-qwen3.5-2b-gpt-4.1-mini-gpt-5.5",
        "label": "Tiered control + terminal",
        "description": (
            "Use GPT-4.1-mini for branching/control nodes, GPT-5.5 for terminal "
            "synthesis, and local model for non-control support."
        ),
        "builder": build_tiered_control_terminal_model_config,
        "model_profiles": TIERED_PROFILES,
    },
    {
        "config_id": "v3-policy-fast-terminal-only-qwen3.5-2b-gpt-4.1-mini",
        "label": "Fast terminal only",
        "description": (
            "Use GPT-4.1-mini only for user-facing terminal synthesis; run all "
            "support LLM nodes locally."
        ),
        "builder": build_fast_terminal_only_model_config,
        "model_profiles": TIERED_PROFILES,
    },
    {
        "config_id": "v3-policy-entry-strong-terminal-qwen3.5-2b-gpt-4.1-mini-gpt-5.5",
        "label": "Entry + strong terminal",
        "description": (
            "Use GPT-4.1-mini for workflow entry nodes, GPT-5.5 for terminal "
            "synthesis, and local model for internal support."
        ),
        "builder": build_entry_strong_terminal_model_config,
        "model_profiles": TIERED_PROFILES,
    },
    {
        "config_id": "v3-policy-internal-control-strong-terminal-qwen3.5-2b-gpt-4.1-mini-gpt-5.5",
        "label": "Internal control + terminal",
        "description": (
            "Use GPT-4.1-mini for internal workflow control nodes, GPT-5.5 for "
            "terminal synthesis, and local model for boundary/support nodes."
        ),
        "builder": build_internal_control_strong_terminal_model_config,
        "model_profiles": TIERED_PROFILES,
    },
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def analysis_complete(config_id: str):
    summary_path = Path("analysis") / config_id / "analysis" / "summary.json"
    if not summary_path.exists():
        return False

    summary = read_json(summary_path)
    return (
        summary.get("eval_count") == QUESTIONS
        and summary.get("trace_count") == QUESTIONS
        and summary.get("questions", {}).get("hotpotqa_level") == HOTPOTQA_LEVEL
    )


def run_policy_experiments():
    load_dotenv(".env", override=True)

    all_questions = load_hotpotqa_examples(HOTPOTQA_LOAD_LIMIT, HOTPOTQA_LEVEL)
    documents = build_hotpotqa_documents(all_questions)
    retriever = build_retriever_from_documents(documents)
    retriever_tool = build_retriever_tool(retriever)
    eval_questions = all_questions[:QUESTIONS]

    for experiment in POLICY_EXPERIMENTS:
        config_id = experiment["config_id"]
        if analysis_complete(config_id):
            print(f"Skipping completed config ID: {config_id}")
            continue

        model_config = experiment["builder"](experiment["model_profiles"])
        model_router = ModelRouter(model_config)
        agent_graph = build_graph(model_router, retriever_tool)

        print(f"Running config ID: {config_id}")
        run_eval_loop(
            agent_graph,
            eval_questions,
            all_questions,
            config_id,
            HOTPOTQA_LEVEL,
            REPEATS,
            PRINT_UPDATES,
            model_router.model_config,
        )
        run_download_traces(config_id)
        analyse_run(config_id)


def total_cost(runs_csv: Path):
    runs_df = pd.read_csv(runs_csv)
    return float(runs_df["total_cost"].fillna(0).sum())


def observed_models(config_id: str):
    components_csv = Path("analysis") / config_id / "analysis" / "components.csv"
    components_df = pd.read_csv(components_csv)
    llm_df = components_df[components_df["run_type"] == "llm"].copy()
    if llm_df.empty:
        return ""

    observed = (
        llm_df.groupby(["model_provider", "model_name"], dropna=False)
        .size()
        .reset_index(name="calls")
        .sort_values(["model_provider", "model_name"])
    )
    return "; ".join(
        f"{row.model_provider}/{row.model_name}: {row.calls}"
        for row in observed.itertuples()
    )


def comparison_row(config, experiment_type):
    config_id = config["config_id"]
    analysis_dir = Path("analysis") / config_id / "analysis"
    summary = read_json(analysis_dir / "summary.json")
    correctness = summary["correctness"]
    latency = summary["latency_s"]

    return {
        "config_id": config_id,
        "type": experiment_type,
        "label": config["label"],
        "description": config["description"],
        "eval_count": summary["eval_count"],
        "trace_count": summary["trace_count"],
        "exact_match": correctness.get("exact_match"),
        "contains_gold_answer": correctness.get("contains_gold_answer"),
        "contains_partial_gold_answer": correctness.get("contains_partial_gold_answer"),
        "answer_token_f1": correctness.get("answer_token_f1"),
        "supporting_title_recall": correctness.get("supporting_title_recall"),
        "mean_latency_s": latency.get("mean"),
        "p95_latency_s": latency.get("p95"),
        "total_cost": total_cost(analysis_dir / "runs.csv"),
        "observed_models": observed_models(config_id),
        "warnings": len(summary.get("warnings") or []),
    }


def write_comparison():
    rows = []
    for config in BASELINE_CONFIGS:
        rows.append(comparison_row(config, "baseline"))
    for config in POLICY_EXPERIMENTS:
        rows.append(comparison_row(config, "policy"))

    output_dir = Path("analysis") / "policy-comparison-expanded"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = pd.DataFrame(rows)
    comparison_df["cost_vs_cloud"] = (
        comparison_df["total_cost"] / comparison_df.loc[0, "total_cost"]
    )
    comparison_df["cost_savings_vs_cloud"] = 1 - comparison_df["cost_vs_cloud"]
    comparison_df["exact_match_vs_cloud"] = (
        comparison_df["exact_match"] / comparison_df.loc[0, "exact_match"]
    )
    comparison_df["exact_matches"] = comparison_df["exact_match"] * comparison_df["eval_count"]
    comparison_df["cost_per_exact_match"] = (
        comparison_df["total_cost"] / comparison_df["exact_matches"]
    ).where(comparison_df["exact_matches"] > 0, 0)

    comparison_df.to_csv(output_dir / "comparison.csv", index=False)
    write_json(output_dir / "comparison.json", comparison_df.to_dict(orient="records"))
    (output_dir / "comparison.html").write_text(
        comparison_df.to_html(index=False, escape=True),
        encoding="utf-8",
    )
    write_summary_markdown(comparison_df, output_dir / "summary.md")
    print(f"Wrote comparison to {output_dir}")
    print(comparison_df.to_string(index=False))


def write_summary_markdown(comparison_df, path: Path):
    rows = comparison_df.sort_values("total_cost")
    lines = [
        "# Expanded Policy Comparison",
        "",
        "All runs use 75 hard HotpotQA questions. Local calls use `qwen3.5:2b`; cloud calls use `gpt-5.5`, `gpt-4.1-mini`, or both.",
        "",
        "| Policy | Type | Description | Exact Match | Answer F1 | Supporting Recall | Mean Latency | P95 Latency | Total Cost | Cost vs Cloud | Cost / Exact |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows.itertuples():
        lines.append(
            "| "
            f"{row.label} | "
            f"{row.type} | "
            f"{row.description} | "
            f"{row.exact_match:.3f} | "
            f"{row.answer_token_f1:.3f} | "
            f"{row.supporting_title_recall:.3f} | "
            f"{row.mean_latency_s:.2f}s | "
            f"{row.p95_latency_s:.2f}s | "
            f"{row.total_cost:.5f} | "
            f"{row.cost_vs_cloud:.3f} | "
            f"{row.cost_per_exact_match:.5f} |"
        )

    best_policy = comparison_df[comparison_df["type"] == "policy"].sort_values(
        ["cost_per_exact_match", "total_cost"]
    ).iloc[0]
    lines.extend(
        [
            "",
            "## Best Cost-Effective Router",
            "",
            (
                f"`{best_policy.label}` is the best policy by cost per exact match: "
                f"exact match {best_policy.exact_match:.3f}, total cost "
                f"{best_policy.total_cost:.5f}, cost vs cloud "
                f"{best_policy.cost_vs_cloud:.3f}."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_policy_experiments()
    write_comparison()
