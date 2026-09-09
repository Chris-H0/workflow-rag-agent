# Optimising Agentic AI Workflows Through Compile-Time LLM Assignment Project

This repo implements a profile-guide compile-time approach to assign LLMs to stages of agentic workflows.

It contains:

- **Question Answer (QA) workflow** (`workflows/qa/`): Workflow to answers HotpotQA questions.
- **Code workflow** (`workflows/code/`): Workflow to answers MBPP Python code questions.
- **Placement compiler** (`placement_compiler/`): Routing approach that proposes and evaluates model assignments for each workflow.
- **Experiment results** (`placement_results/`): Results used in experimentation and evaluation.

## Setup

Use Python 3.13 and `uv`. Run all commands from the repo root.

```bash
uv sync
cp .env.example .env
```

Add your OpenAI API key to `.env`.

The default configuration uses OpenAI models and local Qwen through Ollama. Ensure Ollama is running with `qwen3.5:2b` available:

```bash
ollama pull qwen3.5:2b
```

Model settings are defined in `placement_compiler/examples/model_endpoints.yaml`.

Prepare the datasets before running the workflows or the compiler. If the local data files are not already present:

```bash
uv run python -m workflows.qa.download_data
uv run python -m workflows.code.download_data
```

## Run the compiler

For the QA workflow:

```bash
uv run python -m placement_compiler run \
  --config placement_compiler/examples/qa_pipeline_run.yaml
```

For the code workflow:

```bash
uv run python -m placement_compiler run \
  --config placement_compiler/examples/code_pipeline_run.yaml
```

These configurations evaluate an all-strongest-cloud baseline, then propose and evaluate three model assignments using 10 examples.

Edit the YAML configuration to change the sample size, number of proposals, compiler model, or output directory.

## Run individual workflows

Run a workflow with a fixed model assignment:

```bash
uv run python -m workflows.qa.main
uv run python -m workflows.code.main
```

Each command evaluates 10 examples by default. For a shorter run:

```bash
uv run python -m workflows.qa.main --sample-size 2
uv run python -m workflows.code.main --sample-size 2
```

Edit `ASSIGNMENTS` in the corresponding workflow's `main.py` to choose the model for each stage.

## Results

Individual workflow runs are saved to:

```text
workflow_runs/qa/<timestamp>/
workflow_runs/code/<timestamp>/
```

Compiler outputs use the directory specified by `output` in the YAML configuration.
Historical individual workflow results are in `workflows/qa/archive/analysis` and `workflows/code/archive/analysis`.

Evaluated runs produce:

- `results.json`: model assignments and aggregate quality, latency, token usage, and API cost.
- `runs.jsonl`: individual example outputs, scores, errors, and model-call records.

Existing experiment results are stored in subdirectories of `placement_results/qa/` and `placement_results/code/`.