from analysis.analyse_run import analyse_run
from agent.graph import build_graph, save_graph_image
from agent.model_router import ModelRouter
from benchmarks.mbpp_plus import load_mbpp_plus_examples
from evals.runner import run_eval_loop
from paths import load_workflow_dotenv


CONFIG_ID = "v2-qwen3.5-2b-gpt-5.5"
TASKS = 75
REPEATS = 1

RUN_ANALYSIS = True
SAVE_GRAPH_PNG = False
PRINT_UPDATES = False

NODE_MODEL_CONFIG = {
    "understand_and_plan": {
        "provider": "openai",
        "model": "gpt-5.5",
        "temperature": 0,
    },
    "implement_solution": {
        "provider": "ollama",
        "model": "qwen3.5:2b",
        "temperature": 0,
        "thinking": False,
        "reasoning": False,
        "num_predict": 256,
    },
    "generate_tests": {
        "provider": "ollama",
        "model": "qwen3.5:2b",
        "temperature": 0,
        "thinking": False,
        "reasoning": False,
        "num_predict": 256,
    },
    "review_solution": {
        "provider": "openai",
        "model": "gpt-5.5",
        "temperature": 0,
    },
}

        # "provider": "ollama",
        # "model": "qwen3.5:2b",
        # "temperature": 0,
        # "thinking": False,
        # "reasoning": False,
        # "num_predict": 256,

if __name__ == "__main__":
    load_workflow_dotenv(override=True)

    # Load and prepare data
    all_examples = load_mbpp_plus_examples()
    eval_examples = all_examples[:TASKS]

    # Build agent
    model_router = ModelRouter(NODE_MODEL_CONFIG)
    agent_graph = build_graph(model_router)
    if SAVE_GRAPH_PNG:
        save_graph_image(agent_graph)

    # Run agent + evaluate outputs
    run_eval_loop(
        agent_graph,
        eval_examples,
        all_examples,
        CONFIG_ID,
        REPEATS,
        PRINT_UPDATES,
        model_router.model_config,
    )

    # Run analysis
    if RUN_ANALYSIS:
        analyse_run(CONFIG_ID)
