from analysis.analyse_run import analyse_run
from agent.graph import build_graph, save_graph_image
from agent.model_router import ModelRouter
from agent.tools import build_retriever_tool
from evals.runner import run_eval_loop
from paths import load_workflow_dotenv
from rag.pipeline import build_retriever_from_documents
from rag.sources import build_hotpotqa_documents, load_hotpotqa_examples


CONFIG_ID = "v3-refactor-test-11"
QUESTIONS = 2
REPEATS = 1

RUN_ANALYSIS = True
SAVE_GRAPH_PNG = False
PRINT_UPDATES = False

NODE_MODEL_CONFIG = {
    "generate_query_or_respond": {
        "provider": "ollama",
        "model": "qwen3.5:2b",
        "temperature": 0,
        "thinking": False,
        "reasoning": False,
    },
    "decide_after_retrieval": {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "temperature": 0,
    },
    "rewrite_question": {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "temperature": 0,
    },
    "generate_answer": {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "temperature": 0,
    },
}


if __name__ == "__main__":
    load_workflow_dotenv(override=True)

    # Load and prepare data
    all_questions = load_hotpotqa_examples()
    documents = build_hotpotqa_documents(all_questions)
    retriever = build_retriever_from_documents(documents)
    eval_questions = all_questions[:QUESTIONS]

    # Build agent
    retriever_tool = build_retriever_tool(retriever)
    model_router = ModelRouter(NODE_MODEL_CONFIG)
    agent_graph = build_graph(model_router, {"retriever_tool": retriever_tool})
    if SAVE_GRAPH_PNG:
        save_graph_image(agent_graph)

    # Run agent + evaluate outputs
    run_eval_loop(
        agent_graph,
        eval_questions,
        all_questions,
        CONFIG_ID,
        REPEATS,
        PRINT_UPDATES,
        model_router.model_config,
    )

    # Run analysis
    if RUN_ANALYSIS:
        analyse_run(CONFIG_ID)
