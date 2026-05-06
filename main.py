from dotenv import load_dotenv

from analysis.analyse_run import analyse_run
from agent.graph import build_graph, save_graph_image
from agent.model_router import ModelRouter
from agent.tools import build_retriever_tool
from evals.runner import run_eval_loop, run_download_traces
from rag.pipeline import build_retriever_from_documents
from rag.sources import build_hotpotqa_documents, load_hotpotqa_examples


CONFIG_ID = "v2-tests-qwen3.5-2b-temp-test-1"
HOTPOTQA_LOAD_LIMIT = 100
HOTPOTQA_LEVEL = "hard"  # Options: "easy", "medium", "hard", "any"
QUESTIONS = 5
REPEATS = 1

DOWNLOAD_TRACES = True
SAVE_GRAPH = False
PRINT_UPDATES = False


def load_env():
    load_dotenv(".env", override=True)


def build_agent(documents, model_router):
    retriever = build_retriever_from_documents(documents)
    retriever_tool = build_retriever_tool(retriever)
    return build_graph(model_router, retriever_tool)


if __name__ == "__main__":
    load_env()

    # Load and prepare data
    all_questions = load_hotpotqa_examples(HOTPOTQA_LOAD_LIMIT, HOTPOTQA_LEVEL)
    documents = build_hotpotqa_documents(all_questions)
    eval_questions = all_questions[:QUESTIONS]
    
    # Build agent
    model_router = ModelRouter()
    agent_graph = build_agent(documents, model_router)
    if SAVE_GRAPH:
        save_graph_image(agent_graph)

    # Run evaluation loop
    run_eval_loop(
        agent_graph,
        eval_questions,
        all_questions,
        CONFIG_ID,
        HOTPOTQA_LEVEL,
        REPEATS,
        PRINT_UPDATES,
        model_router.model_config,
    )
    
    if DOWNLOAD_TRACES:
        run_download_traces(CONFIG_ID)
        analyse_run(CONFIG_ID)