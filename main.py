from dotenv import load_dotenv

from agent.graph import build_graph, save_graph_image
from agent.model_router import ModelRouter
from agent.tools import build_retriever_tool
from evals.runner import run_eval_loop
from rag.pipeline import build_retriever_from_documents
from rag.sources import build_hotpotqa_documents, load_hotpotqa_examples


CONFIG_ID = "v0-tests-opus-4-7"
QUESTION_LIMIT = 50
REPEATS = 1
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

    examples = load_hotpotqa_examples()
    documents = build_hotpotqa_documents(examples)
    eval_examples = examples[:QUESTION_LIMIT]
    model_router = ModelRouter()

    agent_graph = build_agent(documents, model_router)
    if SAVE_GRAPH:
        save_graph_image(agent_graph)

    run_eval_loop(
        agent_graph,
        eval_examples,
        CONFIG_ID,
        REPEATS,
        PRINT_UPDATES,
        model_router.model_config,
    )
