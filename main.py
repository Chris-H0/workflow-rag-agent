from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from agent.graph import build_graph, save_graph_image
from agent.tools import build_retriever_tool
from evals.runner import run_eval_loop
from rag.pipeline import build_retriever_from_documents
from rag.sources import build_hotpotqa_documents, load_hotpotqa_examples


CONFIG_ID = "v0-tests"
QUESTION_LIMIT = 5
REPEATS = 1
SAVE_GRAPH = False
PRINT_UPDATES = True


def load_env():
    load_dotenv(".env", override=True)


def build_response_model():
    return init_chat_model("gpt-5.4", temperature=0)


def build_agent(documents):
    retriever = build_retriever_from_documents(documents)
    retriever_tool = build_retriever_tool(retriever)
    response_model = build_response_model()
    grader_model = build_response_model()

    return build_graph(response_model, grader_model, retriever_tool)


if __name__ == "__main__":
    load_env()

    examples = load_hotpotqa_examples()
    documents = build_hotpotqa_documents(examples)
    eval_examples = examples[:QUESTION_LIMIT]

    agent_graph = build_agent(documents)
    if SAVE_GRAPH:
        save_graph_image(agent_graph)

    run_eval_loop(agent_graph, eval_examples, CONFIG_ID, REPEATS, PRINT_UPDATES)
