from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from agent.graph import build_graph
from agent.tools import build_retriever_tool
from rag.pipeline import build_retriever_from_documents
from rag.sources import load_lilian_weng_documents


CONFIG_ID = "v0-tests"
QUESTION = "What does Lilian Weng say about types of reward hacking?"


def load_env():
    load_dotenv(".env", override=True)


def build_response_model():
    return init_chat_model("gpt-5.4", temperature=0)


def build_agent():
    load_env()
    documents = load_lilian_weng_documents()
    retriever = build_retriever_from_documents(documents)
    retriever_tool = build_retriever_tool(retriever)
    response_model = build_response_model()
    grader_model = build_response_model()

    return build_graph(response_model, grader_model, retriever_tool)


def stream_agent_response(graph, question: str, config_id: str):
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": question}]},
        config={"metadata": {"config_id": config_id}},
    ):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n")


if __name__ == "__main__":
    agent_graph = build_agent()
    stream_agent_response(agent_graph, QUESTION, CONFIG_ID)
