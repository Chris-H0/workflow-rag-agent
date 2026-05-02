from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from agent.graph import build_graph, save_graph_image
from agent.tools import build_retriever_tool
from rag.pipeline import build_retriever_from_documents
from rag.sources import build_hotpotqa_documents, load_hotpotqa_examples


CONFIG_ID = "v0-tests"
SAVE_GRAPH = False


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


def stream_agent_response(graph, question: str, config_id: str):
    user_message = HumanMessage(content=question)
    user_message.pretty_print()
    print("\n")

    for chunk in graph.stream(
        {"messages": [user_message]},
        config={"metadata": {"config_id": config_id}},
    ):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n")


if __name__ == "__main__":
    load_env()

    examples = load_hotpotqa_examples()
    documents = build_hotpotqa_documents(examples)
    question = examples[0]["question"]

    agent_graph = build_agent(documents)
    if SAVE_GRAPH:
        save_graph_image(agent_graph)

    stream_agent_response(agent_graph, question, CONFIG_ID)
