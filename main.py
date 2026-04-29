from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState


def load_env():
    load_dotenv(".env", override=True)


def preprocess_documents():
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    return text_splitter.split_documents(docs_list)


def build_retriever(document_chunks):
    vectorstore = InMemoryVectorStore.from_documents(
        documents=document_chunks,
        embedding=OpenAIEmbeddings(),
    )
    return vectorstore.as_retriever()


def build_retriever_tool(retriever):
    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Search and return information about Lilian Weng blog posts."""
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    return retrieve_blog_posts


def build_response_model():
    return init_chat_model("gpt-5.4", temperature=0)


def build_generate_query_or_respond(response_model, retriever_tool):
    def generate_query_or_respond(state: MessagesState):
        """Either answer directly or call the retriever tool."""
        response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}

    return generate_query_or_respond


if __name__ == "__main__":
    load_env()
    document_chunks = preprocess_documents()
    retriever = build_retriever(document_chunks)
    retriever_tool = build_retriever_tool(retriever)
    response_model = build_response_model()
    generate_query_or_respond = build_generate_query_or_respond(
        response_model,
        retriever_tool,
    )

    greeting_input: MessagesState = {"messages": [HumanMessage(content="hello!")]}

    greeting_response = generate_query_or_respond(greeting_input)

    greeting_response["messages"][-1].pretty_print()

    retrieval_input: MessagesState = {
        "messages": [
            HumanMessage(
                content="What does Lilian Weng say about types of reward hacking?"
            )
        ]
    }

    retrieval_response = generate_query_or_respond(retrieval_input)
    retrieval_response["messages"][-1].pretty_print()
