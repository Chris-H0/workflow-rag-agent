from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size: int = 100, chunk_overlap: int = 50):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def build_retriever(document_chunks):
    vectorstore = InMemoryVectorStore.from_documents(
        documents=document_chunks,
        embedding=OpenAIEmbeddings(),
    )
    return vectorstore.as_retriever()


def build_retriever_from_documents(documents):
    document_chunks = split_documents(documents)
    return build_retriever(document_chunks)
