from langchain.tools import tool


def build_retriever_tool(retriever):
    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Search and return information from the indexed documents."""
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    return retrieve_blog_posts
