from langchain_community.document_loaders import WebBaseLoader


LILIAN_WENG_URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]


def load_lilian_weng_documents():
    docs = [WebBaseLoader(url).load() for url in LILIAN_WENG_URLS]
    return [item for sublist in docs for item in sublist]
