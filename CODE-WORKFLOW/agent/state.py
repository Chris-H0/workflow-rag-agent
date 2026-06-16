from langgraph.graph import MessagesState


class CodeState(MessagesState):
    task: dict
    solution: str
    test_result: dict
    attempts: int
