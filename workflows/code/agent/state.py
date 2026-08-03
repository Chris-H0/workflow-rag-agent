from langgraph.graph import MessagesState


class CodeState(MessagesState):
    task: dict
    plan: str
    solution: str
    generated_tests: str
    generated_test_result: dict
    review_comments: str
    review_decision: str
    revision_count: int
