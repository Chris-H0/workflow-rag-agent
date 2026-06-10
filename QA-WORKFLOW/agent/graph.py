from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from paths import WORKFLOW_ROOT
from agent.nodes import (
    build_decide_after_retrieval,
    build_generate_answer,
    build_generate_followup_query,
    build_generate_query_or_respond,
    build_rewrite_question,
)


def build_graph(model_router, retriever_tool):
    generate_query_or_respond = build_generate_query_or_respond(model_router.get_model("generate_query_or_respond"),retriever_tool,)
    decide_after_retrieval = build_decide_after_retrieval(model_router.get_model("decide_after_retrieval"))
    rewrite_question = build_rewrite_question(model_router.get_model("rewrite_question"))
    generate_followup_query = build_generate_followup_query()
    generate_answer = build_generate_answer(model_router.get_model("generate_answer"))

    workflow = StateGraph(MessagesState)
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_followup_query)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_conditional_edges("retrieve", decide_after_retrieval)
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    workflow.add_edge("generate_followup_query", "generate_query_or_respond")

    return workflow.compile()


def save_graph_image(graph, filename: str = "agent_graph.png"):
    path = WORKFLOW_ROOT / filename
    path.write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Saved graph image to {path}")
