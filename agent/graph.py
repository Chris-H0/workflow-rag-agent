from pathlib import Path

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.nodes import (
    build_generate_answer,
    build_generate_query_or_respond,
    build_grade_documents,
    build_rewrite_question,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def build_graph(response_model, grader_model, retriever_tool):
    generate_query_or_respond = build_generate_query_or_respond(
        response_model,
        retriever_tool,
    )
    grade_documents = build_grade_documents(grader_model)
    rewrite_question = build_rewrite_question(response_model)
    generate_answer = build_generate_answer(response_model)

    workflow = StateGraph(MessagesState)
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
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
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()


def save_graph_image(graph, filename: str = "agent_graph.png"):
    path = ROOT_DIR / filename
    path.write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Saved graph image to {path}")
