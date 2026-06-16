from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    build_debug_solution,
    build_generate_solution,
    decide_after_tests,
    run_visible_tests,
)
from agent.state import CodeState
from paths import WORKFLOW_ROOT


def build_graph(model_router):
    generate_solution = build_generate_solution(model_router.get_model("generate_solution"))
    debug_solution = build_debug_solution(model_router.get_model("debug_solution"))

    workflow = StateGraph(CodeState)
    workflow.add_node(generate_solution)
    workflow.add_node(run_visible_tests)
    workflow.add_node(debug_solution)

    workflow.add_edge(START, "generate_solution")
    workflow.add_edge("generate_solution", "run_visible_tests")
    workflow.add_conditional_edges(
        "run_visible_tests",
        decide_after_tests,
        {
            "debug_solution": "debug_solution",
            "__end__": END,
        },
    )
    workflow.add_edge("debug_solution", "run_visible_tests")

    return workflow.compile()


def save_graph_image(graph, filename: str = "agent_graph.png"):
    path = WORKFLOW_ROOT / filename
    path.write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Saved graph image to {path}")
