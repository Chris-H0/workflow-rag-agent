from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    build_generate_tests,
    build_implement_solution,
    build_review_solution,
    build_understand_and_plan,
    decide_after_review,
    run_tests,
)
from agent.state import CodeState
from paths import WORKFLOW_ROOT


def build_graph(model_resolver, runtime=None):
    understand_and_plan = build_understand_and_plan(model_resolver.get_model("understand_and_plan"))
    implement_solution = build_implement_solution(model_resolver.get_model("implement_solution"))
    generate_tests = build_generate_tests(model_resolver.get_model("generate_tests"))
    review_solution = build_review_solution(model_resolver.get_model("review_solution"))

    workflow = StateGraph(CodeState)
    workflow.add_node(understand_and_plan)
    workflow.add_node(implement_solution)
    workflow.add_node(generate_tests)
    workflow.add_node(run_tests)
    workflow.add_node(review_solution)

    workflow.add_edge(START, "understand_and_plan")
    workflow.add_edge("understand_and_plan", "implement_solution")
    workflow.add_edge("implement_solution", "generate_tests")
    workflow.add_edge("generate_tests", "run_tests")
    workflow.add_edge("run_tests", "review_solution")
    workflow.add_conditional_edges(
        "review_solution",
        decide_after_review,
        {
            "implement_solution": "implement_solution",
            "__end__": END,
        },
    )

    return workflow.compile()


def save_graph_image(graph, filename: str = "agent_graph.png"):
    path = WORKFLOW_ROOT / filename
    path.write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Saved graph image to {path}")
