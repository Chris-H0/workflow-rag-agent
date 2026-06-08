from copy import deepcopy


POSITION_AWARE_POLICY = "position_aware"
DOWNSTREAM_IMPACT_POLICY = "downstream_impact"
GRAPH_CENTRALITY_POLICY = DOWNSTREAM_IMPACT_POLICY
FINAL_ANSWER_NODE = "generate_answer"
DEFAULT_DOWNSTREAM_IMPACT_CLOUD_THRESHOLD = 5

WORKFLOW_NODE_METADATA = {
    "generate_query_or_respond": {
        "name": "generate_query_or_respond",
        "role": "initial_query_generation_or_direct_response",
        "stage": "early",
        "is_langgraph_node": True,
        "calls_llm": True,
        "user_facing": False,
        "controls_branching": True,
        "can_retry_or_loop": True,
        "can_end_workflow": True,
        "edges_in": ["START", "rewrite_question", "generate_followup_query"],
        "edges_out": ["retrieve", "END"],
        "downstream_nodes": [
            "retrieve",
            "decide_after_retrieval",
            "generate_followup_query",
            "rewrite_question",
            "generate_answer",
        ],
    },
    "retrieve": {
        "name": "retrieve",
        "role": "document_retrieval",
        "stage": "middle",
        "is_langgraph_node": True,
        "calls_llm": False,
        "user_facing": False,
        "controls_branching": False,
        "can_retry_or_loop": False,
        "can_end_workflow": False,
        "edges_in": ["generate_query_or_respond"],
        "edges_out": ["decide_after_retrieval"],
        "downstream_nodes": [
            "decide_after_retrieval",
            "generate_followup_query",
            "rewrite_question",
            "generate_query_or_respond",
            "generate_answer",
        ],
    },
    "decide_after_retrieval": {
        "name": "decide_after_retrieval",
        "role": "retrieval_decision",
        "stage": "middle",
        "is_langgraph_node": False,
        "calls_llm": True,
        "user_facing": False,
        "controls_branching": True,
        "can_retry_or_loop": True,
        "can_end_workflow": False,
        "edges_in": ["retrieve"],
        "edges_out": [
            "generate_answer",
            "generate_followup_query",
            "rewrite_question",
        ],
        "downstream_nodes": [
            "generate_answer",
            "generate_followup_query",
            "rewrite_question",
            "generate_query_or_respond",
            "retrieve",
        ],
    },
    "rewrite_question": {
        "name": "rewrite_question",
        "role": "question_rewriting",
        "stage": "middle",
        "is_langgraph_node": True,
        "calls_llm": True,
        "user_facing": False,
        "controls_branching": False,
        "can_retry_or_loop": True,
        "can_end_workflow": False,
        "edges_in": ["decide_after_retrieval"],
        "edges_out": ["generate_query_or_respond"],
        "downstream_nodes": [
            "generate_query_or_respond",
            "retrieve",
            "decide_after_retrieval",
            "generate_followup_query",
            "generate_answer",
        ],
    },
    "generate_followup_query": {
        "name": "generate_followup_query",
        "role": "followup_query_prompting",
        "stage": "middle",
        "is_langgraph_node": True,
        "calls_llm": False,
        "user_facing": False,
        "controls_branching": False,
        "can_retry_or_loop": True,
        "can_end_workflow": False,
        "edges_in": ["decide_after_retrieval"],
        "edges_out": ["generate_query_or_respond"],
        "downstream_nodes": [
            "generate_query_or_respond",
            "retrieve",
            "decide_after_retrieval",
            "rewrite_question",
            "generate_answer",
        ],
    },
    "generate_answer": {
        "name": "generate_answer",
        "role": "final_answer_generation",
        "stage": "terminal",
        "is_langgraph_node": True,
        "calls_llm": True,
        "user_facing": True,
        "controls_branching": False,
        "can_retry_or_loop": False,
        "can_end_workflow": True,
        "edges_in": ["decide_after_retrieval"],
        "edges_out": ["END"],
        "downstream_nodes": [],
    },
}

DEFAULT_POSITION_AWARE_PLACEMENT = {
    "generate_query_or_respond": "local",
    "decide_after_retrieval": "local",
    "rewrite_question": "local",
    "generate_answer": "cloud",
}


def build_position_aware_model_config(model_profiles, node_placement=None):
    placements = deepcopy(DEFAULT_POSITION_AWARE_PLACEMENT)
    if node_placement:
        placements.update(node_placement)

    placements[FINAL_ANSWER_NODE] = "cloud"

    model_config = {}
    for node_name, placement in placements.items():
        if placement not in model_profiles:
            raise ValueError(f"No model profile configured for placement: {placement}")

        node_config = deepcopy(model_profiles[placement])
        node_config["placement"] = placement
        node_config["placement_policy"] = POSITION_AWARE_POLICY
        model_config[node_name] = node_config

    return model_config


def downstream_llm_nodes(node_metadata, registry):
    return [
        node_name
        for node_name in node_metadata["downstream_nodes"]
        if registry.get(node_name, {}).get("calls_llm")
    ]


def downstream_impact_score(node_metadata, registry):
    score = len(downstream_llm_nodes(node_metadata, registry))
    if node_metadata["controls_branching"]:
        score += 2
    if node_metadata["can_retry_or_loop"]:
        score += 1
    if node_metadata["user_facing"]:
        score += 3
    return score


def downstream_impact_placement(node_metadata, registry, cloud_threshold):
    score = downstream_impact_score(node_metadata, registry)

    if node_metadata["user_facing"]:
        return "cloud", score, "user-facing node"

    if score >= cloud_threshold:
        return "cloud", score, f"downstream impact score >= {cloud_threshold}"

    return "local", score, f"downstream impact score < {cloud_threshold}"


def build_downstream_impact_model_config(
    model_profiles,
    node_metadata=None,
    cloud_threshold=DEFAULT_DOWNSTREAM_IMPACT_CLOUD_THRESHOLD,
):
    registry = node_metadata or WORKFLOW_NODE_METADATA
    model_config = {}

    for node_name, metadata in registry.items():
        if not metadata["calls_llm"]:
            continue

        placement, score, reason = downstream_impact_placement(
            metadata,
            registry,
            cloud_threshold,
        )
        if placement not in model_profiles:
            raise ValueError(f"No model profile configured for placement: {placement}")

        node_config = deepcopy(model_profiles[placement])
        node_config["placement"] = placement
        node_config["placement_policy"] = DOWNSTREAM_IMPACT_POLICY
        node_config["downstream_impact_score"] = score
        node_config["placement_reason"] = reason
        model_config[node_name] = node_config

    return model_config


def build_graph_centrality_model_config(
    model_profiles,
    node_metadata=None,
    cloud_threshold=DEFAULT_DOWNSTREAM_IMPACT_CLOUD_THRESHOLD,
):
    return build_downstream_impact_model_config(
        model_profiles,
        node_metadata,
        cloud_threshold,
    )
