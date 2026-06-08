from copy import deepcopy


POSITION_AWARE_POLICY = "position_aware"
DOWNSTREAM_IMPACT_POLICY = "downstream_impact"
GRAPH_CENTRALITY_POLICY = DOWNSTREAM_IMPACT_POLICY
TERMINAL_ONLY_POLICY = "terminal_only"
DECISION_TERMINAL_POLICY = "decision_terminal"
BRANCHING_TERMINAL_POLICY = "branching_terminal"
BOUNDARY_IO_POLICY = "boundary_io"
RECOVERY_TERMINAL_POLICY = "recovery_terminal"
TIERED_CONTROL_TERMINAL_POLICY = "tiered_control_terminal"
FAST_TERMINAL_ONLY_POLICY = "fast_terminal_only"
ENTRY_STRONG_TERMINAL_POLICY = "entry_strong_terminal"
INTERNAL_CONTROL_STRONG_TERMINAL_POLICY = "internal_control_strong_terminal"
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


def llm_node_metadata(registry):
    return {
        node_name: metadata
        for node_name, metadata in registry.items()
        if metadata["calls_llm"]
    }


def build_metadata_policy_model_config(
    model_profiles,
    policy_name,
    placement_reasons,
):
    model_config = {}

    for node_name, placement_reason in placement_reasons.items():
        placement, reason = placement_reason
        if placement not in model_profiles:
            raise ValueError(f"No model profile configured for placement: {placement}")

        node_config = deepcopy(model_profiles[placement])
        node_config["placement"] = placement
        node_config["placement_policy"] = policy_name
        node_config["placement_reason"] = reason
        model_config[node_name] = node_config

    return model_config


def terminal_only_placement_reasons(node_metadata):
    placement_reasons = {}
    for node_name, metadata in llm_node_metadata(node_metadata).items():
        if metadata["user_facing"] or metadata["stage"] == "terminal":
            placement_reasons[node_name] = (
                "cloud",
                "user-facing or terminal synthesis node",
            )
        else:
            placement_reasons[node_name] = ("local", "support node")
    return placement_reasons


def decision_terminal_placement_reasons(node_metadata):
    placement_reasons = {}
    for node_name, metadata in llm_node_metadata(node_metadata).items():
        if metadata["user_facing"] or metadata["stage"] == "terminal":
            placement_reasons[node_name] = (
                "cloud",
                "user-facing or terminal synthesis node",
            )
        elif metadata["role"] == "retrieval_decision":
            placement_reasons[node_name] = (
                "cloud",
                "retrieval decision controls downstream execution",
            )
        else:
            placement_reasons[node_name] = ("local", "non-terminal support node")
    return placement_reasons


def branching_terminal_placement_reasons(node_metadata):
    placement_reasons = {}
    for node_name, metadata in llm_node_metadata(node_metadata).items():
        if metadata["user_facing"] or metadata["stage"] == "terminal":
            placement_reasons[node_name] = (
                "cloud",
                "user-facing or terminal synthesis node",
            )
        elif metadata["controls_branching"]:
            placement_reasons[node_name] = (
                "cloud",
                "branching node controls downstream execution",
            )
        else:
            placement_reasons[node_name] = ("local", "non-branching support node")
    return placement_reasons


def boundary_io_placement_reasons(node_metadata):
    placement_reasons = {}
    for node_name, metadata in llm_node_metadata(node_metadata).items():
        if "START" in metadata["edges_in"] or "END" in metadata["edges_out"]:
            placement_reasons[node_name] = (
                "cloud",
                "workflow boundary node connected to START or END",
            )
        else:
            placement_reasons[node_name] = ("local", "internal support node")
    return placement_reasons


def recovery_terminal_placement_reasons(node_metadata):
    placement_reasons = {}
    for node_name, metadata in llm_node_metadata(node_metadata).items():
        if metadata["user_facing"] or metadata["stage"] == "terminal":
            placement_reasons[node_name] = (
                "cloud",
                "user-facing or terminal synthesis node",
            )
        elif metadata["role"] == "question_rewriting":
            placement_reasons[node_name] = (
                "cloud",
                "recovery node repairs failed retrieval context",
            )
        else:
            placement_reasons[node_name] = ("local", "normal support node")
    return placement_reasons


def tiered_control_terminal_placement_reasons(node_metadata):
    placement_reasons = {}
    for node_name, metadata in llm_node_metadata(node_metadata).items():
        if metadata["user_facing"] or metadata["stage"] == "terminal":
            placement_reasons[node_name] = (
                "cloud_strong",
                "strong cloud for user-facing terminal synthesis",
            )
        elif metadata["controls_branching"]:
            placement_reasons[node_name] = (
                "cloud_fast",
                "fast cloud for workflow control node",
            )
        else:
            placement_reasons[node_name] = ("local", "non-control support node")
    return placement_reasons


def fast_terminal_only_placement_reasons(node_metadata):
    placement_reasons = {}
    for node_name, metadata in llm_node_metadata(node_metadata).items():
        if metadata["user_facing"] or metadata["stage"] == "terminal":
            placement_reasons[node_name] = (
                "cloud_fast",
                "fast cloud for user-facing terminal synthesis",
            )
        else:
            placement_reasons[node_name] = ("local", "support node")
    return placement_reasons


def entry_strong_terminal_placement_reasons(node_metadata):
    placement_reasons = {}
    for node_name, metadata in llm_node_metadata(node_metadata).items():
        if metadata["user_facing"] or metadata["stage"] == "terminal":
            placement_reasons[node_name] = (
                "cloud_strong",
                "strong cloud for user-facing terminal synthesis",
            )
        elif "START" in metadata["edges_in"]:
            placement_reasons[node_name] = (
                "cloud_fast",
                "fast cloud for workflow entry node",
            )
        else:
            placement_reasons[node_name] = ("local", "internal support node")
    return placement_reasons


def internal_control_strong_terminal_placement_reasons(node_metadata):
    placement_reasons = {}
    for node_name, metadata in llm_node_metadata(node_metadata).items():
        is_boundary_node = "START" in metadata["edges_in"] or "END" in metadata["edges_out"]
        if metadata["user_facing"] or metadata["stage"] == "terminal":
            placement_reasons[node_name] = (
                "cloud_strong",
                "strong cloud for user-facing terminal synthesis",
            )
        elif metadata["controls_branching"] and not is_boundary_node:
            placement_reasons[node_name] = (
                "cloud_fast",
                "fast cloud for internal workflow control node",
            )
        else:
            placement_reasons[node_name] = ("local", "boundary or support node")
    return placement_reasons


def build_terminal_only_model_config(
    model_profiles,
    node_metadata=None,
):
    registry = node_metadata or WORKFLOW_NODE_METADATA
    return build_metadata_policy_model_config(
        model_profiles,
        TERMINAL_ONLY_POLICY,
        terminal_only_placement_reasons(registry),
    )


def build_decision_terminal_model_config(
    model_profiles,
    node_metadata=None,
):
    registry = node_metadata or WORKFLOW_NODE_METADATA
    return build_metadata_policy_model_config(
        model_profiles,
        DECISION_TERMINAL_POLICY,
        decision_terminal_placement_reasons(registry),
    )


def build_branching_terminal_model_config(
    model_profiles,
    node_metadata=None,
):
    registry = node_metadata or WORKFLOW_NODE_METADATA
    return build_metadata_policy_model_config(
        model_profiles,
        BRANCHING_TERMINAL_POLICY,
        branching_terminal_placement_reasons(registry),
    )


def build_boundary_io_model_config(
    model_profiles,
    node_metadata=None,
):
    registry = node_metadata or WORKFLOW_NODE_METADATA
    return build_metadata_policy_model_config(
        model_profiles,
        BOUNDARY_IO_POLICY,
        boundary_io_placement_reasons(registry),
    )


def build_recovery_terminal_model_config(
    model_profiles,
    node_metadata=None,
):
    registry = node_metadata or WORKFLOW_NODE_METADATA
    return build_metadata_policy_model_config(
        model_profiles,
        RECOVERY_TERMINAL_POLICY,
        recovery_terminal_placement_reasons(registry),
    )


def build_tiered_control_terminal_model_config(
    model_profiles,
    node_metadata=None,
):
    registry = node_metadata or WORKFLOW_NODE_METADATA
    return build_metadata_policy_model_config(
        model_profiles,
        TIERED_CONTROL_TERMINAL_POLICY,
        tiered_control_terminal_placement_reasons(registry),
    )


def build_fast_terminal_only_model_config(
    model_profiles,
    node_metadata=None,
):
    registry = node_metadata or WORKFLOW_NODE_METADATA
    return build_metadata_policy_model_config(
        model_profiles,
        FAST_TERMINAL_ONLY_POLICY,
        fast_terminal_only_placement_reasons(registry),
    )


def build_entry_strong_terminal_model_config(
    model_profiles,
    node_metadata=None,
):
    registry = node_metadata or WORKFLOW_NODE_METADATA
    return build_metadata_policy_model_config(
        model_profiles,
        ENTRY_STRONG_TERMINAL_POLICY,
        entry_strong_terminal_placement_reasons(registry),
    )


def build_internal_control_strong_terminal_model_config(
    model_profiles,
    node_metadata=None,
):
    registry = node_metadata or WORKFLOW_NODE_METADATA
    return build_metadata_policy_model_config(
        model_profiles,
        INTERNAL_CONTROL_STRONG_TERMINAL_POLICY,
        internal_control_strong_terminal_placement_reasons(registry),
    )


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
