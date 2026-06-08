from copy import deepcopy


POSITION_AWARE_POLICY = "position_aware"
FINAL_ANSWER_NODE = "generate_answer"

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
