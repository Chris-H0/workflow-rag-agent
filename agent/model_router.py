from copy import deepcopy

from langchain.chat_models import init_chat_model


# NODE_MODEL_CONFIG = {
#     "generate_query_or_respond": {
#         "provider": "openai",
#         "model": "gpt-5.5",
#     },
#     "decide_after_retrieval": {
#         "provider": "openai",
#         "model": "gpt-5.5",
#     },
#     "rewrite_question": {
#         "provider": "openai",
#         "model": "gpt-5.5",
#     },
#     "generate_answer": {
#         "provider": "openai",
#         "model": "gpt-5.5",
#     },
# }

NODE_MODEL_CONFIG = {
    "generate_query_or_respond": {
        "provider": "anthropic",
        "model": "claude-opus-4-7",
    },
    "decide_after_retrieval": {
        "provider": "anthropic",
        "model": "claude-opus-4-7",
    },
    "rewrite_question": {
        "provider": "anthropic",
        "model": "claude-opus-4-7",
    },
    "generate_answer": {
        "provider": "anthropic",
        "model": "claude-opus-4-7",
    },
}


class ModelRouter:
    def __init__(self):
        self.model_config = NODE_MODEL_CONFIG
        self.models = {}

    def get_model(self, node_name: str):
        if node_name not in self.models:
            self.models[node_name] = self.build_model(self.model_config[node_name])
        return self.models[node_name]

    def model_config(self):
        return deepcopy(self.model_config)

    def build_model(self, config):
        return init_chat_model(
            config["model"],
            model_provider=config["provider"],
        )
