from copy import deepcopy

from langchain.chat_models import init_chat_model


class ModelRouter:
    def __init__(self, node_model_config):
        self.model_config = node_model_config
        self.models = {}

    def get_model(self, node_name: str):
        if node_name not in self.models:
            self.models[node_name] = self.build_model(self.model_config[node_name])
        return self.models[node_name]

    def model_config(self):
        return deepcopy(self.model_config)

    def build_model(self, config):
        required_keys = {"provider", "model"}
        model_kwargs = {
            key: value for key, value in config.items() if key not in required_keys
        }

        return init_chat_model(
            config["model"],
            model_provider=config["provider"],
            **model_kwargs,
        )
