from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from placement_compiler.models import CandidateSetDraft


class LangChainCompilerLLM:
    """Compiler LLM implementation using the repository's LangChain pattern."""

    def __init__(self, *, provider: str, model: str, **model_kwargs: Any):
        from langchain.chat_models import init_chat_model

        self._model = init_chat_model(
            model,
            model_provider=provider,
            **model_kwargs,
        )

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        output_schema: type[CandidateSetDraft],
    ) -> CandidateSetDraft:
        structured = self._model.with_structured_output(output_schema)
        result = structured.invoke(list(messages))
        return output_schema.model_validate(result)
