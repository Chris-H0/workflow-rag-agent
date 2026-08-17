"""LangChain-backed compiler LLM adapter for structured candidate output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from placement_compiler.core.models import PlacementProposal, StructuredOutputMethod
from placement_compiler.core.usage import calculate_api_cost, extract_token_usage
from placement_compiler.generation.candidates import CompilerGeneration


class LangChainCompilerLLM:
    """Compiler LLM implementation using the repository's LangChain pattern."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        structured_output_method: StructuredOutputMethod = "function_calling",
        input_cost_per_million_tokens: float | None = None,
        output_cost_per_million_tokens: float | None = None,
        **model_kwargs: Any,
    ):
        from langchain.chat_models import init_chat_model

        self._model = init_chat_model(
            model,
            model_provider=provider,
            **model_kwargs,
        )
        self._structured_output_method = structured_output_method
        self._input_cost_per_million_tokens = input_cost_per_million_tokens
        self._output_cost_per_million_tokens = output_cost_per_million_tokens

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        output_schema: type[PlacementProposal],
    ) -> CompilerGeneration:
        try:
            structured = self._model.with_structured_output(
                output_schema,
                method=self._structured_output_method,
                include_raw=True,
            )
        except TypeError:
            structured = self._model.with_structured_output(
                output_schema,
                include_raw=True,
            )
        result = structured.invoke(list(messages))
        raw_result = result.get("raw") if isinstance(result, Mapping) else result
        usage = extract_token_usage(raw_result)
        api_cost, cost_known = calculate_api_cost(
            usage,
            input_cost_per_million_tokens=self._input_cost_per_million_tokens,
            output_cost_per_million_tokens=self._output_cost_per_million_tokens,
        )

        parsed = result.get("parsed") if isinstance(result, Mapping) else result
        parsing_error = (
            result.get("parsing_error") if isinstance(result, Mapping) else None
        )
        error = str(parsing_error) if parsing_error is not None else None
        if error is None:
            try:
                parsed = output_schema.model_validate(parsed)
            except Exception as exc:
                error = f"invalid structured LLM output: {exc}"

        return CompilerGeneration(
            output=parsed,
            prompt_tokens=usage["input_tokens"],
            completion_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            api_cost=api_cost,
            cost_known=cost_known,
            error=error,
        )
