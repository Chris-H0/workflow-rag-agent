"""Lock QA holdout and repeat shortlists from completed development evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from placement_compiler.profiling.models import PlanResult, SearchArtifact


QUALITY_RETENTION = 0.95
PRIMARY_QUALITY_METRIC = "answer_token_f1"
SECONDARY_QUALITY_METRIC = "exact_match"
BASELINE_ENDPOINT = "gpt-5.5-cloud"
LOCAL_ENDPOINT = "qwen-local"


@dataclass(frozen=True)
class CompiledPlacement:
    sequence: int
    proposal: int
    source_plan_id: str
    exhaustive_plan_id: str
    result: PlanResult


def lock_shortlists(result_root: Path) -> dict[str, Any]:
    sequences = [
        _load(result_root / f"compiler-sequence-{number:02d}" / "results.json")
        for number in (1, 2, 3)
    ]
    exhaustive = _load(result_root / "exhaustive" / "results.json")
    _validate_inputs(sequences, exhaustive)

    unit_ids = [
        "generate_query_or_respond",
        "decide_after_retrieval",
        "rewrite_question",
        "generate_answer",
    ]
    exhaustive_by_signature = {
        _signature(result, unit_ids): result for result in exhaustive.results
    }
    baseline = next(
        result
        for result in exhaustive.results
        if all(
            result.plan.assignments[unit_id] == BASELINE_ENDPOINT
            for unit_id in unit_ids
        )
    )
    baseline_quality = _metric(baseline, PRIMARY_QUALITY_METRIC)
    threshold = QUALITY_RETENTION * baseline_quality

    compiled = _compiled_placements(
        sequences,
        exhaustive_by_signature=exhaustive_by_signature,
        unit_ids=unit_ids,
    )
    first_sequence = [item for item in compiled if item.sequence == 1]

    seed_43 = [
        _selection_record(
            baseline,
            label="strongest-cloud reference",
            threshold=threshold,
        )
    ]
    for qwen_inclusive, label in (
        (False, "non-Qwen"),
        (True, "Qwen-inclusive"),
    ):
        category = [
            item
            for item in first_sequence
            if _is_qwen_inclusive(item.result) is qwen_inclusive
        ]
        if len(category) < 2:
            raise ValueError(f"sequence 1 has fewer than two {label} proposals")
        feasible = [
            item for item in category if _quality(item.result) >= threshold
        ]
        best_pool = feasible or category
        best = min(
            best_pool,
            key=lambda item: _quality_rank(item, baseline),
        )
        remaining = [item for item in category if item != best]
        cheapest = min(remaining, key=_cost_rank)
        seed_43.extend(
            [
                _selection_record(
                    best,
                    label=(
                        f"best quality-feasible {label} proposal"
                        if feasible
                        else f"highest-quality {label} negative evidence"
                    ),
                    threshold=threshold,
                ),
                _selection_record(
                    cheapest,
                    label=f"cheapest remaining {label} proposal",
                    threshold=threshold,
                ),
            ]
        )

    earliest_by_signature: dict[tuple[str, ...], CompiledPlacement] = {}
    for item in compiled:
        signature = _signature(item.result, unit_ids)
        earliest_by_signature.setdefault(signature, item)
    unique_compiled = list(earliest_by_signature.values())

    seed_56 = [
        _selection_record(
            baseline,
            label="strongest-cloud reference",
            threshold=threshold,
        )
    ]
    for qwen_inclusive, label in (
        (False, "non-Qwen"),
        (True, "Qwen-inclusive"),
    ):
        category = [
            item
            for item in unique_compiled
            if _is_qwen_inclusive(item.result) is qwen_inclusive
            and len(set(item.result.plan.assignments.values())) > 1
        ]
        if not category:
            raise ValueError(f"no compiled {label} hybrid was proposed")
        feasible = [
            item for item in category if _quality(item.result) >= threshold
        ]
        selected = min(
            feasible or category,
            key=lambda item: _quality_rank(item, baseline),
        )
        seed_56.append(
            _selection_record(
                selected,
                label=(
                    f"best compiled {label} hybrid"
                    if feasible
                    else f"highest-quality compiled {label} negative evidence"
                ),
                threshold=threshold,
            )
        )

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "workflow": "qa",
        "result_root": str(result_root.resolve()),
        "selection_protocol": {
            "quality_retention": QUALITY_RETENTION,
            "primary_quality_metric": PRIMARY_QUALITY_METRIC,
            "secondary_quality_metric": SECONDARY_QUALITY_METRIC,
            "exhaustive_baseline_answer_token_f1": baseline_quality,
            "quality_threshold": threshold,
            "seed_43": "GGGG plus four representative sequence-1 proposals",
            "seed_56": (
                "GGGG plus the best compiled non-Qwen and Qwen-inclusive hybrids "
                "across all three sequences, scored in the exhaustive table"
            ),
        },
        "holdout_seed_43": seed_43,
        "holdout_seed_56_and_repeats": seed_56,
    }


def _load(path: Path) -> SearchArtifact:
    return SearchArtifact.model_validate_json(path.read_text())


def _validate_inputs(
    sequences: list[SearchArtifact], exhaustive: SearchArtifact
) -> None:
    if any(artifact.workflow != "qa" for artifact in [*sequences, exhaustive]):
        raise ValueError("all shortlist inputs must belong to the QA workflow")
    if any(len(artifact.results) != 13 for artifact in sequences):
        raise ValueError("each compiler sequence must contain 13 placements")
    if len(exhaustive.results) != 81:
        raise ValueError("the exhaustive artifact must contain 81 placements")
    expected_ids = exhaustive.selected_example_ids
    if any(artifact.selected_example_ids != expected_ids for artifact in sequences):
        raise ValueError("compiler and exhaustive development samples do not match")


def _compiled_placements(
    sequences: list[SearchArtifact],
    *,
    exhaustive_by_signature: dict[tuple[str, ...], PlanResult],
    unit_ids: list[str],
) -> list[CompiledPlacement]:
    output = []
    for sequence_number, artifact in enumerate(sequences, start=1):
        for result in artifact.results:
            if result.plan.source == "baseline":
                continue
            try:
                proposal = int(result.plan.id.removeprefix("candidate-"))
            except ValueError as exc:
                raise ValueError(f"unexpected compiler plan ID {result.plan.id!r}") from exc
            exhaustive_result = exhaustive_by_signature.get(_signature(result, unit_ids))
            if exhaustive_result is None:
                raise ValueError(
                    f"compiler placement {sequence_number}/{result.plan.id} "
                    "is absent from exhaustive results"
                )
            output.append(
                CompiledPlacement(
                    sequence=sequence_number,
                    proposal=proposal,
                    source_plan_id=result.plan.id,
                    exhaustive_plan_id=exhaustive_result.plan.id,
                    result=exhaustive_result,
                )
            )
    return output


def _signature(result: PlanResult, unit_ids: list[str]) -> tuple[str, ...]:
    return tuple(result.plan.assignments[unit_id] for unit_id in unit_ids)


def _metric(result: PlanResult, name: str) -> float:
    value = result.metrics.get(name)
    if not isinstance(value, int | float):
        raise ValueError(f"placement {result.plan.id} has no numeric {name!r}")
    return float(value)


def _quality(result: PlanResult) -> float:
    return _metric(result, PRIMARY_QUALITY_METRIC)


def _quality_rank(
    item: CompiledPlacement,
    baseline: PlanResult,
) -> tuple[Any, ...]:
    result = item.result
    improves_both = (
        _metric(result, "cloud_api_cost") < _metric(baseline, "cloud_api_cost")
        and _metric(result, "mean_latency_seconds")
        < _metric(baseline, "mean_latency_seconds")
    )
    return (
        not improves_both,
        -_quality(result),
        -_metric(result, SECONDARY_QUALITY_METRIC),
        _metric(result, "cloud_api_cost"),
        _metric(result, "mean_latency_seconds"),
        item.sequence,
        item.proposal,
    )


def _cost_rank(item: CompiledPlacement) -> tuple[Any, ...]:
    return (
        _metric(item.result, "cloud_api_cost"),
        _metric(item.result, "mean_latency_seconds"),
        -_quality(item.result),
        -_metric(item.result, SECONDARY_QUALITY_METRIC),
        item.proposal,
    )


def _is_qwen_inclusive(result: PlanResult) -> bool:
    return LOCAL_ENDPOINT in result.plan.assignments.values()


def _selection_record(
    selected: PlanResult | CompiledPlacement,
    *,
    label: str,
    threshold: float,
) -> dict[str, Any]:
    if isinstance(selected, CompiledPlacement):
        result = selected.result
        source = {
            "compiler_sequence": selected.sequence,
            "compiler_plan_id": selected.source_plan_id,
        }
        exhaustive_plan_id = selected.exhaustive_plan_id
    else:
        result = selected
        source = {"compiler_sequence": None, "compiler_plan_id": None}
        exhaustive_plan_id = result.plan.id
    return {
        "label": label,
        "exhaustive_plan_id": exhaustive_plan_id,
        **source,
        "assignments": result.plan.assignments,
        "quality_feasible": _quality(result) >= threshold,
        "exhaustive_metrics": {
            "answer_token_f1": _quality(result),
            "exact_match": _metric(result, SECONDARY_QUALITY_METRIC),
            "cloud_api_cost": _metric(result, "cloud_api_cost"),
            "mean_latency_seconds": _metric(result, "mean_latency_seconds"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.result_root / "selection.json"
    selection = lock_shortlists(args.result_root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Wrote locked QA shortlists to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
