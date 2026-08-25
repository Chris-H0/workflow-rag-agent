from placement_compiler.core.models import PlacementPlan
from placement_compiler.experiments.qa_shortlist import (
    CompiledPlacement,
    _cost_rank,
    _quality_rank,
)
from placement_compiler.profiling.models import PlanResult


def _result(plan_id: str, *, quality: float, f1: float, cost: float, latency: float):
    return PlanResult(
        plan=PlacementPlan(
            id=plan_id,
            source="candidate",
            assignments={
                "generate_query_or_respond": "gpt-4.1-mini-cloud",
                "decide_after_retrieval": "gpt-4.1-mini-cloud",
                "rewrite_question": "gpt-4.1-mini-cloud",
                "generate_answer": "gpt-5.5-cloud",
            },
        ),
        metrics={
            "exact_match": quality,
            "answer_token_f1": f1,
            "cloud_api_cost": cost,
            "mean_latency_seconds": latency,
        },
    )


def _compiled(result: PlanResult, proposal: int) -> CompiledPlacement:
    return CompiledPlacement(
        sequence=1,
        proposal=proposal,
        source_plan_id=f"candidate-{proposal}",
        exhaustive_plan_id=f"placement-{proposal + 1:03d}",
        result=result,
    )


def test_quality_rank_prefers_improving_both_before_higher_quality():
    baseline = _result("baseline", quality=0.6, f1=0.7, cost=2.0, latency=10.0)
    efficient = _compiled(
        _result("efficient", quality=0.58, f1=0.68, cost=1.0, latency=8.0),
        2,
    )
    higher_quality_but_slower = _compiled(
        _result("slower", quality=0.7, f1=0.8, cost=1.0, latency=11.0),
        1,
    )

    assert min(
        [higher_quality_but_slower, efficient],
        key=lambda item: _quality_rank(item, baseline),
    ) == efficient


def test_cost_rank_selects_cheapest_remaining_proposal():
    cheaper = _compiled(
        _result("cheaper", quality=0.5, f1=0.6, cost=0.1, latency=12.0),
        2,
    )
    better_quality = _compiled(
        _result("quality", quality=0.7, f1=0.8, cost=0.2, latency=6.0),
        1,
    )

    assert min([better_quality, cheaper], key=_cost_rank) == cheaper
