"""Fixed-candidate profiling, ranking, and artifact generation."""

from placement_compiler.profiling.config import load_profile_run_config
from placement_compiler.profiling.runner import (
    CandidateProfiler,
    EvaluationAdapter,
    RuntimeAdapter,
    aggregate_candidate_profile,
    apply_quality_constraints,
    build_baseline_plan,
    deterministic_sample,
    load_candidate_artifact,
    rank_candidates,
    run_record_from_execution,
    write_profile_artifacts,
)

__all__ = [
    "CandidateProfiler",
    "EvaluationAdapter",
    "RuntimeAdapter",
    "aggregate_candidate_profile",
    "apply_quality_constraints",
    "build_baseline_plan",
    "deterministic_sample",
    "load_candidate_artifact",
    "load_profile_run_config",
    "rank_candidates",
    "run_record_from_execution",
    "write_profile_artifacts",
]

