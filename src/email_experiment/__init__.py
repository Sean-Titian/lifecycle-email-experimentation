"""Public, privacy-safe tools for lifecycle email experimentation."""

from .assignment import (
    ACTIVE_ARMS,
    ALL_ARMS,
    HOLDOUT_ARM,
    assignment_sha256,
    audit_sample_ratio,
    stratified_factorial_assignment,
)
from .contracts import ContractError, one_to_one_join, validate_unique_key
from .decision import (
    REQUIRED_QUALITY_GATES,
    DecisionResult,
    evaluate_guardrail_family,
    evaluate_primary_family,
    make_launch_decision,
)
from .funnel import (
    new_link_funnel_membership,
    ordered_funnel_membership,
    summarize_ordered_funnel,
    unsubscribe_metrics,
)
from .prospective import (
    ProspectiveSyntheticConfig,
    ProspectiveSyntheticData,
    audit_contamination,
    audit_followup,
    build_prospective_synthetic_benchmark,
    construct_prospective_analysis,
    generate_prospective_synthetic,
    validate_events,
    write_prospective_synthetic_benchmark,
)
from .statistics import (
    BinaryEffect,
    adjust_pvalues,
    approximate_mde,
    compare_binary_proportions,
    compare_experiment_groups,
)
from .synthetic import (
    SyntheticExperimentConfig,
    SyntheticExperimentData,
    generate_synthetic_experiment,
)
from .time_windows import censoring_summary, construct_windowed_outcome

__version__ = "0.3.0"

__all__ = [
    "ACTIVE_ARMS",
    "ALL_ARMS",
    "BinaryEffect",
    "ContractError",
    "DecisionResult",
    "HOLDOUT_ARM",
    "ProspectiveSyntheticConfig",
    "ProspectiveSyntheticData",
    "REQUIRED_QUALITY_GATES",
    "SyntheticExperimentConfig",
    "SyntheticExperimentData",
    "__version__",
    "adjust_pvalues",
    "approximate_mde",
    "assignment_sha256",
    "audit_contamination",
    "audit_followup",
    "audit_sample_ratio",
    "build_prospective_synthetic_benchmark",
    "censoring_summary",
    "compare_binary_proportions",
    "compare_experiment_groups",
    "construct_prospective_analysis",
    "construct_windowed_outcome",
    "evaluate_guardrail_family",
    "evaluate_primary_family",
    "generate_prospective_synthetic",
    "generate_synthetic_experiment",
    "make_launch_decision",
    "new_link_funnel_membership",
    "one_to_one_join",
    "ordered_funnel_membership",
    "stratified_factorial_assignment",
    "summarize_ordered_funnel",
    "unsubscribe_metrics",
    "validate_unique_key",
    "validate_events",
    "write_prospective_synthetic_benchmark",
]
