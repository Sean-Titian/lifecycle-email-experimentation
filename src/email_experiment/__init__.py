"""Public, privacy-safe tools for lifecycle email experimentation."""

from .contracts import ContractError, one_to_one_join, validate_unique_key
from .funnel import (
    new_link_funnel_membership,
    ordered_funnel_membership,
    summarize_ordered_funnel,
    unsubscribe_metrics,
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

__all__ = [
    "BinaryEffect",
    "ContractError",
    "SyntheticExperimentConfig",
    "SyntheticExperimentData",
    "adjust_pvalues",
    "approximate_mde",
    "censoring_summary",
    "compare_binary_proportions",
    "compare_experiment_groups",
    "construct_windowed_outcome",
    "generate_synthetic_experiment",
    "new_link_funnel_membership",
    "one_to_one_join",
    "ordered_funnel_membership",
    "summarize_ordered_funnel",
    "unsubscribe_metrics",
    "validate_unique_key",
]
