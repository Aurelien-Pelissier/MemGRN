"""MemGRN: Learning Regulatory Dynamics with Explicit Weights and Adaptive Memory."""

from memgrn.dynamics import (
    MemGRNConfig,
    MemGRNParams,
    RateInputMode,
    adaptive_rate_law,
    compute_exact_contributions,
    memory_reset_splice_decomposition,
    rollout_latent_ode,
    rollout_memgrn_midpoint,
    rollout_memgrn_rk4,
)
from memgrn.identification import (
    ScheduleBoundResult,
    ar1_innovation_schedule_bound,
    decode_swapped_pairs,
    exact_swap_cancellation_accuracy,
    invert_swap_waveforms,
    paired_schedule_certificate,
    swap_singular_values,
    template_margin_lower_bound,
)

__all__ = [
    "MemGRNConfig",
    "MemGRNParams",
    "RateInputMode",
    "ScheduleBoundResult",
    "adaptive_rate_law",
    "ar1_innovation_schedule_bound",
    "compute_exact_contributions",
    "decode_swapped_pairs",
    "exact_swap_cancellation_accuracy",
    "invert_swap_waveforms",
    "memory_reset_splice_decomposition",
    "paired_schedule_certificate",
    "rollout_latent_ode",
    "rollout_memgrn_midpoint",
    "rollout_memgrn_rk4",
    "swap_singular_values",
    "template_margin_lower_bound",
]
