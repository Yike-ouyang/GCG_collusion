"""Information-theoretic collusion detection for multi-agent games."""

from netinf.influence import (
    detect,
    detect_calibrated,
    individual_influence,
    influence_matrix,
    net_influence,
    pair_scores,
    permutation_pvalues,
)

__all__ = [
    "detect", "detect_calibrated", "individual_influence", "influence_matrix", "net_influence", "pair_scores",
    "permutation_pvalues",
]
