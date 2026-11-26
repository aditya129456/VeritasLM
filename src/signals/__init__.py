"""
Internal generation signal extraction package for hallucination detection.
"""

from src.signals.token_signals import (
    TokenSignal,
    SequenceSignals,
    compute_token_signals_from_logits,
    extract_raw_sequence_signals,
)
from src.signals.score_labeled_responses import (
    format_scoring_prompt,
    score_single_response,
    score_labeled_dataset,
    validate_scored_dataset,
)
from src.signals.intermediate_probing import (
    DEFAULT_DEPTH_FRACTIONS,
    resolve_model_components,
    get_intermediate_layer_indices,
    get_intermediate_scalar_feature_names,
    extract_single_intermediate_features,
    extract_dataset_intermediate_representations,
)

__all__ = [
    "TokenSignal",
    "SequenceSignals",
    "compute_token_signals_from_logits",
    "extract_raw_sequence_signals",
    "format_scoring_prompt",
    "score_single_response",
    "score_labeled_dataset",
    "validate_scored_dataset",
    "DEFAULT_DEPTH_FRACTIONS",
    "resolve_model_components",
    "get_intermediate_layer_indices",
    "get_intermediate_scalar_feature_names",
    "extract_single_intermediate_features",
    "extract_dataset_intermediate_representations",
]
