"""
Feature group definitions for hallucination detection experiments.

Two method families:
  MHAD — "Detecting Hallucination through Deep Internal Representation Analysis"
          (Zhang et al., IJCAI 2025). Uses hidden states at the FIRST and LAST
          generated token positions as the "hallucination awareness vector", with
          linear probes to rank/select features and then an MLP classifier.

  SEPs  — "Semantic Entropy Probes: Robust and Cheap Hallucination Detection"
          (Kossen et al., arXiv 2406.15927). Trains logistic regression probes
          on hidden state vectors at specific layers/positions to approximate
          semantic entropy without multiple-sample inference.

In our dataset the hidden state vectors are stored as aggregated statistics
(norm, mean, std, maxabs) rather than full vectors, so both methods are adapted
to work with these 4-dimensional summaries per layer × position cell.
"""

# ---------------------------------------------------------------------------
# Metadata / token-count features
# ---------------------------------------------------------------------------
META_FEATURES = [
    "prompt_tokens",
    "response_tokens",
    "scored_response_tokens",
    "truncated_total_tokens",
    "response_text_chars",
]

# ---------------------------------------------------------------------------
# Model uncertainty signals  (logprob / entropy / margin statistics)
# These form the "MHAD logprob" feature group used in uncertainty-based methods.
# ---------------------------------------------------------------------------
UNCERTAINTY_FEATURES = [
    # Log-probability distribution statistics over response tokens
    "avg_logprob", "min_logprob", "max_logprob", "std_logprob",
    "p05_logprob", "p10_logprob",
    # Entropy statistics
    "avg_entropy", "max_entropy", "std_entropy",
    # Top-1 probability and vocabulary margin statistics
    "avg_top1_prob", "min_top1_prob", "avg_top5_mass",
    "avg_margin", "min_margin", "std_margin",
    # Fraction of tokens with high uncertainty
    "frac_logprob_lt_-1", "frac_logprob_lt_-2", "frac_logprob_lt_-3",
    "frac_entropy_gt_2", "frac_entropy_gt_5",
    # Perplexity and position of the most uncertain token
    "perplexity",
    "min_logprob_token_index_local",
    "min_logprob_token_position_global",
]

# ---------------------------------------------------------------------------
# Hidden-state layer configuration
# ---------------------------------------------------------------------------
LAYERS    = [22, 24, 25]            # late LLM layers stored in the dataset
POSITIONS = ["first", "last", "minlogprob", "mean"]
STATS     = ["norm", "mean", "std", "maxabs"]


def _layer_cols(layer: int, position: str, stats=STATS) -> list:
    """Return column names for a (layer, position) cell."""
    return [f"layer{layer}_{position}_{stat}" for stat in stats]


# Organized nested dict:  LAYER_FEATURES_BY_LAYER[layer][position] = [col, ...]
LAYER_FEATURES_BY_LAYER = {
    layer: {pos: _layer_cols(layer, pos) for pos in POSITIONS}
    for layer in LAYERS
}

# Flat lists by position across all layers
HIDDEN_FIRST     = [c for L in LAYERS for c in _layer_cols(L, "first")]
HIDDEN_LAST      = [c for L in LAYERS for c in _layer_cols(L, "last")]
HIDDEN_MINLOGPROB = [c for L in LAYERS for c in _layer_cols(L, "minlogprob")]
HIDDEN_MEAN      = [c for L in LAYERS for c in _layer_cols(L, "mean")]

# ---------------------------------------------------------------------------
# Inter-layer difference features  (L2 distance and cosine similarity)
# ---------------------------------------------------------------------------
ALL_LAYERDIFF_FEATURES = [
    "layerdiff_25_24_first_l2",      "layerdiff_25_24_first_cos",
    "layerdiff_24_22_first_l2",      "layerdiff_24_22_first_cos",
    "layerdiff_25_24_last_l2",       "layerdiff_25_24_last_cos",
    "layerdiff_24_22_last_l2",       "layerdiff_24_22_last_cos",
    "layerdiff_25_24_minlogprob_l2", "layerdiff_25_24_minlogprob_cos",
    "layerdiff_24_22_minlogprob_l2", "layerdiff_24_22_minlogprob_cos",
    "layerdiff_25_24_mean_l2",       "layerdiff_25_24_mean_cos",
    "layerdiff_24_22_mean_l2",       "layerdiff_24_22_mean_cos",
]

LAYERDIFF_FIRST     = [c for c in ALL_LAYERDIFF_FEATURES if "_first_" in c]
LAYERDIFF_LAST      = [c for c in ALL_LAYERDIFF_FEATURES if "_last_" in c]
LAYERDIFF_MINLOGPROB = [c for c in ALL_LAYERDIFF_FEATURES if "_minlogprob_" in c]
LAYERDIFF_MEAN      = [c for c in ALL_LAYERDIFF_FEATURES if "_mean_" in c]

# ---------------------------------------------------------------------------
# Compound feature sets  (used across training scripts)
# ---------------------------------------------------------------------------

# MHAD: combines initial (first token) + final (last token) hidden states
MHAD_HIDDEN_FEATURES = (
    HIDDEN_FIRST + HIDDEN_LAST + LAYERDIFF_FIRST + LAYERDIFF_LAST
)

# SEP: all layer / position statistics
ALL_LAYER_FEATURES = (
    HIDDEN_FIRST + HIDDEN_LAST + HIDDEN_MINLOGPROB + HIDDEN_MEAN
    + ALL_LAYERDIFF_FEATURES
)

# Named sets used by train_classical.py and train_neural_nets.py
FEATURE_SETS = {
    "uncertainty":      UNCERTAINTY_FEATURES,
    "meta":             META_FEATURES,
    "uncertainty+meta": UNCERTAINTY_FEATURES + META_FEATURES,
    "mhad_hidden":      MHAD_HIDDEN_FEATURES,
    "mhad_full":        UNCERTAINTY_FEATURES + META_FEATURES + MHAD_HIDDEN_FEATURES,
    "sep_layers":       ALL_LAYER_FEATURES,
    "sep_first+last":   HIDDEN_FIRST + HIDDEN_LAST + LAYERDIFF_FIRST + LAYERDIFF_LAST,
    "all_features":     META_FEATURES + UNCERTAINTY_FEATURES + ALL_LAYER_FEATURES,
}

# Per-layer, per-position probe sets  (used by train_sep_probes.py)
SEP_PROBE_SETS: dict = {}
for _L in LAYERS:
    for _pos in POSITIONS:
        SEP_PROBE_SETS[f"layer{_L}_{_pos}"] = _layer_cols(_L, _pos)
    SEP_PROBE_SETS[f"layer{_L}_all"] = [
        c for _pos in POSITIONS for c in _layer_cols(_L, _pos)
    ]
SEP_PROBE_SETS["diff_first"]      = LAYERDIFF_FIRST
SEP_PROBE_SETS["diff_last"]       = LAYERDIFF_LAST
SEP_PROBE_SETS["diff_minlogprob"] = LAYERDIFF_MINLOGPROB
SEP_PROBE_SETS["diff_mean"]       = LAYERDIFF_MEAN
SEP_PROBE_SETS["diff_all"]        = ALL_LAYERDIFF_FEATURES

# Sequential feature columns (serialised as JSON strings in the CSV)
SEQ_COLUMNS = [
    "token_logprobs_json",
    "token_entropies_json",
    "token_top1_probs_json",
    "token_margins_json",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def available_cols(columns, df_cols) -> list:
    """Return only the columns that actually exist in the dataframe."""
    df_set = set(df_cols)
    return [c for c in columns if c in df_set]
