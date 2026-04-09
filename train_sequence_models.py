"""
Sequence models for hallucination detection using per-token signals.

The CSV stores per-token arrays as JSON strings in columns:
  token_logprobs_json, token_entropies_json,
  token_top1_probs_json, token_margins_json

Each model treats the response as a variable-length sequence of
(logprob, entropy, top1_prob, margin) feature vectors.

Models:
  - Conv1DClassifier  – 1-D temporal CNN with global pooling
  - GRUClassifier     – Bi-directional GRU with attention pooling
  - TCN               – Temporal Convolutional Network with dilated convolutions
  - TransformerSeq    – Transformer encoder with CLS token

All models:
  - Pad/truncate to a fixed sequence length
  - Trained with BCE + positive class weight
  - Stratified K-Fold CV
  - PR-AUC metric

Usage:
    python train_sequence_models.py --data final_features.csv --label_col label
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from features import SEQ_COLUMNS

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Sequence models on per-token logprob / entropy signals."
    )
    p.add_argument("--data",      default="final_features.csv")
    p.add_argument("--label_col", default="label")
    p.add_argument("--n_splits",  type=int,   default=5)
    p.add_argument("--epochs",    type=int,   default=40)
    p.add_argument("--batch",     type=int,   default=128)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--max_len",   type=int,   default=256,
                   help="Pad / truncate sequences to this length")
    p.add_argument("--output",    default="results_sequence_models.csv")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--device",    default="auto")
    return p.parse_args()


def get_device(pref: str) -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():         return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


# ---------------------------------------------------------------------------
# Sequence loading
# ---------------------------------------------------------------------------

def load_token_sequences(df: pd.DataFrame, max_len: int) -> np.ndarray:
    """
    Parse JSON columns and build (N, max_len, C) tensor.
    C = number of channels = len(SEQ_COLUMNS).
    Missing columns are treated as zero channels.
    """
    n_channels = len(SEQ_COLUMNS)
    out = np.zeros((len(df), max_len, n_channels), dtype=np.float32)

    for ch_idx, col in enumerate(SEQ_COLUMNS):
        if col not in df.columns:
            continue
        for row_idx, raw in enumerate(df[col]):
            try:
                seq = np.array(json.loads(raw), dtype=np.float32)
            except Exception:
                continue
            L = min(len(seq), max_len)
            out[row_idx, :L, ch_idx] = seq[:L]

    return out   # (N, T, C)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class Conv1DClassifier(nn.Module):
    """1-D CNN with stacked convolutions and global average pooling."""

    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(hidden,      hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1), nn.GELU(),
        )
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, C, T) for Conv1d
        h = self.conv(x.transpose(1, 2))   # (B, hidden*2, T)
        # Masked global average pooling
        mask_exp = mask.unsqueeze(1).float()          # (B, 1, T)
        h = (h * mask_exp).sum(-1) / mask_exp.sum(-1).clamp(min=1)
        return self.head(h).squeeze(-1)


class GRUClassifier(nn.Module):
    """Bidirectional GRU with attention-weighted pooling."""

    def __init__(self, in_channels: int, hidden: int = 64,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            in_channels, hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)                 # (B, T, hidden*2)
        attn_w = self.attn(h).squeeze(-1)  # (B, T)
        attn_w = attn_w.masked_fill(~mask, float("-inf"))
        attn_w = torch.softmax(attn_w, dim=-1)
        pooled = (h * attn_w.unsqueeze(-1)).sum(1)    # (B, hidden*2)
        return self.head(pooled).squeeze(-1)


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float):
        super().__init__()
        p = dilation   # causal pad
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=2, dilation=dilation, padding=p),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=2, dilation=dilation, padding=p),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trim to same length (causal conv adds padding on both sides)
        y = self.net(x)
        return nn.functional.gelu(y[..., :x.shape[-1]] + self.skip(x))


class TCN(nn.Module):
    """Temporal Convolutional Network with dilated convolutions."""

    def __init__(self, in_channels: int, hidden: int = 64,
                 n_levels: int = 4, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i in range(n_levels):
            in_ch  = in_channels if i == 0 else hidden
            layers.append(_TCNBlock(in_ch, hidden, dilation=2**i, dropout=dropout))
        self.network = nn.ModuleList(layers)
        self.head    = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)          # (B, C, T)
        for blk in self.network:
            h = blk(h)
        # Masked global average pooling
        mask_exp = mask.unsqueeze(1).float()
        pooled   = (h * mask_exp).sum(-1) / mask_exp.sum(-1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


class TransformerSeq(nn.Module):
    """Lightweight Transformer encoder with a learnable CLS token."""

    def __init__(self, in_channels: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head    = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.proj(x)                           # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        h   = torch.cat([cls, h], dim=1)           # (B, T+1, d_model)
        # Pad mask with True (attend) for CLS position
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=~full_mask)
        return self.head(h[:, 0]).squeeze(-1)      # CLS → prediction


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_eval_seq(
    model_cls,
    model_kwargs: dict,
    X_seqs: np.ndarray,   # (N, T, C)
    y: np.ndarray,
    epochs: int, batch_size: int, lr: float,
    n_splits: int, seed: int, device: torch.device,
) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pos_weight = torch.tensor(
        [(y == 0).sum() / max(1, (y == 1).sum())],
        dtype=torch.float32, device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    fold_scores = []

    for train_idx, val_idx in cv.split(X_seqs, y):
        X_tr, X_val = X_seqs[train_idx], X_seqs[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Sequence-level statistics for normalisation (use training fold only)
        valid_mask_tr = (X_tr.sum(-1) != 0)   # (N_tr, T)
        flat = X_tr[valid_mask_tr]             # (N_tr*T_valid, C)
        mean_ = flat.mean(0, keepdims=True).mean(0) if len(flat) else np.zeros(X_tr.shape[-1])
        std_  = flat.std(0,  keepdims=True).mean(0) if len(flat) else np.ones(X_tr.shape[-1])
        std_  = np.where(std_ < 1e-6, 1.0, std_)

        X_tr_n  = (X_tr  - mean_) / std_
        X_val_n = (X_val - mean_) / std_

        # Mask: True where the token is non-padding
        mask_tr  = torch.tensor((X_tr.sum(-1)  != 0), dtype=torch.bool,  device=device)
        mask_val = torch.tensor((X_val.sum(-1) != 0), dtype=torch.bool,  device=device)
        X_tr_t   = torch.tensor(X_tr_n,  dtype=torch.float32, device=device)
        X_val_t  = torch.tensor(X_val_n, dtype=torch.float32, device=device)
        y_tr_t   = torch.tensor(y_tr,    dtype=torch.float32, device=device)

        in_channels = X_seqs.shape[-1]
        model = model_cls(in_channels=in_channels, **model_kwargs).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr * 0.05)

        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(X_tr_t), device=device)
            for i in range(0, len(X_tr_t), batch_size):
                b     = perm[i : i + batch_size]
                loss  = criterion(model(X_tr_t[b], mask_tr[b]), y_tr_t[b])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t, mask_val).cpu().numpy()
        proba_val = 1.0 / (1.0 + np.exp(-logits_val))
        fold_scores.append(average_precision_score(y_val, proba_val))

    return float(np.mean(fold_scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"Device: {device}")
    print(f"Loading {args.data} …")
    df = pd.read_csv(args.data)

    if args.label_col not in df.columns:
        sys.exit(f"ERROR: label column '{args.label_col}' not found.")
    y = df[args.label_col].astype(int).values

    # Check whether any sequence columns are present
    present = [c for c in SEQ_COLUMNS if c in df.columns]
    if not present:
        sys.exit(
            f"No sequence columns ({SEQ_COLUMNS}) found in dataset. "
            "These are required for sequence models."
        )
    print(f"Sequence columns found: {present}")

    print(f"Parsing token sequences (max_len={args.max_len}) …")
    X_seqs = load_token_sequences(df, args.max_len)    # (N, T, C)
    C = X_seqs.shape[-1]
    print(
        f"Sequences shape: {X_seqs.shape} "
        f"| pos rate: {y.mean():.3%}"
    )

    models_to_run = [
        ("Conv1D_64",    Conv1DClassifier, {"hidden": 64}),
        ("Conv1D_128",   Conv1DClassifier, {"hidden": 128}),
        ("GRU_64_bi",    GRUClassifier,    {"hidden": 64, "n_layers": 2}),
        ("GRU_128_bi",   GRUClassifier,    {"hidden": 128,"n_layers": 2}),
        ("TCN_64",       TCN,              {"hidden": 64, "n_levels": 4}),
        ("TransformerSeq_64_2L", TransformerSeq, {"d_model": 64, "n_heads": 4, "n_layers": 2}),
    ]

    results = []
    print(f"\n{'='*60}")
    for model_name, model_cls, model_kwargs in models_to_run:
        pr_auc = train_eval_seq(
            model_cls, model_kwargs, X_seqs, y,
            epochs=args.epochs, batch_size=args.batch, lr=args.lr,
            n_splits=args.n_splits, seed=args.seed, device=device,
        )
        print(f"  {model_name:<30}  PR-AUC={pr_auc:.4f}")
        results.append({
            "model":      model_name,
            "n_channels": C,
            "max_len":    args.max_len,
            "pr_auc":     round(pr_auc, 5),
        })

    df_res = (
        pd.DataFrame(results)
        .sort_values("pr_auc", ascending=False)
        .reset_index(drop=True)
    )
    out_path = Path(args.output)
    df_res.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
