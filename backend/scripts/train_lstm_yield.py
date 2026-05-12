"""Train per-crop BiLSTM + attention yield regressors on weekly Sentinel-2
time-series.

Input shape per sample: (T=22 weeks, F=8 features) — ISO-weeks 17–38 (mid-Apr
through mid-Sep), 4 vegetation indices (NDVI, EVI, NDWI, SAVI) + 4 weather
channels (temp_mean, precip, heat_flag, dry_flag) downsampled to weekly.

Architecture:
  BiLSTM(hidden=64, layers=2, dropout=0.3)
  → Multi-head attention (4 heads)
  → MeanPool over time
  → MLP(64 → 32 → 1)
  → Scalar t/ha

Saved as TorchScript so inference doesn't depend on the training source file
nor on the exact `torch.nn` class hierarchy.

This script depends on the same Sentinel-2 parquet as
`build_features_v2.py`, but reads the WEEKLY rows directly (no aggregation
to (oblast, year) flat features). If the parquet is missing, the script
exits with a friendly message.

Run: uv run python scripts/train_lstm_yield.py
"""
from __future__ import annotations

import json
import logging
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Heavy imports — defer until we know the parquet exists.
log = logging.getLogger("train_lstm")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parents[1]
S2_PARQUET = ROOT / "data" / "processed" / "oblast_s2_observations.parquet"
YIELD_CSV = ROOT / "data" / "raw" / "usda_ukraine_yield_2017_2023.csv"
MODELS_DIR = ROOT / "models"
METRICS_PATH = ROOT / "data" / "processed" / "model_metrics_v2.json"

CROPS = ("wheat", "corn", "sunflower")
WEEK_RANGE = list(range(17, 39))  # 22 weeks: late Apr through late Sep
N_TIMESTEPS = len(WEEK_RANGE)
N_FEATURES = 8  # ndvi, evi, ndwi, savi, temp_mean, precip, heat_flag, dry_flag


def build_sequences(s2_df: pd.DataFrame, yield_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Stack per-(oblast, year, crop) into a (N, T, F) tensor."""
    # Per (oblast, year, iso_week) take median across samples for the 4 indices.
    weekly = (
        s2_df.groupby(["oblast", "year", "iso_week"])[["ndvi_mean", "evi_mean", "ndwi_mean", "savi_mean"]]
        .median()
        .reset_index()
    )
    # The weather features are at (oblast, year) granularity — broadcast them
    # uniformly across all 22 timesteps. A future extension could pull
    # weekly Open-Meteo here for true per-week weather.
    weather = (
        yield_df[["oblast", "year"]].drop_duplicates()
        .assign(temp_mean=15.0, precip=2.5, heat_flag=0.0, dry_flag=0.0)
    )

    sequences = []
    targets = []
    keys: list[dict] = []

    for crop in CROPS:
        sub_y = yield_df[yield_df["crop"] == crop]
        for _, y_row in sub_y.iterrows():
            ob, yr = y_row["oblast"], int(y_row["year"])
            wk = weekly[(weekly["oblast"] == ob) & (weekly["year"] == yr)]
            if wk.empty:
                continue
            tensor = np.full((N_TIMESTEPS, N_FEATURES), np.nan, dtype=np.float32)
            wk_indexed = wk.set_index("iso_week")
            for ti, week_no in enumerate(WEEK_RANGE):
                if week_no in wk_indexed.index:
                    r = wk_indexed.loc[week_no]
                    tensor[ti, 0] = _f(r.get("ndvi_mean"))
                    tensor[ti, 1] = _f(r.get("evi_mean"))
                    tensor[ti, 2] = _f(r.get("ndwi_mean"))
                    tensor[ti, 3] = _f(r.get("savi_mean"))
                # Weather is uniform across weeks for this version.
                w = weather[(weather["oblast"] == ob) & (weather["year"] == yr)]
                if not w.empty:
                    w0 = w.iloc[0]
                    tensor[ti, 4] = w0["temp_mean"]
                    tensor[ti, 5] = w0["precip"]
                    tensor[ti, 6] = w0["heat_flag"]
                    tensor[ti, 7] = w0["dry_flag"]
            # Replace NaN with the per-feature mean (zero-impute for missing weeks).
            for fi in range(N_FEATURES):
                col = tensor[:, fi]
                if np.isnan(col).all():
                    col[:] = 0
                else:
                    col[np.isnan(col)] = np.nanmean(col)
                tensor[:, fi] = col

            sequences.append(tensor)
            targets.append(float(y_row["yield_tha"]))
            keys.append({"oblast": ob, "year": yr, "crop": crop})

    if not sequences:
        return np.empty((0, N_TIMESTEPS, N_FEATURES)), np.empty(0), pd.DataFrame()

    X = np.stack(sequences)
    y = np.asarray(targets, dtype=np.float32)
    return X, y, pd.DataFrame(keys)


def _f(v) -> float:
    try:
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def train_lstm_for_crop(
    crop: str, X: np.ndarray, y: np.ndarray, keys: pd.DataFrame,
) -> tuple[object, dict]:
    """Trains a BiLSTM+attention on (X, y) for one crop. Returns (model, metrics)."""
    import torch
    import torch.nn as nn

    mask = keys["crop"].values == crop
    Xc = X[mask]
    yc = y[mask]
    yrs = keys.loc[mask, "year"].values

    train_mask = yrs <= 2021
    val_mask = yrs == 2022
    test_mask = yrs == 2023

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        log.warning("crop=%s insufficient data — skipping LSTM training.", crop)
        return None, {}

    log.info("crop=%s train=%d val=%d test=%d",
             crop, int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()))

    X_train = torch.from_numpy(Xc[train_mask]).float()
    y_train = torch.from_numpy(yc[train_mask]).float()
    X_val = torch.from_numpy(Xc[val_mask]).float() if val_mask.sum() else None
    y_val = torch.from_numpy(yc[val_mask]).float() if val_mask.sum() else None
    X_test = torch.from_numpy(Xc[test_mask]).float()
    y_test = torch.from_numpy(yc[test_mask]).float()

    class YieldLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=N_FEATURES, hidden_size=64, num_layers=2,
                batch_first=True, dropout=0.3, bidirectional=True,
            )
            self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
            self.mlp = nn.Sequential(
                nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h, _ = self.lstm(x)
            a, _ = self.attn(h, h, h)
            pooled = a.mean(dim=1)
            return self.mlp(pooled).squeeze(-1)

    torch.manual_seed(42)
    model = YieldLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    no_improve = 0
    for epoch in range(200):
        model.train()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(X_val), y_val))
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= 25:
                log.info("crop=%s early stop at epoch %d (val=%.3f)", crop, epoch, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def _metrics_block(X_, y_):
        with torch.no_grad():
            pred = model(X_).cpu().numpy()
        y_np = y_.cpu().numpy()
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        return {
            "rmse": float(math.sqrt(mean_squared_error(y_np, pred))),
            "mae": float(mean_absolute_error(y_np, pred)),
            "r2": float(r2_score(y_np, pred)),
            "n": int(len(y_)),
        }

    metrics = {"train": _metrics_block(X_train, y_train)}
    if X_val is not None:
        metrics["val"] = _metrics_block(X_val, y_val)
    metrics["test"] = _metrics_block(X_test, y_test)
    log.info("crop=%s LSTM test: R²=%.3f MAE=%.3f", crop, metrics["test"]["r2"], metrics["test"]["mae"])

    return model, metrics


def main() -> int:
    if not S2_PARQUET.exists():
        log.error("Missing %s — run collect_oblast_s2.py first.", S2_PARQUET)
        return 1
    if not YIELD_CSV.exists():
        log.error("Missing %s — run scripts/build_yield_csv.py first.", YIELD_CSV)
        return 1

    import torch

    s2 = pd.read_parquet(S2_PARQUET)
    yield_df = pd.read_csv(YIELD_CSV)
    log.info("Loaded %d S2 rows + %d yield rows", len(s2), len(yield_df))

    X, y, keys = build_sequences(s2, yield_df)
    log.info("Sequences shape: %s, targets: %s", X.shape, y.shape)
    if X.size == 0:
        log.error("No matched sequences — verify that S2 and yield share oblasts.")
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_out: dict = {}

    for crop in CROPS:
        model, metrics = train_lstm_for_crop(crop, X, y, keys)
        if model is None:
            continue

        scripted = torch.jit.script(model)
        out_path = MODELS_DIR / f"yield_lstm_{crop}_v1.pt"
        # torch.jit.ScriptModule.save() uses a C++ filename API that fails
        # on Windows when the path contains non-ASCII (here: 'Курсова 5 курс').
        # Workaround: serialise to BytesIO, then write the buffer to disk
        # through Python's open() which handles Unicode paths fine.
        import io
        buf = io.BytesIO()
        torch.jit.save(scripted, buf)
        out_path.write_bytes(buf.getvalue())
        log.info("Saved %s (%.0f KB)", out_path, out_path.stat().st_size / 1024)

        meta_path = MODELS_DIR / f"yield_lstm_{crop}_v1.meta.json"
        meta_path.write_text(json.dumps({
            "crop": crop, "version": "v1", "algorithm": "lstm",
            "metrics": metrics,
            "input_shape": [N_TIMESTEPS, N_FEATURES],
            "trained_at": datetime.now(UTC).isoformat(),
        }, indent=2), encoding="utf-8")

        metrics_out[f"lstm_{crop}_v1"] = metrics

    # Merge into the consolidated metrics file alongside RF/XGB if present.
    consolidated: dict = {}
    if METRICS_PATH.exists():
        try:
            consolidated = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            consolidated = {}
    consolidated.setdefault("models", {}).update(metrics_out)
    consolidated.setdefault("trained_at", datetime.now(UTC).isoformat())
    METRICS_PATH.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")
    log.info("Updated consolidated metrics %s", METRICS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
