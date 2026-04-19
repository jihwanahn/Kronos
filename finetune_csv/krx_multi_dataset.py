"""Multi-ticker dataset for KRX daily CSVs.

Solves the boundary issue: the vanilla CustomKlineDataset concatenates data
from a single CSV into one time series. With per-ticker CSVs that would
produce windows spanning two unrelated tickers. This class keeps each ticker
isolated: sliding windows never cross a ticker boundary.

Split strategy: **ticker-level**. A fraction of tickers is held out for
validation (not time-wise within ticker, since each ticker has only ~1200
daily bars — too short for a temporal split with lookback=512). Additionally,
the last `reserve_tail` bars of every ticker are stripped so the backtest
window used in `examples/krx_forecast/run_forecast.py` stays unseen.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURE_LIST = ["open", "high", "low", "close", "volume", "amount"]
TIME_FEATURE_LIST = ["minute", "hour", "weekday", "day", "month"]


class KRXMultiTickerDataset(Dataset):
    """One dataset spanning N per-ticker CSVs, with boundary-aware windows."""

    def __init__(
        self,
        data_dir: str | Path,
        data_type: str = "train",
        lookback_window: int = 512,
        predict_window: int = 12,
        clip: float = 5.0,
        seed: int = 42,
        ticker_val_ratio: float = 0.1,
        reserve_tail: int = 60,
        min_length: int | None = None,
        # Compatibility placeholders (unused — ticker-level split replaces them)
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_type = data_type
        self.lookback = lookback_window
        self.predict = predict_window
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
        self.feature_list = FEATURE_LIST
        self.time_feature_list = TIME_FEATURE_LIST

        if min_length is None:
            min_length = self.window + 20  # small buffer so degenerate edge windows are avoided

        csv_paths = sorted(self.data_dir.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No CSVs under {self.data_dir}")

        rng = random.Random(seed)
        shuffled = list(csv_paths)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * ticker_val_ratio)))
        train_set = shuffled[n_val:]
        val_set = shuffled[:n_val]
        subset = train_set if data_type == "train" else val_set

        self.arrays: List[np.ndarray] = []
        self.index: List[Tuple[int, int]] = []
        skipped_short = 0

        for p in subset:
            df = pd.read_csv(p, parse_dates=["timestamps"])
            df = df.sort_values("timestamps").reset_index(drop=True)
            if reserve_tail > 0:
                df = df.iloc[:-reserve_tail]
            if len(df) < min_length:
                skipped_short += 1
                continue
            if df[self.feature_list].isnull().any().any():
                df[self.feature_list] = df[self.feature_list].ffill().bfill()

            ts = df["timestamps"].dt
            df["minute"] = ts.minute
            df["hour"] = ts.hour
            df["weekday"] = ts.weekday
            df["day"] = ts.day
            df["month"] = ts.month

            arr = df[self.feature_list + self.time_feature_list].to_numpy(dtype=np.float32)
            arr_idx = len(self.arrays)
            self.arrays.append(arr)
            n_valid = len(arr) - self.window + 1
            for s in range(n_valid):
                self.index.append((arr_idx, s))

        self.n_samples = len(self.index)
        self.py_rng = random.Random(seed)
        self.current_epoch = 0

        print(
            f"[{data_type.upper()}] tickers used={len(self.arrays)} "
            f"(skipped {skipped_short} too-short)  samples={self.n_samples}  "
            f"window={self.window} (lookback {self.lookback}+predict {self.predict}+1)"
        )

    def set_epoch_seed(self, epoch: int) -> None:
        self.py_rng.seed(self.seed + epoch)
        self.current_epoch = epoch

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        if self.n_samples == 0:
            raise IndexError("Dataset is empty")
        if self.data_type == "train":
            # Deterministic shuffle per epoch, matching the vanilla loader's style.
            real_idx = (idx * 9973 + (self.current_epoch + 1) * 104729) % self.n_samples
        else:
            real_idx = idx % self.n_samples

        arr_idx, start = self.index[real_idx]
        window = self.arrays[arr_idx][start : start + self.window]

        x = window[:, : len(self.feature_list)]
        x_stamp = window[:, len(self.feature_list) :]

        mean = x.mean(axis=0)
        std = x.std(axis=0)
        x = (x - mean) / (std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(
            np.ascontiguousarray(x_stamp, dtype=np.float32)
        )
