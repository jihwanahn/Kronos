"""Run NeoQuasar/Kronos-base backtest + forward forecast on cached KRX daily CSVs.

Reads CSVs produced by fetch_krx.py from ./data/{code}.csv and writes:
    ./output/backtest/{code}.csv     - predicted vs actual OHLCV for last `pred_len` days
    ./output/forward/{code}.csv      - predicted OHLCV for next `pred_len` business days
    ./output/metrics.csv             - per-ticker backtest metrics
    ./output/run_skipped.csv         - tickers skipped with reason
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

# Make the repo root importable so `from model import ...` works.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402
PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def load_model(device: str, max_context: int = 512) -> KronosPredictor:
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    return KronosPredictor(model, tokenizer, device=device, max_context=max_context)


def future_business_days(last_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """Return the next `n` weekdays strictly after `last_date`.

    Uses pandas bdate_range (Mon–Fri). KRX public holidays are not excluded, but
    Kronos only consumes weekday/day/month timestamp features so a small offset
    from true trading days does not affect the model's time embeddings meaningfully.
    """
    start = pd.Timestamp(last_date) + pd.Timedelta(days=1)
    return pd.bdate_range(start=start, periods=n)


def compute_metrics(pred: pd.DataFrame, actual: pd.DataFrame) -> dict:
    p = pred["close"].to_numpy(dtype=float)
    a = actual["close"].to_numpy(dtype=float)
    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(np.abs(a) < 1e-9, np.nan, a)
    mape = float(np.nanmean(np.abs(err / denom)) * 100.0)

    # Directional accuracy: sign of day-over-day change (first diff uses last_actual_context as anchor)
    p_dir = np.sign(np.diff(p))
    a_dir = np.sign(np.diff(a))
    if len(p_dir) == 0:
        dir_acc = float("nan")
    else:
        dir_acc = float(np.mean(p_dir == a_dir))
    return {"mae_close": mae, "rmse_close": rmse, "mape_close_pct": mape, "dir_acc": dir_acc}


def run_batch(predictor: KronosPredictor, items: list, mode: str, pred_len: int,
              T: float, top_p: float, sample_count: int) -> list:
    """items: list of dicts with keys {code, x_df, x_ts, y_ts}. Returns list of (code, pred_df)."""
    if not items:
        return []
    preds = predictor.predict_batch(
        df_list=[it["x_df"] for it in items],
        x_timestamp_list=[it["x_ts"] for it in items],
        y_timestamp_list=[it["y_ts"] for it in items],
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )
    return list(zip([it["code"] for it in items], preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "output"))
    ap.add_argument("--codes-json", default=str(Path(__file__).resolve().parents[2] / "krx_all_codes.json"))
    ap.add_argument("--mode", choices=["backtest", "forward", "both"], default="both")
    ap.add_argument("--lookback", type=int, default=512)
    ap.add_argument("--pred-len", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0, help="0 = all CSVs in data dir")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (auto if omitted)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--sample-count", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[run_forecast] device={device}  mode={args.mode}  lookback={args.lookback}  pred_len={args.pred_len}")

    codes_meta = {}
    codes_path = Path(args.codes_json)
    if codes_path.exists():
        with open(codes_path, encoding="utf-8") as f:
            codes_meta = json.load(f)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "backtest").mkdir(parents=True, exist_ok=True)
    (out_dir / "forward").mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(data_dir.glob("*.csv"))
    if args.limit > 0:
        csv_paths = csv_paths[: args.limit]
    print(f"[run_forecast] found {len(csv_paths)} ticker CSVs in {data_dir}")

    # Load CSVs and prepare inputs per mode
    min_len = args.lookback + (args.pred_len if args.mode in ("backtest", "both") else 0)
    backtest_items, forward_items = [], []
    skipped = []

    for p in csv_paths:
        code = p.stem
        try:
            df = pd.read_csv(p, parse_dates=["timestamps"])
        except Exception as e:
            skipped.append({"code": code, "reason": f"read error: {e}"})
            continue
        if len(df) < min_len:
            skipped.append({"code": code, "reason": f"rows={len(df)} < required={min_len}"})
            continue

        df = df.sort_values("timestamps").reset_index(drop=True)

        if args.mode in ("backtest", "both"):
            ctx = df.iloc[-(args.lookback + args.pred_len): -args.pred_len]
            tgt = df.iloc[-args.pred_len:]
            backtest_items.append({
                "code": code,
                "x_df": ctx[PRICE_COLS].reset_index(drop=True),
                "x_ts": ctx["timestamps"].reset_index(drop=True),
                "y_ts": tgt["timestamps"].reset_index(drop=True),
                "actual": tgt[PRICE_COLS].reset_index(drop=True),
            })

        if args.mode in ("forward", "both"):
            ctx = df.iloc[-args.lookback:]
            last_ts = pd.Timestamp(df["timestamps"].iloc[-1])
            y_ts = future_business_days(last_ts, args.pred_len)
            forward_items.append({
                "code": code,
                "x_df": ctx[PRICE_COLS].reset_index(drop=True),
                "x_ts": ctx["timestamps"].reset_index(drop=True),
                "y_ts": pd.Series(y_ts),
            })

    print(f"[run_forecast] backtest items={len(backtest_items)}  forward items={len(forward_items)}  skipped={len(skipped)}")

    predictor = load_model(device=device)

    # ---- Backtest ----
    metrics_rows = []
    if backtest_items:
        print(f"[run_forecast] running backtest ({len(backtest_items)} series, batch={args.batch_size})")
        t0 = time.time()
        for i in range(0, len(backtest_items), args.batch_size):
            batch = backtest_items[i: i + args.batch_size]
            try:
                results = run_batch(predictor, batch, mode="backtest",
                                    pred_len=args.pred_len, T=args.temperature,
                                    top_p=args.top_p, sample_count=args.sample_count)
            except Exception as e:
                traceback.print_exc()
                for it in batch:
                    skipped.append({"code": it["code"], "reason": f"backtest batch error: {e}"})
                continue
            for (code, pred_df), it in zip(results, batch):
                actual = it["actual"].copy()
                actual.index = pred_df.index
                merged = pd.DataFrame({
                    "timestamps": pred_df.index,
                    **{f"pred_{c}": pred_df[c].to_numpy() for c in PRICE_COLS},
                    **{f"actual_{c}": actual[c].to_numpy() for c in PRICE_COLS},
                })
                merged.to_csv(out_dir / "backtest" / f"{code}.csv", index=False)
                m = compute_metrics(pred_df, actual)
                meta = codes_meta.get(code, {})
                metrics_rows.append({"code": code, "name": meta.get("name", ""),
                                     "market": meta.get("market", ""), **m})
            done = min(i + args.batch_size, len(backtest_items))
            print(f"  backtest progress: {done}/{len(backtest_items)}  elapsed={time.time()-t0:.1f}s")

        if metrics_rows:
            pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)
            mdf = pd.DataFrame(metrics_rows)
            print(f"[run_forecast] metrics summary: MAE={mdf['mae_close'].mean():.4f}  "
                  f"RMSE={mdf['rmse_close'].mean():.4f}  "
                  f"MAPE={mdf['mape_close_pct'].mean():.2f}%  "
                  f"DirAcc={mdf['dir_acc'].mean():.3f}")

    # ---- Forward ----
    if forward_items:
        print(f"[run_forecast] running forward ({len(forward_items)} series, batch={args.batch_size})")
        t0 = time.time()
        for i in range(0, len(forward_items), args.batch_size):
            batch = forward_items[i: i + args.batch_size]
            try:
                results = run_batch(predictor, batch, mode="forward",
                                    pred_len=args.pred_len, T=args.temperature,
                                    top_p=args.top_p, sample_count=args.sample_count)
            except Exception as e:
                traceback.print_exc()
                for it in batch:
                    skipped.append({"code": it["code"], "reason": f"forward batch error: {e}"})
                continue
            for (code, pred_df), it in zip(results, batch):
                out = pred_df.reset_index().rename(columns={"index": "timestamps"})
                out.to_csv(out_dir / "forward" / f"{code}.csv", index=False)
            done = min(i + args.batch_size, len(forward_items))
            print(f"  forward progress: {done}/{len(forward_items)}  elapsed={time.time()-t0:.1f}s")

    if skipped:
        pd.DataFrame(skipped).to_csv(out_dir / "run_skipped.csv", index=False)
        print(f"[run_forecast] wrote {len(skipped)} skipped entries to {out_dir / 'run_skipped.csv'}")

    print("[run_forecast] done.")


if __name__ == "__main__":
    main()
