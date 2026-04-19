"""Download KRX daily OHLCV + trading value via pykrx and cache as per-ticker CSVs.

CSV schema matches the Kronos input format:
    timestamps, open, high, low, close, volume, amount
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from pykrx import stock

COLS_OHLCV_KO = ["시가", "고가", "저가", "종가", "거래량"]
COLS_OHLCV_EN = ["open", "high", "low", "close", "volume"]


def fetch_one(ticker: str, fromdate: str, todate: str) -> pd.DataFrame:
    ohlcv = stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()

    missing = [c for c in COLS_OHLCV_KO if c not in ohlcv.columns]
    if missing:
        raise RuntimeError(f"{ticker}: pykrx returned unexpected columns {list(ohlcv.columns)}")

    df = ohlcv[COLS_OHLCV_KO].rename(columns=dict(zip(COLS_OHLCV_KO, COLS_OHLCV_EN)))
    if "거래대금" in ohlcv.columns:
        df["amount"] = ohlcv["거래대금"].astype(float)
    else:
        df["amount"] = (df["close"].astype(float) * df["volume"].astype(float))

    df = df.reset_index().rename(columns={"날짜": "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes-json", required=True, help="Path to krx_all_codes.json")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--todate", default=datetime.today().strftime("%Y%m%d"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all tickers")
    ap.add_argument("--refresh", action="store_true", help="Overwrite existing CSVs")
    ap.add_argument("--sleep", type=float, default=0.2, help="Seconds between requests")
    ap.add_argument("--min-rows", type=int, default=572, help="Skip if fewer bars returned (lookback 512 + pred 60)")
    args = ap.parse_args()

    codes_path = Path(args.codes_json).resolve()
    with open(codes_path, encoding="utf-8") as f:
        codes = json.load(f)
    tickers = list(codes.keys())
    if args.limit > 0:
        tickers = tickers[: args.limit]

    todate = args.todate
    fromdate = (pd.to_datetime(todate) - pd.DateOffset(years=args.years)).strftime("%Y%m%d")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped_path = out_dir.parent / "output" / "fetch_skipped.csv"
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_rows = []

    print(f"[fetch_krx] range {fromdate}..{todate}  tickers={len(tickers)}  out={out_dir}")

    ok, skip, err = 0, 0, 0
    for i, code in enumerate(tickers, 1):
        meta = codes.get(code, {})
        name = meta.get("name", "")
        market = meta.get("market", "")
        out_file = out_dir / f"{code}.csv"

        if out_file.exists() and not args.refresh:
            skip += 1
            if i % 100 == 0:
                print(f"  [{i}/{len(tickers)}] cached skips={skip} ok={ok} err={err}")
            continue

        try:
            df = fetch_one(code, fromdate, todate)
            if df.empty or len(df) < args.min_rows:
                skipped_rows.append({"code": code, "name": name, "market": market,
                                     "reason": f"rows={len(df)} < min={args.min_rows}"})
                err += 1
            else:
                df.to_csv(out_file, index=False)
                ok += 1
        except Exception as e:
            skipped_rows.append({"code": code, "name": name, "market": market,
                                 "reason": f"error: {type(e).__name__}: {e}"})
            err += 1
            print(f"  ! {code} ({name}): {e}", file=sys.stderr)

        if args.sleep > 0:
            time.sleep(args.sleep)

        if i % 50 == 0 or i == len(tickers):
            print(f"  [{i}/{len(tickers)}] ok={ok} skipped(cached)={skip} err={err}")

    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(skipped_path, index=False)
        print(f"[fetch_krx] wrote {len(skipped_rows)} skipped entries to {skipped_path}")

    print(f"[fetch_krx] done. ok={ok} cached={skip} err={err}")


if __name__ == "__main__":
    main()
