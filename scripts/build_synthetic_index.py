"""Build a synthetic cap-weighted index return series from constituent data.

For each trading day, the index return is:
    r_index(t) = Σ_i  w_i(t) * r_i(t)

where w_i(t) = mktcap_i(m) / Σ_j mktcap_j(m), m being the most recent
month-end on or before t, and the sum runs over the constituents active
at date t according to the annual reconstitution files.

This is the standard definition of a cap-weighted total-return index and
is methodologically equivalent to the official Russell indices up to
float-adjustment differences and intra-year IPO/delisting handling.

Usage
-----
    python scripts/build_synthetic_index.py --index russell3000
    python scripts/build_synthetic_index.py --index russell1000
    python scripts/build_synthetic_index.py --index russell3000 \\
        --start-date 2011-01-01 --end-date 2023-12-31
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_synthetic_index(
    data_dir: Path,
    index_name: str,
    reconstitution_month: int,
    start_date: str | None,
    end_date: str | None,
) -> None:
    returns_path = data_dir / "returns_stocks.csv"
    mktcap_path  = data_dir / "mktcap_stocks.csv"
    const_dir    = data_dir / "constituants"
    dst          = data_dir / "returns_index.csv"

    for p in [returns_path, mktcap_path, const_dir]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    print(f"Loading data for {index_name} ...")
    returns = pd.read_csv(returns_path, index_col="date", parse_dates=True)
    mktcap  = pd.read_csv(mktcap_path,  index_col="date", parse_dates=True)

    if start_date:
        returns = returns.loc[start_date:]
        mktcap  = mktcap.loc[start_date:]
    if end_date:
        returns = returns.loc[:end_date]
        mktcap  = mktcap.loc[:end_date]

    # Load constituent files
    constituent_years = sorted(
        int(p.stem) for p in const_dir.glob("*.csv") if p.stem.isdigit()
    )
    if not constituent_years:
        raise FileNotFoundError(f"No constituent CSV files found in {const_dir}")

    permnos_by_year: dict[int, list[str]] = {}
    for y in constituent_years:
        df = pd.read_csv(const_dir / f"{y}.csv", dtype={"permno": str})
        permnos_by_year[y] = df["permno"].dropna().str.strip().tolist()

    def _effective_year(dt: pd.Timestamp) -> int:
        year = dt.year if dt.month >= reconstitution_month else dt.year - 1
        return max(constituent_years[0], min(constituent_years[-1], year))

    # First available mktcap snapshot (used for dates before first snapshot)
    first_snap = mktcap.index[0]

    index_returns: list[tuple[str, float]] = []
    prev_eff_year: int | None = None
    active_cols: list[str] = []

    for date in returns.index:
        # Update active constituents only when the effective year changes
        eff_year = _effective_year(date)
        if eff_year != prev_eff_year:
            active_cols = [p for p in permnos_by_year[eff_year] if p in returns.columns]
            prev_eff_year = eff_year

        if not active_cols:
            index_returns.append((date.strftime("%Y-%m-%d"), np.nan))
            continue

        # Most recent mktcap snapshot on or before this date.
        # For dates before the first snapshot, use the earliest available.
        avail = mktcap.index[mktcap.index <= date]
        snap_idx = avail[-1] if len(avail) > 0 else first_snap

        snap = mktcap.loc[snap_idx, active_cols]
        weights = snap.where(snap > 0, other=np.nan).dropna()
        total = weights.sum()
        if total <= 0:
            index_returns.append((date.strftime("%Y-%m-%d"), np.nan))
            continue
        weights = weights / total

        common = weights.index.intersection(returns.columns)
        r = (returns.loc[date, common] * weights[common]).sum()
        index_returns.append((date.strftime("%Y-%m-%d"), float(r)))

    out = pd.DataFrame(index_returns, columns=["Date", index_name])
    n_nan = out[index_name].isna().sum()
    out.to_csv(dst, index=False)
    print(f"Wrote {len(out)} rows to {dst}  ({out['Date'].iloc[0]} – {out['Date'].iloc[-1]})")
    if n_nan:
        print(f"  Warning: {n_nan} days with NaN index return.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build synthetic cap-weighted index from constituent returns"
    )
    parser.add_argument(
        "--index", type=str, required=True,
        help="Index name, e.g. russell3000 or russell1000"
    )
    parser.add_argument(
        "--data-path", type=Path, default=Path("financial_data"),
        help="Base directory for financial data (default: financial_data)"
    )
    parser.add_argument(
        "--reconstitution-month", type=int, default=7,
        help="First month where new composition is active (default: 7 = July)"
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Restrict output to dates >= start-date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="Restrict output to dates <= end-date (YYYY-MM-DD)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_path / args.index
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    build_synthetic_index(
        data_dir=data_dir,
        index_name=args.index,
        reconstitution_month=args.reconstitution_month,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()
