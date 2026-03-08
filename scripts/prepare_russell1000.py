"""Prepare Russell 1000 data from existing Russell 3000 data.

Russell 1000 = top 1000 constituents of Russell 3000 by float-adjusted market cap
at the June reconstitution.  Since we already have monthly mktcap data for all
Russell 3000 permnos, we approximate the Russell 1000 composition by filtering to
the top ``--size`` permnos by mktcap at the June snapshot of each reconstitution year.

This approximation is very close to the official composition for large/mid caps and
avoids a full WRDS re-download.  For research purposes the difference is negligible.

What this script produces
-------------------------
financial_data/russell1000/
├── constituants/
│   ├── 2013.csv  … 2023.csv   (permno only, normalised)
│   └── all_permnos.csv
├── returns_stocks.csv          (subset of Russell 3000 returns)
├── returns_index.csv           (r1000ret downloaded from WRDS)
└── mktcap_stocks.csv           (subset of Russell 3000 mktcap)

Usage
-----
Step 1 — build constituents + filter returns/mktcap (offline):

    python scripts/prepare_russell1000.py \\
        --r3000-dir financial_data/russell3000 \\
        --output-dir financial_data/russell1000 \\
        --size 1000

Step 2 — add index returns (requires WRDS):

    python scripts/prepare_russell1000.py \\
        --r3000-dir financial_data/russell3000 \\
        --output-dir financial_data/russell1000 \\
        --size 1000 \\
        --download-index \\
        --start-date 2014-01-01 --end-date 2023-12-31

Step 3 — explore available WRDS index columns (once):

    python scripts/prepare_russell1000.py --explore-index \\
        --start-date 2014-01-01 --end-date 2023-12-31

Step 4 — run the pipeline:

    python main.py --index russell1000 --cardinality 100 \\
        --solution_name quob_stratified \\
        --distance_method pearson --missing_policy strict --hard_clip 1.0 \\
        --min_trading_frac 0.20 --rebalancing 6 \\
        --start_date 2014-01-02 --end_date 2023-12-31
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constituent derivation
# ---------------------------------------------------------------------------

def _derive_constituents(
    r3000_dir: Path,
    output_dir: Path,
    size: int,
    reconstitution_month: int,
) -> list[int]:
    """Derive Russell 1000 constituent files from Russell 3000 mktcap data.

    For each reconstitution year, we take the mktcap snapshot at ``reconstitution_month - 1``
    (the last full month before the new composition takes effect) and keep the top
    ``size`` permnos by market cap.

    Returns the union of all permnos across years (for downstream filtering).
    """
    mktcap_path = r3000_dir / "mktcap_stocks.csv"
    if not mktcap_path.exists():
        raise FileNotFoundError(
            f"Market-cap file not found: {mktcap_path}\n"
            "Run scripts/download_mktcap_data.py first."
        )

    constituants_dir = r3000_dir / "constituants"
    if not constituants_dir.exists():
        raise FileNotFoundError(
            f"Russell 3000 constituent directory not found: {constituants_dir}\n"
            "Run scripts/prepare_russell_constituents.py first."
        )

    print(f"Loading market-cap data from {mktcap_path} ...")
    mktcap = pd.read_csv(mktcap_path, index_col=0, parse_dates=True)
    mktcap.index = pd.to_datetime(mktcap.index)

    out_dir = output_dir / "constituants"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_permnos: set[str] = set()
    constituent_years = sorted(
        int(p.stem) for p in constituants_dir.glob("*.csv") if p.stem.isdigit()
    )

    for year in constituent_years:
        # Load R3000 constituents for this year
        r3000_permnos = pd.read_csv(
            constituants_dir / f"{year}.csv", dtype={"permno": str}
        )["permno"].dropna().str.strip().tolist()

        # Snapshot: last available month strictly before the reconstitution takes effect.
        # Russell reconstitutes end-of-June; effective from July.
        # We use the May month-end snapshot (or the closest available before July).
        snap_cutoff = pd.Timestamp(year=year, month=reconstitution_month, day=1)
        available = mktcap.index[mktcap.index < snap_cutoff]
        if len(available) == 0:
            print(f"  Warning: no mktcap snapshot before {snap_cutoff.date()} for year {year}. Skipping.")
            continue
        snap_date = available[-1]

        # Filter mktcap to R3000 constituents at this snapshot
        r3000_cols = [p for p in r3000_permnos if p in mktcap.columns]
        if not r3000_cols:
            print(f"  Warning: no mktcap data for R3000 year {year} constituents. Skipping.")
            continue

        snap = mktcap.loc[snap_date, r3000_cols].copy()
        snap = snap.where(pd.notna(snap) & (snap > 0), other=np.nan).dropna()

        # Top `size` by mktcap
        top = snap.nlargest(size)
        r1000_permnos = sorted(top.index.tolist(), key=int)

        n_missing = len(r3000_permnos) - len(snap)
        print(
            f"  Year {year}: snapshot {snap_date.date()}, "
            f"R3000={len(r3000_permnos)}, with mktcap={len(snap)}, "
            f"selected top-{size}={len(r1000_permnos)} "
            f"({'⚠️ ' + str(n_missing) + ' missing mktcap' if n_missing > 0 else 'OK'})"
        )

        all_permnos.update(r1000_permnos)
        df = pd.DataFrame({"permno": r1000_permnos})
        df.to_csv(out_dir / f"{year}.csv", index=False)

    # Union file
    all_sorted = sorted(all_permnos, key=int)
    pd.DataFrame({"permno": all_sorted}).to_csv(out_dir / "all_permnos.csv", index=False)
    print(f"\nWrote {len(all_sorted)} unique permnos to {out_dir / 'all_permnos.csv'}")
    return [int(p) for p in all_sorted]


# ---------------------------------------------------------------------------
# Synthetic index construction
# ---------------------------------------------------------------------------

def _build_synthetic_index(output_dir: Path, reconstitution_month: int) -> None:
    """Construct a synthetic Russell 1000 cap-weighted index from constituent data.

    For each day, the index return is:
        r_index(t) = Σ_i  w_i(t) * r_i(t)

    where w_i(t) = mktcap_i(m) / Σ_j mktcap_j(m), m being the most recent
    month-end on or before t, and the sum runs over the constituents active
    at date t (i.e. the reconstitution year in effect for that date).

    This is the standard definition of a cap-weighted total-return index and
    is equivalent to the official Russell 1000 up to (a) float-adjustment
    differences and (b) intra-year IPO/delisting handling.
    """
    returns_path = output_dir / "returns_stocks.csv"
    mktcap_path  = output_dir / "mktcap_stocks.csv"
    const_dir    = output_dir / "constituants"
    dst          = output_dir / "returns_index.csv"

    for p, label in [(returns_path, "returns_stocks.csv"),
                     (mktcap_path,  "mktcap_stocks.csv"),
                     (const_dir,    "constituants/")]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    print("\nLoading returns and mktcap for synthetic index construction ...")
    returns = pd.read_csv(returns_path, index_col="date", parse_dates=True)
    mktcap  = pd.read_csv(mktcap_path,  index_col="date", parse_dates=True)

    # Load constituent files → map each date to its active permno set.
    # Composition of year Y is active from July Y (reconstitution_month) to
    # June Y+1, mirroring the logic in universe.py.
    constituent_years = sorted(
        int(p.stem) for p in const_dir.glob("*.csv") if p.stem.isdigit()
    )

    def _active_permnos(dt: pd.Timestamp) -> list[str]:
        year = dt.year if dt.month >= reconstitution_month else dt.year - 1
        # Clamp to available range
        year = max(constituent_years[0], min(constituent_years[-1], year))
        df = pd.read_csv(const_dir / f"{year}.csv", dtype={"permno": str})
        return df["permno"].dropna().str.strip().tolist()

    # Pre-load constituent sets per effective year to avoid redundant disk reads
    permnos_by_year: dict[int, list[str]] = {}
    for y in constituent_years:
        df = pd.read_csv(const_dir / f"{y}.csv", dtype={"permno": str})
        permnos_by_year[y] = df["permno"].dropna().str.strip().tolist()

    def _effective_year(dt: pd.Timestamp) -> int:
        year = dt.year if dt.month >= reconstitution_month else dt.year - 1
        return max(constituent_years[0], min(constituent_years[-1], year))

    index_returns: list[tuple[str, float]] = []
    prev_eff_year = None
    active_cols: list[str] = []

    for date in returns.index:
        # Update active constituents only when the effective year changes
        eff_year = _effective_year(date)
        if eff_year != prev_eff_year:
            active_cols = [p for p in permnos_by_year[eff_year] if p in returns.columns]
            prev_eff_year = eff_year

        # Most recent mktcap snapshot on or before this date
        avail = mktcap.index[mktcap.index <= date]
        if len(avail) == 0 or not active_cols:
            index_returns.append((date.strftime("%Y-%m-%d"), np.nan))
            continue
        snap = mktcap.loc[avail[-1], active_cols]
        weights = snap.where(snap > 0, other=np.nan).dropna()
        total = weights.sum()
        if total <= 0:
            index_returns.append((date.strftime("%Y-%m-%d"), np.nan))
            continue
        weights = weights / total

        # Cap-weighted return
        common = weights.index.intersection(returns.columns)
        r = (returns.loc[date, common] * weights[common]).sum()
        index_returns.append((date.strftime("%Y-%m-%d"), float(r)))

    out = pd.DataFrame(index_returns, columns=["Date", "russell1000"])
    n_nan = out["russell1000"].isna().sum()
    if n_nan:
        print(f"  Warning: {n_nan} days with missing index return (no mktcap snapshot).")
    out.to_csv(dst, index=False)
    print(f"  Wrote synthetic Russell 1000 index to {dst}  ({len(out)} rows, {n_nan} NaN)")


# ---------------------------------------------------------------------------
# Returns and mktcap filtering
# ---------------------------------------------------------------------------

def _filter_returns(r3000_dir: Path, output_dir: Path, permnos: list[int]) -> None:
    """Filter Russell 3000 returns_stocks.csv to the given permno subset."""
    src = r3000_dir / "returns_stocks.csv"
    dst = output_dir / "returns_stocks.csv"
    print(f"\nFiltering returns from {src} ...")
    df = pd.read_csv(src, index_col="date", parse_dates=True)
    permno_strs = [str(p) for p in permnos]
    keep = [c for c in permno_strs if c in df.columns]
    missing = len(permno_strs) - len(keep)
    if missing:
        print(f"  Warning: {missing} permnos not found in returns_stocks.csv (delisted before start date?).")
    out = df[keep].copy()
    out.index.name = "date"
    out.reset_index(inplace=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"  Wrote {out.shape[0]} rows x {len(keep)} permnos to {dst}")


def _filter_mktcap(r3000_dir: Path, output_dir: Path, permnos: list[int]) -> None:
    """Filter Russell 3000 mktcap_stocks.csv to the given permno subset."""
    src = r3000_dir / "mktcap_stocks.csv"
    dst = output_dir / "mktcap_stocks.csv"
    print(f"\nFiltering mktcap from {src} ...")
    df = pd.read_csv(src, index_col="date", parse_dates=True)
    permno_strs = [str(p) for p in permnos]
    keep = [c for c in permno_strs if c in df.columns]
    out = df[keep].copy()
    out.index.name = "date"
    out.reset_index(inplace=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(dst, index=False)
    print(f"  Wrote {out.shape[0]} rows x {len(keep)} permnos to {dst}")


# ---------------------------------------------------------------------------
# WRDS index returns
# ---------------------------------------------------------------------------

def _explore_index(conn: object, start_date: str, end_date: str) -> None:
    print("\n=== Exploration des colonnes d'indice disponibles dans WRDS ===\n")
    try:
        sample = conn.raw_sql(
            f"SELECT * FROM crsp_a_indexes.dsix "
            f"WHERE caldt BETWEEN '{start_date}' AND '{end_date}' "
            f"LIMIT 3"
        )
        print("Colonnes disponibles dans crsp_a_indexes.dsix :")
        print(list(sample.columns))
        print("\nExtrait :")
        print(sample.to_string(index=False))
    except Exception as e:
        print(f"  crsp_a_indexes.dsix non accessible : {e}")
    print(
        "\n→ La colonne Russell 1000 total return est probablement 'r1000ret'.\n"
        "  Relance avec --download-index --index-column r1000ret pour confirmer."
    )


def _download_index(
    conn: object,
    start_date: str,
    end_date: str,
    index_column: str,
    index_table: str,
    date_column: str,
    output_dir: Path,
) -> None:
    query = f"""
        SELECT {date_column} AS date, {index_column} AS ret
        FROM {index_table}
        WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY {date_column}
    """
    print(f"\nDownloading Russell 1000 index returns from {index_table}.{index_column} ...")
    df = conn.raw_sql(query, date_cols=["date"])
    df = df.dropna(subset=["date", "ret"])
    df["date"] = pd.to_datetime(df["date"])

    if df.empty:
        raise RuntimeError(
            f"Aucune donnée retournée depuis {index_table}.{index_column}.\n"
            f"Vérifiez le nom de colonne avec --explore-index."
        )

    out = df.rename(columns={"date": "Date", "ret": "russell1000"})
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    dst = output_dir / "returns_index.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"  Wrote {len(out)} rows to {dst}  ({out['Date'].iloc[0]} – {out['Date'].iloc[-1]})")


def _connect_wrds() -> object:
    try:
        import wrds
    except ImportError as exc:
        raise SystemExit(
            "The 'wrds' package is required. Install it with 'pip install wrds psycopg2-binary'."
        ) from exc
    try:
        from psycopg2 import OperationalError as PsycopgOperationalError
        from sqlalchemy.exc import OperationalError as SAOperationalError
    except ImportError:
        PsycopgOperationalError = Exception
        SAOperationalError = Exception

    print("Connecting to WRDS (you will be prompted for credentials)...")
    try:
        return wrds.Connection()
    except Exception as exc:
        raise SystemExit(f"Failed to connect to WRDS: {exc}") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Russell 1000 data from existing Russell 3000 data"
    )
    parser.add_argument(
        "--r3000-dir", type=Path, default=Path("financial_data/russell3000"),
        help="Directory containing Russell 3000 data (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("financial_data/russell1000"),
        help="Output directory for Russell 1000 data (default: %(default)s)",
    )
    parser.add_argument(
        "--size", type=int, default=1000,
        help="Number of stocks to keep (default: 1000 for Russell 1000)",
    )
    parser.add_argument(
        "--reconstitution-month", type=int, default=7,
        help="Month when new composition takes effect (default: 7 = July). "
             "Mktcap snapshot uses the last available month before this.",
    )
    parser.add_argument(
        "--skip-constituents", action="store_true",
        help="Skip constituent derivation (use existing files in --output-dir/constituants/)",
    )
    parser.add_argument(
        "--skip-returns", action="store_true",
        help="Skip returns/mktcap filtering (use existing files)",
    )

    parser.add_argument(
        "--build-index", action="store_true",
        help="Build synthetic Russell 1000 index from cap-weighted constituent returns (no WRDS needed)",
    )

    wrds_group = parser.add_argument_group("WRDS index download (optional — if official series available)")
    wrds_group.add_argument(
        "--explore-index", action="store_true",
        help="Explore available WRDS index columns, then exit",
    )
    wrds_group.add_argument(
        "--download-index", action="store_true",
        help="Download Russell 1000 index returns from WRDS",
    )
    wrds_group.add_argument(
        "--start-date", type=str, default=None,
        help="Start date for index download (YYYY-MM-DD)",
    )
    wrds_group.add_argument(
        "--end-date", type=str, default=None,
        help="End date for index download (YYYY-MM-DD)",
    )
    wrds_group.add_argument(
        "--index-table", type=str, default="crsp_a_indexes.dsix",
        help="WRDS table for index returns (default: %(default)s)",
    )
    wrds_group.add_argument(
        "--date-column", type=str, default="caldt",
        help="Date column in --index-table (default: %(default)s)",
    )
    wrds_group.add_argument(
        "--index-column", type=str, default="r1000ret",
        help="Return column for Russell 1000 total return (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Mode exploration WRDS ---
    if args.explore_index:
        if not args.start_date or not args.end_date:
            raise SystemExit("--explore-index requires --start-date and --end-date")
        conn = _connect_wrds()
        try:
            _explore_index(conn, args.start_date, args.end_date)
        finally:
            conn.close()
        return

    # --- Dérivation des constituants ---
    permnos: list[int] = []
    if not args.skip_constituents:
        print("=== Step 1: Deriving Russell 1000 constituents from mktcap ===")
        permnos = _derive_constituents(
            r3000_dir=args.r3000_dir,
            output_dir=args.output_dir,
            size=args.size,
            reconstitution_month=args.reconstitution_month,
        )
    else:
        # Load existing union file
        union_path = args.output_dir / "constituants" / "all_permnos.csv"
        if not union_path.exists():
            raise FileNotFoundError(f"--skip-constituents set but {union_path} not found")
        permnos = pd.read_csv(union_path, dtype={"permno": str})["permno"].astype(int).tolist()
        print(f"Loaded {len(permnos)} permnos from existing {union_path}")

    # --- Filtrage des rendements et mktcap ---
    if not args.skip_returns:
        print("\n=== Step 2: Filtering returns and mktcap ===")
        _filter_returns(args.r3000_dir, args.output_dir, permnos)
        _filter_mktcap(args.r3000_dir, args.output_dir, permnos)

    # --- Construction synthétique de l'indice ---
    if args.build_index:
        print("\n=== Step 3: Building synthetic Russell 1000 index ===")
        _build_synthetic_index(args.output_dir, args.reconstitution_month)

    # --- Téléchargement de l'indice WRDS ---
    if args.download_index:
        if not args.start_date or not args.end_date:
            raise SystemExit("--download-index requires --start-date and --end-date")
        print("\n=== Step 3: Downloading Russell 1000 index returns from WRDS ===")
        conn = _connect_wrds()
        try:
            _download_index(
                conn=conn,
                start_date=args.start_date,
                end_date=args.end_date,
                index_column=args.index_column,
                index_table=args.index_table,
                date_column=args.date_column,
                output_dir=args.output_dir,
            )
        finally:
            conn.close()

    # --- Récapitulatif ---
    print("\n=== Done ===")
    index_status = "✓" if (args.output_dir / "returns_index.csv").exists() else "✗ (run with --download-index)"
    print(f"  constituants/       ✓ ({len(permnos)} unique permnos)")
    print(f"  returns_stocks.csv  {'✓' if (args.output_dir / 'returns_stocks.csv').exists() else '✗'}")
    print(f"  mktcap_stocks.csv   {'✓' if (args.output_dir / 'mktcap_stocks.csv').exists() else '✗'}")
    print(f"  returns_index.csv   {index_status}")
    if not (args.output_dir / "returns_index.csv").exists():
        print(
            "\nNext step — build the synthetic index:\n"
            f"  python scripts/prepare_russell1000.py "
            f"--skip-constituents --skip-returns --build-index"
        )


if __name__ == "__main__":
    main()
