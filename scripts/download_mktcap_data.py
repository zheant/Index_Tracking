"""Download monthly market-cap data for Russell 3000 constituents from WRDS.

Market cap is computed as abs(prc) * shrout * 1000 from CRSP's monthly stock
file (crsp.msf).  The result is saved as a wide CSV (date x permno) under
financial_data/russell3000/mktcap_stocks.csv.

This file is consumed by universe.py to build market-cap weights passed to
the QUOB solver for market-cap-weighted K-medoids selection.

Typical usage
-------------
    python scripts/download_mktcap_data.py \\
        --permno-csv financial_data/russell3000/constituants/all_permnos.csv \\
        --start-date 2014-01-01 --end-date 2023-12-31
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

try:
    import wrds
except ImportError as exc:
    raise SystemExit(
        "The 'wrds' package is required. Install it with 'pip install wrds psycopg2-binary'."
    ) from exc

from psycopg2 import OperationalError as PsycopgOperationalError
from sqlalchemy.exc import OperationalError as SAOperationalError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunked(sequence: Sequence[int], chunk_size: int) -> Iterable[List[int]]:
    for start in range(0, len(sequence), chunk_size):
        yield list(sequence[start : start + chunk_size])


def _load_permnos(path: Path) -> List[int]:
    df = pd.read_csv(path, dtype={"permno": "string"})
    if "permno" not in df.columns:
        raise ValueError(f"Expected a 'permno' column in {path}")
    permnos = df["permno"].dropna().astype(str).str.strip()
    permnos = permnos[permnos != ""]
    unique_permnos = sorted({int(v) for v in permnos})
    if not unique_permnos:
        raise ValueError(f"No permnos found in {path}")
    return unique_permnos


# ---------------------------------------------------------------------------
# Market cap fetch (crsp.msf)
# ---------------------------------------------------------------------------

def _fetch_mktcap(
    conn: "wrds.Connection",
    permnos: Sequence[int],
    start_date: str,
    end_date: str,
    chunk_size: int,
) -> pd.DataFrame:
    """Fetch monthly market cap for permnos from crsp.msf.

    Market cap = abs(prc) * shrout * 1000.
    prc is negative in CRSP when the closing price is unavailable and the
    bid-ask midpoint is used instead; abs() handles both cases uniformly.
    shrout is in thousands of shares.
    """
    frames: list[pd.DataFrame] = []
    total_chunks = math.ceil(len(permnos) / chunk_size)
    for idx, chunk in enumerate(_chunked(permnos, chunk_size), start=1):
        permno_sql = ",".join(str(n) for n in chunk)
        query = f"""
            SELECT permno, date, abs(prc) * shrout * 1000 AS mktcap
            FROM crsp.msf
            WHERE permno IN ({permno_sql})
              AND date BETWEEN '{start_date}' AND '{end_date}'
              AND prc IS NOT NULL
              AND shrout IS NOT NULL
              AND shrout > 0
        """
        print(f"Downloading chunk {idx}/{total_chunks} (size={len(chunk)})...")
        frames.append(conn.raw_sql(query, date_cols=["date"]))

    if not frames:
        raise RuntimeError("No market-cap data returned from crsp.msf")

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["date", "mktcap"])
    data = data[data["mktcap"] > 0]
    data["permno"] = data["permno"].astype(int)
    data = data.sort_values(["date", "permno"]).drop_duplicates(
        subset=["date", "permno"], keep="last"
    )
    return data


def _pivot_mktcap(data: pd.DataFrame) -> pd.DataFrame:
    wide = data.pivot(index="date", columns="permno", values="mktcap").sort_index()
    wide.columns = [str(col) for col in wide.columns]
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    return wide


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _write_mktcap(wide: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = wide.reset_index()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    output.to_csv(destination, index=False)
    print(f"Wrote market-cap matrix ({wide.shape[0]} months x {wide.shape[1]} permnos) to {destination}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download monthly market-cap data for Russell 3000 constituents from WRDS"
    )
    parser.add_argument(
        "--permno-csv",
        type=Path,
        default=Path("financial_data/russell3000/constituants/all_permnos.csv"),
        help="Path to the CSV with all permnos (generated by prepare_russell_constituents.py)",
    )
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--chunk-size", type=int, default=200,
        help="Number of permnos per SQL query (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("financial_data/russell3000"),
        help="Destination folder (default: %(default)s)",
    )
    return parser.parse_args()


def _connect() -> "wrds.Connection":
    print("Connecting to WRDS (you will be prompted for credentials)...")
    try:
        return wrds.Connection()
    except (PsycopgOperationalError, SAOperationalError) as exc:
        raise SystemExit(
            f"Failed to authenticate with WRDS.\n"
            f"Verify your username/password and WRDS PostgreSQL access.\n"
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:
        raise SystemExit("Unexpected error while connecting to WRDS: " + str(exc)) from exc


def main() -> None:
    args = parse_args()

    permnos = _load_permnos(args.permno_csv)
    print(f"Loaded {len(permnos)} unique permnos from {args.permno_csv}")

    conn = _connect()
    try:
        data = _fetch_mktcap(
            conn=conn,
            permnos=permnos,
            start_date=args.start_date,
            end_date=args.end_date,
            chunk_size=args.chunk_size,
        )
    finally:
        conn.close()

    print(f"Fetched {len(data)} (permno, month) observations.")
    wide = _pivot_mktcap(data)
    _write_mktcap(wide, args.output_dir / "mktcap_stocks.csv")


if __name__ == "__main__":
    main()
