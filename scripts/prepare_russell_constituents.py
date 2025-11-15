"""Utility helpers to stage Russell 3000 constituent files for the pipeline.

The historical Russell 3000 compositions that were added to the repository live
under ``composition historique russel3000`` and only contain a single ``permno``
column.  The portfolio construction code expects to find per-year constituent
files under ``financial_data/<index>/constituants`` with the identifiers stored
as strings.  This script copies the raw CSVs into the expected location while
normalising their contents and also writes a convenience file listing the union
of every permno observed across the available years.

Usage
-----
Activate your virtual environment, move to the repository root and execute::

    python scripts/prepare_russell_constituents.py

    Pass ``--source`` or ``--output`` if you keep the CSVs elsewhere.  Use
    ``--union-name`` to customise the aggregated permno file that will be
    generated alongside the yearly snapshots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Russell 3000 constituent files")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("composition historique russel3000"),
        help="Folder containing the raw Russell 3000 CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("financial_data/russel3000/constituants"),
        help="Destination folder for the normalised constituent files (default: %(default)s)",
    )
    parser.add_argument(
        "--union-name",
        default="all_permnos.csv",
        help="File name for the aggregated permno list (default: %(default)s)",
    )
    return parser.parse_args()


def normalise_constituents(source: Path, destination: Path, union_name: str) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Source directory '{source}' not found")

    destination.mkdir(parents=True, exist_ok=True)

    all_permnos: set[str] = set()

    csv_files = sorted(p for p in source.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{source}'")

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, dtype={"permno": str})
        if "permno" not in df.columns:
            raise ValueError(f"Column 'permno' missing from {csv_file}")

        df = df[["permno"]].dropna().drop_duplicates()
        df["permno"] = df["permno"].astype(str).str.strip()
        df = df[df["permno"] != ""]

        all_permnos.update(df["permno"].tolist())

        year_output = destination / csv_file.name
        df.sort_values("permno").to_csv(year_output, index=False)
        print(f"Wrote {len(df)} permnos to {year_output}")

    union_name = union_name if union_name.endswith(".csv") else f"{union_name}.csv"
    union_path = destination / union_name
    pd.DataFrame(sorted(all_permnos), columns=["permno"]).to_csv(union_path, index=False)
    print(f"Saved {len(all_permnos)} unique permnos to {union_path}")

    return union_path


def main() -> None:
    args = parse_args()
    normalise_constituents(args.source, args.output, args.union_name)


if __name__ == "__main__":
    main()
