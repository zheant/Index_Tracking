"""Download Russell 3000 constituent data and official index returns from WRDS.

This helper queries the CRSP daily stock file for every permno listed in the
normalized Russell 3000 constituent CSVs and downloads the official Russell 3000
total-return index from CRSP's daily index file (crsp_a_indexes.dsix).

Typical workflow
----------------
1. Run --explore-index once to identify the correct table and column names
   for the official Russell 3000 series in your WRDS subscription::

       python scripts/download_wrds_russell_data.py --explore-index \\
           --start-date 2014-01-01 --end-date 2023-12-31

2. Then run the full download with the confirmed column name::

       python scripts/download_wrds_russell_data.py \\
           --permno-csv financial_data/russell3000/constituants/all_permnos.csv \\
           --start-date 2014-01-01 --end-date 2023-12-31 \\
           --index-column r3000ret

Install the ``wrds`` and ``psycopg2-binary`` packages beforehand::

    pip install wrds psycopg2-binary
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

try:
    import wrds
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'wrds' package is required. Install it with 'pip install wrds psycopg2-binary'."
    ) from exc

from psycopg2 import OperationalError as PsycopgOperationalError
from sqlalchemy.exc import OperationalError as SAOperationalError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunked(sequence: Sequence[int], chunk_size: int) -> Iterable[List[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(sequence), chunk_size):
        yield list(sequence[start : start + chunk_size])


def _load_permnos(path: Path) -> List[int]:
    df = pd.read_csv(path, dtype={"permno": "string"})
    if "permno" not in df.columns:
        raise ValueError(f"Expected a 'permno' column in {path}")
    permnos = df["permno"].dropna().astype(str).str.strip()
    permnos = permnos[permnos != ""]
    unique_permnos = sorted({int(value) for value in permnos})
    if not unique_permnos:
        raise ValueError(f"No permnos found in {path}")
    return unique_permnos


# ---------------------------------------------------------------------------
# Stock returns (crsp.dsf)
# ---------------------------------------------------------------------------

def _fetch_crsp_returns(
    conn: "wrds.Connection",
    permnos: Sequence[int],
    start_date: str,
    end_date: str,
    chunk_size: int,
) -> pd.DataFrame:
    """Fetch daily stock returns for ``permnos`` from CRSP.DSF."""
    frames: list[pd.DataFrame] = []
    for idx, chunk in enumerate(_chunked(permnos, chunk_size), start=1):
        permno_sql = ",".join(str(n) for n in chunk)
        query = f"""
            select permno, date, ret
            from crsp.dsf
            where permno in ({permno_sql})
              and date between '{start_date}' and '{end_date}'
        """
        print(f"Downloading chunk {idx} / {math.ceil(len(permnos) / chunk_size)} (size={len(chunk)})...")
        frames.append(conn.raw_sql(query, date_cols=["date"]))

    if not frames:
        raise RuntimeError("No data returned from CRSP.DSF")

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["date"])
    data["permno"] = data["permno"].astype(int)
    data = data.sort_values(["date", "permno"]).drop_duplicates(subset=["date", "permno"], keep="last")
    return data


def _pivot_returns(data: pd.DataFrame) -> pd.DataFrame:
    wide = data.pivot(index="date", columns="permno", values="ret").sort_index()
    wide.columns = [str(col) for col in wide.columns]
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    return wide


# ---------------------------------------------------------------------------
# Official Russell 3000 index returns (crsp_a_indexes.dsix)
# ---------------------------------------------------------------------------

def explore_index_tables(conn: "wrds.Connection", start_date: str, end_date: str) -> None:
    """Affiche les tables et colonnes WRDS susceptibles de contenir l'indice Russell 3000.

    À exécuter une fois avec --explore-index pour identifier le bon nom de colonne
    avant de lancer le téléchargement complet.
    """
    print("\n=== Exploration des tables d'indices disponibles dans WRDS ===\n")

    # 1. Lister les tables contenant "index" ou "russell" dans les schémas CRSP
    schema_query = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema IN ('crsp', 'crsp_a_indexes', 'crsp_q_indexes')
          AND (table_name ILIKE '%index%'
               OR table_name ILIKE '%dsi%'
               OR table_name ILIKE '%russell%')
        ORDER BY table_schema, table_name
    """
    try:
        tables = conn.raw_sql(schema_query)
        print("Tables candidates :")
        print(tables.to_string(index=False))
    except Exception as e:
        print(f"  Impossible de lister les tables : {e}")

    # 2. Inspecter crsp_a_indexes.dsix — table la plus probable pour Russell
    print("\n=== Colonnes de crsp_a_indexes.dsix ===")
    try:
        sample = conn.raw_sql(
            f"SELECT * FROM crsp_a_indexes.dsix "
            f"WHERE caldt BETWEEN '{start_date}' AND '{end_date}' "
            f"LIMIT 3"
        )
        print("Colonnes disponibles :", list(sample.columns))
        print("\nExtrait :")
        print(sample.to_string(index=False))
    except Exception as e:
        print(f"  crsp_a_indexes.dsix non accessible : {e}")

    # 3. Essayer aussi crsp.dsi (indices de marché CRSP)
    print("\n=== Colonnes de crsp.dsi ===")
    try:
        sample = conn.raw_sql(
            f"SELECT * FROM crsp.dsi "
            f"WHERE date BETWEEN '{start_date}' AND '{end_date}' "
            f"LIMIT 3"
        )
        print("Colonnes disponibles :", list(sample.columns))
        print("\nExtrait :")
        print(sample.to_string(index=False))
    except Exception as e:
        print(f"  crsp.dsi non accessible : {e}")

    print(
        "\n→ Identifie la colonne Russell 3000 total return dans les résultats ci-dessus "
        "et relance avec --index-column <nom_colonne>."
    )


def _fetch_official_index(
    conn: "wrds.Connection",
    start_date: str,
    end_date: str,
    index_table: str,
    date_column: str,
    index_column: str,
) -> pd.DataFrame:
    """Télécharge la série de rendements officielle depuis WRDS.

    Parameters
    ----------
    index_table:
        Table WRDS contenant l'indice, ex. 'crsp_a_indexes.dsix'.
    date_column:
        Colonne de date dans cette table, ex. 'caldt' ou 'date'.
    index_column:
        Colonne de rendement Russell 3000 total return, ex. 'r3000ret'.
    """
    query = f"""
        SELECT {date_column} AS date, {index_column} AS ret
        FROM {index_table}
        WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY {date_column}
    """
    print(f"Downloading official index from {index_table} (column: {index_column})...")
    df = conn.raw_sql(query, date_cols=["date"])
    df = df.dropna(subset=["date", "ret"])
    df["date"] = pd.to_datetime(df["date"])

    if df.empty:
        raise RuntimeError(
            f"Aucune donnée retournée depuis {index_table}.{index_column} "
            f"pour la période {start_date}–{end_date}. "
            f"Vérifiez le nom de table/colonne avec --explore-index."
        )

    print(f"  → {len(df)} observations téléchargées "
          f"({df['date'].min().date()} – {df['date'].max().date()})")
    return df


def _format_index_for_pipeline(df: pd.DataFrame, column_name: str = "russell3000") -> pd.DataFrame:
    """Met en forme la série d'indice au format attendu par le pipeline."""
    out = df[["date", "ret"]].copy()
    out = out.rename(columns={"date": "Date", "ret": column_name})
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_returns(wide: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = wide.reset_index()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    output.to_csv(destination, index=False)
    print(f"Wrote stock return matrix to {destination}")


def _write_index(index_df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(destination, index=False)
    print(f"Wrote index returns to {destination}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Russell 3000 data from WRDS")
    parser.add_argument(
        "--permno-csv",
        type=Path,
        default=Path("financial_data/russell3000/constituants/all_permnos.csv"),
        help="Path to the CSV generated by prepare_russell_constituents.py",
    )
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--chunk-size", type=int, default=200,
        help="Number of permnos per SQL query (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("financial_data/russell3000"),
        help="Destination folder for generated CSV files",
    )

    # --- Index officiel ---
    index_group = parser.add_argument_group("Official index options")
    index_group.add_argument(
        "--explore-index", action="store_true",
        help="Explore available WRDS tables/columns for Russell 3000 index, then exit",
    )
    index_group.add_argument(
        "--index-table", type=str, default="crsp_a_indexes.dsix",
        help="WRDS table containing the official index (default: %(default)s)",
    )
    index_group.add_argument(
        "--date-column", type=str, default="caldt",
        help="Date column in --index-table (default: %(default)s)",
    )
    index_group.add_argument(
        "--index-column", type=str, default=None,
        help=(
            "Return column for Russell 3000 total return in --index-table "
            "(ex: r3000ret). Required unless --explore-index is set."
        ),
    )
    index_group.add_argument(
        "--skip-index", action="store_true",
        help="Skip index download entirely (stock returns only)",
    )

    return parser.parse_args()


def _connect() -> "wrds.Connection":
    print("Connecting to WRDS (you will be prompted for credentials)...")
    try:
        return wrds.Connection()
    except (PsycopgOperationalError, SAOperationalError) as exc:
        raise SystemExit(
            "Failed to authenticate with WRDS.\n"
            "Verify your username/password, ensure any required VPN is active, "
            "and confirm your WRDS account is provisioned for PostgreSQL access.\n"
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:
        raise SystemExit("Unexpected error while connecting to WRDS: " + str(exc)) from exc


def main() -> None:
    args = parse_args()

    conn = _connect()

    try:
        # --- Mode exploration ---
        if args.explore_index:
            explore_index_tables(conn, args.start_date, args.end_date)
            return

        # --- Téléchargement des rendements des stocks ---
        permnos = _load_permnos(args.permno_csv)
        print(f"Loaded {len(permnos)} unique permnos from {args.permno_csv}")

        returns_long = _fetch_crsp_returns(
            conn=conn,
            permnos=permnos,
            start_date=args.start_date,
            end_date=args.end_date,
            chunk_size=args.chunk_size,
        )
        returns_wide = _pivot_returns(returns_long)
        _write_returns(returns_wide, args.output_dir / "returns_stocks.csv")

        # --- Téléchargement de l'indice officiel ---
        if not args.skip_index:
            if args.index_column is None:
                raise SystemExit(
                    "Spécifie --index-column <nom_colonne> pour le rendement Russell 3000, "
                    "ou utilise --explore-index pour identifier le bon nom de colonne, "
                    "ou --skip-index pour ignorer l'indice."
                )
            index_long = _fetch_official_index(
                conn=conn,
                start_date=args.start_date,
                end_date=args.end_date,
                index_table=args.index_table,
                date_column=args.date_column,
                index_column=args.index_column,
            )
            index_df = _format_index_for_pipeline(index_long)
            _write_index(index_df, args.output_dir / "returns_index.csv")
        else:
            print("Skipping index download (--skip-index).")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
