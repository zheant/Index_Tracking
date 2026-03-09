"""Exploratory queries against the WRDS Russell dataset.

Used during development to understand the FTSE Russell WRDS schema.
Requires a valid WRDS account and the `wrds` Python package.

Usage:
    python scripts/explore_wrds_russell.py
"""


def main() -> None:
    try:
        import wrds
    except ImportError as exc:
        raise SystemExit("The 'wrds' package is required: pip install wrds psycopg2-binary") from exc

    conn = wrds.Connection()
    try:
        print("=== Date range of idx_holdings_us ===")
        df = conn.raw_sql("""
            SELECT MIN(date) AS min_date, MAX(date) AS max_date, COUNT(DISTINCT date) AS n_dates
            FROM ftsesamp_russell_us.idx_holdings_us
        """)
        print(df.to_string())

        print("\n=== R3000 and R1000 weight sums at 2014-01-31 ===")
        df2 = conn.raw_sql("""
            SELECT
                SUM(r3000_wt) AS sum_r3000_wt,
                SUM(r1000_wt) AS sum_r1000_wt,
                COUNT(*) AS n_stocks,
                SUM(CASE WHEN russell1000 = 'Y' THEN 1 ELSE 0 END) AS n_r1000,
                SUM(CASE WHEN russell2000 = 'Y' THEN 1 ELSE 0 END) AS n_r2000
            FROM ftsesamp_russell_us.idx_holdings_us
            WHERE date = '2014-01-31'
        """)
        print(df2.to_string())

        print("\n=== Cap-weighted R3000 returns (sample dates) ===")
        df3 = conn.raw_sql("""
            SELECT date,
                SUM(return * r3000_wt) / 100.0 AS r3000_ret,
                SUM(return * r1000_wt) / 100.0 AS r1000_ret
            FROM ftsesamp_russell_us.idx_holdings_us
            WHERE date BETWEEN '2014-01-01' AND '2014-01-10'
              AND r3000_wt IS NOT NULL
            GROUP BY date
            ORDER BY date
        """)
        print(df3.to_string())
    finally:
        conn.close()


if __name__ == "__main__":
    main()
