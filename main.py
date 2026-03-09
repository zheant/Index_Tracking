import argparse
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from prafa.portfolio import Portfolio
from prafa.universe import Universe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index replication via k-medoid selection (QUOB / ReplicaTOR)."
    )

    # --- Data paths ---
    parser.add_argument("--data_path", type=str, default="financial_data")
    parser.add_argument("--result_path", type=str, default="results")

    # --- Solver ---
    parser.add_argument("--solution_name", type=str, default="quob",
                        choices=["quob", "quob_stratified"])
    parser.add_argument("--cardinality", type=int, default=50,
                        help="Target number of stocks K (default: 50)")
    parser.add_argument("--replicator_cores", type=int,
                        default=int(os.environ.get("REPLICATOR_CORES", 8)),
                        help="OpenMP threads for ReplicaTOR (overrides REPLICATOR_CORES env)")
    parser.add_argument("--time_limit", type=float, default=300.0,
                        help="Time limit in seconds for the ReplicaTOR solve (default: 300)")
    parser.add_argument("--d_scale", type=float, default=1.0,
                        help="D_scale_factor for ReplicaTOR: weight of the dispersion term (default: 1.0)")
    parser.add_argument("--distance_method", type=str, choices=["dcor", "pearson"], default="dcor",
                        help="Distance metric for the solver matrix (default: dcor)")

    # --- Index and date range ---
    parser.add_argument("--index", type=str, default="sp500",
                        choices=["sp500", "russell3000", "russell1000"])
    parser.add_argument("--start_date", type=str, default="2014-01-02")
    parser.add_argument("--end_date", type=str, default="2023-12-31")
    parser.add_argument("--T", type=int, default=3,
                        help="Training window length in years (default: 3)")
    parser.add_argument("--rebalancing", type=int, default=12,
                        help="Rebalancing frequency in months (default: 12)")

    # --- Missing-data policy ---
    parser.add_argument("--missing_policy", type=str,
                        choices=["auto", "strict", "legacy"], default="auto",
                        help="auto: legacy for SP500, strict otherwise")
    parser.add_argument("--reconstitution_month", type=int, default=7,
                        help="First month where the new Russell composition is active (default: 7)")
    parser.add_argument("--max_missing_frac", type=float, default=0.10,
                        help="Maximum fraction of missing days allowed per stock (default: 0.10)")
    parser.add_argument("--min_trading_frac", type=float, default=0.50,
                        help="Minimum fraction of non-zero return days required (default: 0.50)")
    parser.add_argument("--winsor_sigma", type=float, default=3.0,
                        help="Winsorisation threshold in σ per stock; 0 to disable (default: 3.0)")
    parser.add_argument("--hard_clip", type=float, default=1.0,
                        help="Hard clip at ±fraction before winsorisation; 0 to disable (default: 1.0)")

    args = parser.parse_args()

    if args.missing_policy == "auto":
        args.missing_policy = "legacy" if args.index.lower() == "sp500" else "strict"

    portfolio_duration = relativedelta(years=args.T)
    time_increment = relativedelta(months=args.rebalancing)

    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)

    dates = [start_date]
    current = start_date + time_increment
    while current < end_date:
        dates.append(current)
        current += time_increment
    if dates[-1] != end_date:
        dates.append(end_date)

    portfolio = Portfolio(Universe(args))
    for rebalancing_date in dates:
        train_start = rebalancing_date - portfolio_duration
        portfolio.rebalance_portfolio(train_start, rebalancing_date)
        print(f"Rebalanced: training [{train_start.date()} → {rebalancing_date.date()}]")

    portfolio.save_portfolio()


if __name__ == "__main__":
    main()
