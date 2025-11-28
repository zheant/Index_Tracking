# Codebase discrepancies and methodological concerns

## Universe construction (`prafa/universe.py`)

1. **Calendar alignment applies twice but still permits index-driven drops.** The calendar intersection occurs before cleaning, yet `data_cleaning` later drops index NaNs and re-intersects, potentially shrinking the date range unpredictably relative to training and stored weights; no record of removed dates is persisted for downstream analysis. 
2. **Forward/back-fill counts are printed but not recorded.** The limited `ffill`/`bfill` step reports the number of filled cells, but the information is not returned or logged per window, preventing reproducibility of how much imputation was used when universes are stitched together in analysis.
3. **Full-coverage filter can break target cardinality.** Dropping every column that still contains a NaN after limited fills keeps only perfectly observed tickers. Training and backtesting still request a fixed cardinality (e.g., 300), but the cleaned universe may contain fewer names, causing the optimizer either to fail or to solve a different problem without warning.
4. **Silent weight decay due to truncation/padding.** Because universes are cleaned per window, the weight vectors stored from training may no longer align with the cleaned backtest columns. `_align_weights` trims or pads weights to match the new column set but never renormalizes, so portfolio exposure can drop below 100% without any audit trail, distorting returns.
5. **Constituent list ignores intra-window changes.** `update_stock_list` loads a single year’s constituent list based on the start/end date, then `dropna(axis=1)` removes tickers missing data anywhere in the entire window. This approach effectively allows look-ahead to end-of-window constituents while eliminating mid-window entrants/exits, mixing future information with survivorship bias.
6. **Row drops are driven by stock gaps only.** After aligning calendars, rows are dropped when `df_return` retains NaNs, but index rows removed via `dropna()` also shrink the window. No safeguard exists to ensure the same date set was used when the original portfolios were trained, so backtest evaluation may be on a different timeline than the optimization assumed.

## Result analysis (`scripts/analyze_results.py`)

1. **Weight alignment still hides universe drift.** When the cleaned test universe has fewer tickers than the stored weights, `_align_weights` zeros any missing tickers without renormalizing, meaning the cumulative return curve is computed with less than full capital and can be biased downward.
2. **Assertion only checks date alignment, not coverage.** The analysis asserts that `X_test` and `Y_test` share the same index but doesn’t verify that the cleaned universe still satisfies the target cardinality or that weights sum to one after truncation/padding, so tracking error and cumulative returns may reflect an unintended, underinvested portfolio.
