import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np


class Universe:
    """Manages the stock universe and data for a given training or test window.

    Loads returns, index returns, and market-cap data from disk once at
    construction.  Each call to ``new_universe()`` slices the relevant
    time window, applies the configured data-cleaning pipeline, and
    exposes the cleaned arrays to the solver.
    """

    def __init__(self, args) -> None:
        self.args = args
        self.missing_policy = getattr(args, "missing_policy", "strict")

        self.constituent_dir, self.constituent_years = self._discover_constituent_files()
        self._load_data()

        self.df_return = None
        self.df_index = None
        self.year = -1
        self.stock_list = self._update_stock_list()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        data_path = self.args.data_path
        index_name = self.args.index

        self.df_return_all = pd.read_csv(f"{data_path}/{index_name}/returns_stocks.csv")
        self.df_return_all["date"] = pd.to_datetime(self.df_return_all["date"])
        self.df_return_all.set_index("date", inplace=True)
        self.df_return_all.sort_index(inplace=True)

        self.df_index_all = pd.read_csv(f"{data_path}/{index_name}/returns_index.csv")
        self.df_index_all["Date"] = pd.to_datetime(self.df_index_all["Date"])
        self.df_index_all.set_index("Date", inplace=True)
        self.df_index_all.sort_index(inplace=True)

        mktcap_path = Path(f"{data_path}/{index_name}/mktcap_stocks.csv")
        if mktcap_path.exists():
            self.df_mktcap_all = pd.read_csv(mktcap_path)
            self.df_mktcap_all["date"] = pd.to_datetime(self.df_mktcap_all["date"])
            self.df_mktcap_all.set_index("date", inplace=True)
            self.df_mktcap_all.sort_index(inplace=True)
            print(f"Loaded market-cap data: {self.df_mktcap_all.shape[0]} months x {self.df_mktcap_all.shape[1]} permnos.")
        else:
            self.df_mktcap_all = None
            print(f"No market-cap data found at {mktcap_path}. QUOB will use QP weights.")

    # ------------------------------------------------------------------
    # Constituent management
    # ------------------------------------------------------------------

    def _effective_constituent_year(self, dt: pd.Timestamp) -> int:
        """Map a date to the constituent file year to use.

        Russell reconstitutes end-of-June; the new composition is active from
        July.  The file ``{year}.csv`` represents the post-June-{year}
        reconstitution.  For dates before July, we use the previous year's file.
        The ``reconstitution_month`` arg (default 7) is configurable.
        """
        month = getattr(self.args, "reconstitution_month", 7)
        return dt.year if dt.month >= month else dt.year - 1

    def _update_stock_list(self, ref_date: pd.Timestamp | None = None) -> list[str]:
        if ref_date is None:
            ref_date = pd.Timestamp(self.args.start_date)

        effective_year = self._effective_constituent_year(ref_date)
        if effective_year != self.year:
            self.year = effective_year
            selected_year = self._select_constituent_year(effective_year)
            filepath = self.constituent_dir / f"{selected_year}.csv"
            if selected_year != effective_year:
                print(f"Warning: constituents for {effective_year} unavailable; using {selected_year}.")
            self.stock_list = (
                pd.read_csv(filepath, dtype={"permno": str})["permno"].tolist()
            )
        return self.stock_list

    # ------------------------------------------------------------------
    # Window creation
    # ------------------------------------------------------------------

    def new_universe(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        training: bool = True,
    ) -> None:
        """Slice and clean the universe for a given time window.

        Parameters
        ----------
        start_datetime, end_datetime:
            Bounds of the window (inclusive).
        training:
            True for the optimisation window (full cleaning pipeline applied);
            False for the out-of-sample evaluation window (only hard-clip).
            The ``ref_date`` for constituent lookup is ``end_datetime`` in
            training mode and ``start_datetime`` in test mode, avoiding
            look-ahead bias.
        """
        start_datetime = pd.Timestamp(start_datetime)
        end_datetime = pd.Timestamp(end_datetime)

        ref_date = end_datetime if training else start_datetime
        self._update_stock_list(ref_date)
        print(
            f"Universe {'training' if training else 'test'} "
            f"[{start_datetime.date()} → {end_datetime.date()}]: "
            f"{len(self.stock_list)} constituents (ref: {ref_date.date()})."
        )

        valid_stocks = [s for s in self.stock_list if s in self.df_return_all.columns]
        missing = set(self.stock_list) - set(valid_stocks)
        if missing:
            print(f"Warning: {len(missing)} constituent(s) absent from returns data.")
        self.stock_list = valid_stocks

        ordered = [s for s in self.df_return_all.columns if s in self.stock_list]
        self.df_return = self.df_return_all.loc[start_datetime:end_datetime, ordered].copy()
        self.df_index = self.df_index_all.loc[start_datetime:end_datetime].copy()

        if self.missing_policy == "legacy":
            filled = int(self.df_return.isna().sum().sum())
            self.df_return = self.df_return.fillna(0)
            self.df_index = self.df_index.fillna(0)
            self.stock_list = list(self.df_return.columns)
            self.last_cleaning_stats = {
                "initial_shape": self.df_return.shape,
                "calendar_dates_removed": [],
                "values_filled": filled,
                "dropped_columns": [],
                "dropped_rows": [],
                "final_shape": self.df_return.shape,
            }
            print("Legacy policy: filled all NaNs with zero, kept all columns.")
            self.mktcap_weights = self.get_index_weights(ref_date)
            self.full_mktcap_weights = self.mktcap_weights
            self.pre_liquidity_columns = list(self.stock_list)
            self.year = -1
            return

        common_index = self.df_return.index.intersection(self.df_index.index)
        dropped_calendar = (set(self.df_return.index) | set(self.df_index.index)) - set(common_index)
        self.df_return = self.df_return.loc[common_index]
        self.df_index = self.df_index.loc[common_index]

        self._clean_data(
            target_cardinality=getattr(self.args, "cardinality", None),
            dropped_calendar_dates=sorted(dropped_calendar),
            training=training,
        )
        self.stock_list = list(self.df_return.columns)
        self.mktcap_weights = self.get_index_weights(ref_date)

        pre_liq = getattr(self, "_pre_liquidity_columns", None)
        if training and pre_liq is not None and len(pre_liq) > len(self.stock_list):
            self.pre_liquidity_columns = pre_liq
            self.full_mktcap_weights = self.get_index_weights(ref_date, stock_list=pre_liq)
        else:
            self.pre_liquidity_columns = list(self.stock_list)
            self.full_mktcap_weights = self.mktcap_weights

        # Reset constituent cache so the next call always reloads from disk.
        self.year = -1

    # ------------------------------------------------------------------
    # Market-cap weights
    # ------------------------------------------------------------------

    def get_index_weights(
        self, ref_date: pd.Timestamp, stock_list: list[str] | None = None
    ) -> np.ndarray | None:
        """Return market-cap weights for stock_list at ref_date.

        Uses the most recent monthly snapshot on or before ref_date.
        Stocks missing from the mktcap data receive weight 0.
        Returns None if no mktcap data is loaded or no snapshot precedes ref_date.
        """
        if self.df_mktcap_all is None:
            return None
        if stock_list is None:
            stock_list = self.stock_list

        available = self.df_mktcap_all.index[self.df_mktcap_all.index <= ref_date]
        if len(available) == 0:
            print(f"Warning: no market-cap snapshot on or before {ref_date.date()}. Using uniform weights.")
            return None

        snap = self.df_mktcap_all.loc[available[-1]]
        weights = np.array([float(snap.get(s, np.nan)) for s in stock_list], dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

        total = weights.sum()
        if total <= 0:
            print(f"Warning: all market caps are zero/missing at {available[-1].date()}. Using uniform weights.")
            return None

        weights /= total
        n_missing = (weights == 0).sum()
        if n_missing > 0:
            print(f"Market-cap weights at {available[-1].date()}: {n_missing}/{len(stock_list)} stocks missing (weight=0).")
        return weights

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_stocks_returns(self) -> pd.DataFrame:
        return self.df_return

    def get_index_returns(self) -> pd.Series:
        if isinstance(self.df_index, pd.DataFrame):
            return self.df_index.iloc[:, 0]
        return self.df_index

    def get_stock_name_in_order(self) -> pd.Index:
        return self.df_return.columns

    def get_number_of_stocks(self) -> int:
        return len(self.stock_list)

    # ------------------------------------------------------------------
    # Data cleaning
    # ------------------------------------------------------------------

    def _clean_data(
        self,
        target_cardinality: int | None = None,
        dropped_calendar_dates: list | None = None,
        training: bool = True,
    ) -> None:
        """Apply the strict data-cleaning pipeline.

        Training-only steps (avoid look-ahead in test mode):
          1. Drop stocks exceeding ``max_missing_frac`` NaN threshold.
          3. Liquidity filter: drop stocks with fewer than ``min_trading_frac``
             non-zero return days.
          4. Winsorisation at ±``winsor_sigma`` σ per stock (computed on
             trading days only, ignoring fill-zero days).

        Applied in both training and test:
          2. Fill remaining NaNs with 0 (no-trade day = zero return).
          2b. Hard-clip at ±``hard_clip`` (fixed threshold, no future stats).
          5. Drop rows where the index return is missing.
          6. Verify cardinality (training only).
        """
        max_missing_frac = getattr(self.args, "max_missing_frac", 0.10)
        min_trading_frac = getattr(self.args, "min_trading_frac", 0.50)
        winsor_sigma = getattr(self.args, "winsor_sigma", 3.0)
        hard_clip = getattr(self.args, "hard_clip", 1.0)

        stats = {
            "initial_shape": self.df_return.shape,
            "calendar_dates_removed": dropped_calendar_dates or [],
        }

        # Step 1 — drop stocks with too many missing values (training only)
        if training:
            missing_frac = self.df_return.isna().mean(axis=0)
            keep = missing_frac <= max_missing_frac
            dropped = missing_frac[~keep].index.tolist()
            self.df_return = self.df_return.loc[:, keep]
            stats["dropped_columns"] = dropped
            print(f"Dropped {len(dropped)} stocks exceeding {max_missing_frac:.0%} missing-data threshold.")
        else:
            stats["dropped_columns"] = []

        # Step 2 — fill NaNs with 0 (no-trade day = zero return)
        nan_count = int(self.df_return.isna().sum().sum())
        self.df_return = self.df_return.fillna(0)
        stats["values_filled"] = nan_count
        print(f"Filled {nan_count} missing values with 0 (no-trade days).")

        # Step 2b — hard-clip extreme returns (both training and test)
        if hard_clip > 0:
            clipped = int((self.df_return.abs() > hard_clip).sum().sum())
            self.df_return = self.df_return.clip(lower=-hard_clip, upper=hard_clip)
            if clipped > 0:
                print(f"Hard-clipped {clipped} extreme return(s) to ±{hard_clip:.0%}.")

        # Step 3 — liquidity filter (training only)
        # Save pre-liquidity column list for Pool B market-cap attribution.
        if training:
            self._pre_liquidity_columns = list(self.df_return.columns)
            nonzero_frac = (self.df_return != 0).mean(axis=0)
            liquid = nonzero_frac >= min_trading_frac
            illiquid = nonzero_frac[~liquid].index.tolist()
            self.df_return = self.df_return.loc[:, liquid]
            stats["dropped_illiquid"] = illiquid
            print(f"Dropped {len(illiquid)} illiquid stocks (< {min_trading_frac:.0%} non-zero days).")
        else:
            stats["dropped_illiquid"] = []

        # Step 4 — winsorise at ±σ per stock (training only, on trading days)
        if training and winsor_sigma > 0:
            trading = self.df_return.replace(0, np.nan)
            mean = trading.mean(axis=0)
            std = trading.std(axis=0)
            self.df_return = self.df_return.clip(
                lower=mean - winsor_sigma * std,
                upper=mean + winsor_sigma * std,
                axis=1,
            )
            print(f"Winsorised returns at ±{winsor_sigma}σ per stock (trading days only).")
        elif not training and winsor_sigma > 0:
            print("Winsorisation skipped (out-of-sample evaluation).")

        # Step 5 — drop rows with missing index values
        rows_before = self.df_return.shape[0]
        valid_rows = self.df_index.notna().all(axis=1)
        dropped_rows = self.df_return.index[~valid_rows].tolist()
        self.df_return = self.df_return.loc[valid_rows]
        self.df_index = self.df_index.loc[valid_rows]
        stats["dropped_rows"] = dropped_rows
        print(f"Dropped {rows_before - self.df_return.shape[0]} rows with missing index values.")

        # Step 6 — cardinality check (training only)
        if training and target_cardinality is not None and self.df_return.shape[1] < target_cardinality:
            raise ValueError(
                f"Universe cardinality after cleaning ({self.df_return.shape[1]}) is below "
                f"the target ({target_cardinality}). Relax --max_missing_frac or --min_trading_frac."
            )

        stats["final_shape"] = self.df_return.shape
        stats["calendar_count"] = int(self.df_return.shape[0])
        stats["calendar_hash"] = self._hash_calendar(self.df_return.index)
        self.last_cleaning_stats = stats

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_calendar(index: pd.Index) -> str:
        payload = "|".join(ts.isoformat() for ts in index)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _discover_constituent_files(self) -> tuple[Path, list[int]]:
        data_path = self.args.data_path
        for directory in [
            Path(f"{data_path}/{self.args.index}/constituants"),
            Path(f"{data_path}/{self.args.index}/constituants_raw"),
        ]:
            if not directory.exists():
                continue
            years = sorted(
                int(p.stem) for p in directory.glob("*.csv") if p.stem.isdigit()
            )
            if years:
                return directory, years
        raise FileNotFoundError(
            "No constituent CSV files found under 'constituants' or 'constituants_raw'."
        )

    def _select_constituent_year(self, requested_year: int) -> int:
        if requested_year in self.constituent_years:
            return requested_year
        if requested_year < self.constituent_years[0]:
            return self.constituent_years[0]
        prior = [y for y in self.constituent_years if y <= requested_year]
        return prior[-1] if prior else self.constituent_years[-1]
