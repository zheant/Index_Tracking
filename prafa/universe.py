from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


class Universe:
    """Data access layer for the optimisation workflow."""

    def __init__(self, args) -> None:
        self.args = args
        self._load_time_series()

        self.first_available_date = max(
            self.df_return_all.index.min(),
            self.df_index_all.index.min(),
        )
        self.initial_year = max(
            pd.Timestamp(self.args.start_date).year,
            self.first_available_date.year,
        )

        self.df_return = pd.DataFrame()
        self.df_index = pd.Series(dtype=float)
        self.year: int | None = None
        self.stock_list: List[str] = self.update_stock_list(None)

    def _load_time_series(self) -> None:
        base_path = Path("financial_data") / self.args.index

        returns_path = base_path / "returns_stocks.csv"
        index_path = base_path / "returns_index.csv"

        self.df_return_all = pd.read_csv(returns_path)
        returns_date_col = "date" if "date" in self.df_return_all.columns else "Date"
        self.df_return_all[returns_date_col] = pd.to_datetime(self.df_return_all[returns_date_col])
        if returns_date_col != "date":
            self.df_return_all.rename(columns={returns_date_col: "date"}, inplace=True)
        self.df_return_all.set_index("date", inplace=True)
        self.df_return_all.sort_index(inplace=True)

        index_df = pd.read_csv(index_path)
        date_column = "Date" if "Date" in index_df.columns else "date"
        index_df[date_column] = pd.to_datetime(index_df[date_column])
        index_df.set_index(date_column, inplace=True)
        index_df.sort_index(inplace=True)

        if index_df.empty:
            raise ValueError(f"Le fichier d'indice est vide : {index_path}")

        value_columns = [col for col in index_df.columns if col not in {date_column}]
        if len(value_columns) == 0:
            # All columns were date columns (unlikely but defensive); treat the only column as values.
            value_series = index_df.squeeze("columns")
        elif len(value_columns) == 1:
            value_series = index_df[value_columns[0]]
        else:
            raise ValueError(
                "Le fichier returns_index.csv doit contenir une seule colonne de valeurs. "
                f"Colonnes détectées : {value_columns}"
            )

        self.df_index_all = value_series.astype(float)

    def update_stock_list(self, current_datetime: datetime | None) -> List[str]:
        """Refresh the list of tradable securities for the requested year."""
        constituents_dir = Path("financial_data") / self.args.index / "constituants"

        def load_year(year: int) -> List[str]:
            csv_path = constituents_dir / f"{year}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Fichier de composition manquant : {csv_path}")
            df = pd.read_csv(csv_path, dtype={"permno": str})
            return df["permno"].dropna().astype(str).str.strip().tolist()

        if current_datetime is None:
            # Initial load when the universe is created.
            self.year = self.initial_year
            self.stock_list = load_year(self.year)
        elif current_datetime.year != self.year:
            self.year = current_datetime.year
            self.stock_list = load_year(self.year)

        return self.stock_list

    def new_universe(self, start_datetime: datetime, end_datetime: datetime, training: bool = True) -> None:
        start = max(pd.Timestamp(start_datetime), self.first_available_date)
        end = pd.Timestamp(end_datetime)

        if training:
            self.update_stock_list(end)
        else:
            self.update_stock_list(start)

        valid_stocks = [stock for stock in self.stock_list if stock in self.df_return_all.columns]
        missing_stocks = sorted(set(self.stock_list) - set(valid_stocks))
        if missing_stocks:
            print(
                "⚠️ Les actions suivantes ne sont pas dans les données de rendement : "
                + ", ".join(missing_stocks)
            )

        ordered_stocks = [stock for stock in self.df_return_all.columns if stock in valid_stocks]
        returns_slice = self.df_return_all.loc[start:end, ordered_stocks].copy()

        # Drop columns that never trade (all NaN) or stay constant across the
        # training window. They only contribute noise to the optimisation while
        # dramatically increasing the cost of building the distance matrix.
        non_missing_mask = returns_slice.notna().any(axis=0)
        filtered_returns = returns_slice.loc[:, non_missing_mask]

        variance_mask = filtered_returns.var(axis=0, skipna=True) > 0
        filtered_returns = filtered_returns.loc[:, variance_mask]

        dropped = [
            stock
            for stock in ordered_stocks
            if stock not in filtered_returns.columns
        ]
        if dropped:
            print(
                "⚠️ Suppression des titres sans activité ou constants sur la fenêtre : "
                + ", ".join(dropped)
            )

        returns_slice = filtered_returns.fillna(0)
        index_slice = self.df_index_all.loc[start:end].copy().fillna(0)

        common_index = returns_slice.index.intersection(index_slice.index)
        self.df_return = returns_slice.loc[common_index]
        self.df_index = index_slice.loc[common_index]
        self.stock_list = list(self.df_return.columns)

    def get_stocks_returns(self) -> pd.DataFrame:
        return self.df_return

    def get_index_returns(self) -> pd.Series:
        return pd.Series(self.df_index, index=self.df_return.index)

    def get_data_start_date(self) -> pd.Timestamp:
        return self.first_available_date

    def get_stock_namme_in_order(self) -> List[str]:  # noqa: D401 - kept for backward compatibility.
        return self.df_return.columns.tolist()

    def get_number_of_stocks(self) -> int:
        return len(self.stock_list)
