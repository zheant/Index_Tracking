import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np





class Universe():
    
    
    def __init__(
        self,
        args,
    ) :
        self.args = args
        self.missing_policy = getattr(args, "missing_policy", "strict")

        self.constituent_dir, self.constituent_years = self._discover_constituent_files()
        
        #données sur toutes l'historique
        self.initialisation_donnes()

        #timeseries sur la periode en cours
        self.df_return = None
        self.df_index = None

        self.stock_list = self.update_stock_list(None)
    

    def initialisation_donnes(self):
        #données sur toutes l'historique
        data_path = self.args.data_path
        self.df_return_all = pd.read_csv(f"{data_path}/{self.args.index}/returns_stocks.csv")  #return des stocks
        self.df_return_all['date'] = pd.to_datetime(self.df_return_all['date'])
        self.df_return_all.set_index('date', inplace=True)
        self.df_return_all.sort_index(inplace=True)

        self.df_index_all = pd.read_csv(f"{data_path}/{self.args.index}/returns_index.csv")   #return de l'indice
        self.df_index_all['Date'] = pd.to_datetime(self.df_index_all['Date'])
        self.df_index_all.set_index('Date', inplace=True)
        self.df_index_all.sort_index(inplace=True)

    
    def _effective_constituent_year(self, dt: pd.Timestamp) -> int:
        """Convertit une date en année de fichier constituant à utiliser.

        Le Russell 3000 se reconstitue fin juin. Les fichiers {year}.csv
        représentent la composition post-reconstitution de juin {year},
        effective à partir de juillet {year}. Pour une date antérieure à
        juillet, la composition connue est celle de la reconstitution de
        l'année précédente.

        Le mois de reconstitution est configurable via args.reconstitution_month
        (défaut : 7 = juillet, premier mois où la nouvelle composition est active).
        """
        reconstitution_month = getattr(self.args, "reconstitution_month", 7)
        if dt.month < reconstitution_month:
            return dt.year - 1
        return dt.year

    def update_stock_list(self, datetime : datetime = None):
        def load_constituents(year: int) -> list[str]:
            selected_year = self._select_constituent_year(year)
            filepath = self.constituent_dir / f"{selected_year}.csv"
            df = pd.read_csv(filepath, dtype={"permno": str})["permno"]
            if selected_year != year:
                print(
                    f"⚠️ Constituents for {year} unavailable; using {selected_year} from '{self.constituent_dir}'."
                )
            return df.tolist()

        if datetime is None:
            # Appel depuis le constructeur : initialisation sur start_date.
            effective_year = self._effective_constituent_year(pd.Timestamp(self.args.start_date))
            self.year = effective_year
            self.stock_list = load_constituents(effective_year)
        else:
            effective_year = self._effective_constituent_year(pd.Timestamp(datetime))
            if effective_year != self.year:
                self.year = effective_year
                self.stock_list = load_constituents(effective_year)

        return self.stock_list
    

    def new_universe(
        self,
        start_datetime : datetime,
        end_datetime : datetime,
        training : bool = True
    )   :
        """
            Create a new universe with the specified time range.

            par contre, dependamment si l'univers est pour entrainement ou pour le backtesting, on va devoir changer 
            ou on appelle la fonction get_stock_list
            si c'est pour l'entrainement, on va chercher la liste des stocks au moment end_datetime
            si c'est pour le backtesting, on va chercher la liste des stocks au moment start
        """
      
        
        if not isinstance(start_datetime, pd.Timestamp):
            start_datetime = pd.Timestamp(start_datetime)
        if not isinstance(end_datetime, pd.Timestamp):
            end_datetime = pd.Timestamp(end_datetime)
        
        # En training, on utilise la composition à end_datetime (date de décision d'investissement).
        # Les stocks entrés en cours de training sont ainsi inclus s'ils sont constituants
        # au moment de la construction du portefeuille. Les stocks ayant quitté l'indice
        # avant end_datetime sont absents du fichier constituant correspondant.
        # En test, on utilise start_datetime (= fin du training précédent, composition active
        # au moment de l'investissement — pas de look-ahead).
        ref_date = end_datetime if training else start_datetime
        self.update_stock_list(ref_date)
        print(f"ℹ️ Univers {'training' if training else 'test'} [{start_datetime.date()} → {end_datetime.date()}]: {len(self.stock_list)} constituant(s) chargés (ref: {ref_date.date()}).")

        # Filtrer les stocks absents des données de rendement
        valid_stocks = [stock for stock in self.stock_list if stock in self.df_return_all.columns]
        missing_stocks = set(self.stock_list) - set(valid_stocks)
     
        if missing_stocks:
            print(f"⚠️ Les actions suivantes ne sont pas dans les données de rendement : {missing_stocks}")
        self.stock_list = valid_stocks
        
        
        # On trie self.stock_list selon l'ordre des colonnes de df_return_all
        ordered_stocks = [stock for stock in self.df_return_all.columns if stock in self.stock_list]
        #retourne les stocks de l'univers au bonne periode de temps
        self.df_return = self.df_return_all.loc[start_datetime:end_datetime, ordered_stocks].copy()
        self.df_index = self.df_index_all.loc[start_datetime:end_datetime].copy()

        # Pour la reproduction SP500, conserver le comportement historique :
        # pas d'intersection de calendriers ni de nettoyage, on remplit les NaN
        # par des zéros et on conserve toutes les colonnes.
        if self.missing_policy == "legacy":
            filled_values = int(self.df_return.isna().sum().sum())
            self.df_return = self.df_return.fillna(0)
            self.df_index = self.df_index.fillna(0)
            self.stock_list = list(self.df_return.columns)
            self.last_cleaning_stats = {
                "initial_shape": self.df_return.shape,
                "calendar_dates_removed": [],
                "values_filled": filled_values,
                "dropped_columns": [],
                "dropped_rows": [],
                "final_shape": self.df_return.shape,
            }
            print(
                "Legacy missing-data policy: filled all NaNs with zero and kept all columns/rows.",
            )
            self.year = -1
            return

        # Aligne explicitement les calendriers pour éviter les NaN liés aux jours non communs
        common_index = self.df_return.index.intersection(self.df_index.index)
        dropped_calendar = (
            set(self.df_return.index) | set(self.df_index.index)
        ) - set(common_index)
        self.df_return = self.df_return.loc[common_index]
        self.df_index = self.df_index.loc[common_index]

        self.data_cleaning(
            target_cardinality=getattr(self.args, "cardinality", None),
            dropped_calendar_dates=sorted(dropped_calendar),
            training=training,
        )
        self.stock_list = list(self.df_return.columns)

        # Reset the constituent-year cache so the next call to update_stock_list()
        # always reloads from disk.  Without this, if two consecutive windows share
        # the same effective_year the second call would skip the reload and return
        # the already-cleaned (shorter) list from the previous window.
        self.year = -1

    

    
    def get_stocks_returns(self):
        return self.df_return
    
    def get_index_returns(self):
        # Toujours renvoyer une série (y compris pour une seule date) pour éviter
        # d'écraser la dimension temps lors de l'extraction du sous-ensemble.
        if isinstance(self.df_index, pd.DataFrame):
            return self.df_index.iloc[:, 0]
        return self.df_index
    
    def get_stock_namme_in_order(self):
        return self.df_return.columns
    
    def get_number_of_stocks(self):
        return len(self.stock_list)
    
    
    def data_cleaning(self, target_cardinality=None, dropped_calendar_dates=None, training=True):
        # Note: the "legacy" policy is handled in new_universe() before this method
        # is called, so this method only ever runs under the "strict" policy.
        stats = {
            "initial_shape": self.df_return.shape,
            "calendar_dates_removed": dropped_calendar_dates or [],
        }

        # Seuils configurables via args (avec valeurs par défaut).
        max_missing_frac = getattr(self.args, "max_missing_frac", 0.10)
        min_trading_frac = getattr(self.args, "min_trading_frac", 0.50)
        winsor_sigma = getattr(self.args, "winsor_sigma", 3.0)
        hard_clip = getattr(self.args, "hard_clip", 1.0)

        # --- Étape 1 : suppression des stocks trop lacunaires (training uniquement) ---
        # En mode test, ce filtre utiliserait les statistiques de TOUTE la fenêtre
        # de test → regard dans le futur. On conserve tous les stocks disponibles
        # et on laisse fillna(0) gérer les NaN.
        if training:
            missing_frac = self.df_return.isna().mean(axis=0)
            keep_cols = missing_frac <= max_missing_frac
            colonnes_supprimees = missing_frac[~keep_cols].index.tolist()
            self.df_return = self.df_return.loc[:, keep_cols]
            stats["dropped_columns"] = colonnes_supprimees
            print(
                f"Removed {len(colonnes_supprimees)} columns exceeding "
                f"{max_missing_frac:.0%} missing-data threshold."
            )
        else:
            stats["dropped_columns"] = []

        # --- Étape 2 : remplissage par 0 ---
        # Un jour sans rendement CRSP = pas de transaction = rendement nul.
        nan_count = self.df_return.isna().sum().sum()
        self.df_return = self.df_return.fillna(0)
        stats["values_filled"] = int(nan_count)
        print(f"Filled {nan_count} missing values with 0 (no-trade days).")

        # --- Étape 2b : clip absolu des rendements aberrants (training et test) ---
        # Les données CRSP peuvent contenir des artefacts extrêmes (reverse mergers,
        # erreurs d'ajustement de prix, penny stocks) qui faussent la matrice de
        # distances et la winsorisation σ-based. Un clip fixe ±hard_clip s'applique
        # aussi bien en training qu'en test : il ne dépend d'aucune statistique de
        # la fenêtre courante, donc n'introduit pas de regard dans le futur.
        if hard_clip > 0:
            clipped = (self.df_return.abs() > hard_clip).sum().sum()
            self.df_return = self.df_return.clip(lower=-hard_clip, upper=hard_clip)
            if clipped > 0:
                print(f"Hard-clipped {clipped} extreme return(s) to ±{hard_clip:.0%}.")

        # --- Étape 3 : filtre de liquidité (training uniquement) ---
        # Sur Russell 3000, les small/micro caps peu liquides génèrent beaucoup
        # de jours à rendement nul. Ces zéros artificiels compriment leur variance
        # et poussent l'optimiseur à les sélectionner à tort (bon tracking apparent
        # in-sample, mauvais out-of-sample). On exige un minimum de jours actifs.
        # En mode test, ce filtre utiliserait les stats de la fenêtre future → skippé.
        if training:
            nonzero_frac = (self.df_return != 0).mean(axis=0)
            liquid_cols = nonzero_frac >= min_trading_frac
            colonnes_illiquides = nonzero_frac[~liquid_cols].index.tolist()
            self.df_return = self.df_return.loc[:, liquid_cols]
            stats["dropped_illiquid"] = colonnes_illiquides
            print(
                f"Removed {len(colonnes_illiquides)} illiquid stocks "
                f"(< {min_trading_frac:.0%} non-zero trading days)."
            )
        else:
            stats["dropped_illiquid"] = []

        # --- Étape 4 : winsorisation des rendements extrêmes (training uniquement) ---
        # En évaluation out-of-sample, winsoriser avec les stats de la fenêtre
        # de test introduirait un regard dans le futur. Les rendements bruts
        # sont utilisés pour le backtest.
        if winsor_sigma > 0 and training:
            # Compute mean/std on actual trading days only (exclude fillna(0) zeros
            # which would artificially compress the bounds toward zero).
            trading_returns = self.df_return.replace(0, np.nan)
            mean = trading_returns.mean(axis=0)
            std = trading_returns.std(axis=0)
            lower = mean - winsor_sigma * std
            upper = mean + winsor_sigma * std
            self.df_return = self.df_return.clip(lower=lower, upper=upper, axis=1)
            print(f"Winsorized returns at ±{winsor_sigma}σ per stock (computed on trading days only).")
        elif winsor_sigma > 0 and not training:
            print("Winsorization skipped (out-of-sample evaluation — no future stats used).")

        # --- Étape 5 : suppression des lignes où l'indice est manquant ---
        lignes_avant = self.df_return.shape[0]
        index_valid = self.df_index.notna().all(axis=1)
        lignes_supprimees = self.df_return.index[~index_valid].tolist()
        self.df_return = self.df_return.loc[index_valid]
        self.df_index = self.df_index.loc[index_valid]
        stats["dropped_rows"] = lignes_supprimees
        lignes_apres = self.df_return.shape[0]
        print(f"Removed {lignes_avant - lignes_apres} rows due to missing index values.")

        # --- Étape 6 : vérification de cardinalité (training uniquement) ---
        # En test, on n'impose pas de cardinalité minimum — l'univers de test
        # peut contenir plus ou moins de stocks que K, _align_weights s'en charge.
        if training and target_cardinality is not None and self.df_return.shape[1] < target_cardinality:
            raise ValueError(
                "Cleaned universe cardinality below target after dropping incomplete columns: "
                f"{self.df_return.shape[1]} < {target_cardinality}. Consider reducing the cardinality "
                "or relaxing the missing-data policy (--max_missing_frac, --min_trading_frac)."
            )

        stats["final_shape"] = self.df_return.shape
        stats["calendar_count"] = int(self.df_return.shape[0])
        stats["calendar_hash"] = self._hash_calendar(self.df_return.index)
        self.last_cleaning_stats = stats

    @staticmethod
    def _hash_calendar(index: pd.Index) -> str:
        payload = "|".join(ts.isoformat() for ts in index)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    
    def _discover_constituent_files(self) -> tuple[Path, list[int]]:
        data_path = self.args.data_path
        candidate_dirs = [
            Path(f"{data_path}/{self.args.index}/constituants"),
            Path(f"{data_path}/{self.args.index}/constituants_raw"),
        ]
        for directory in candidate_dirs:
            if not directory.exists():
                continue

            years = sorted(
                int(path.stem)
                for path in directory.glob("*.csv")
                if path.stem.isdigit()
            )
            if years:
                return directory, years

        raise FileNotFoundError(
            "No constituent CSV files found under 'constituants' or 'constituants_raw'."
        )


    def _select_constituent_year(self, requested_year: int) -> int:
        if requested_year in self.constituent_years:
            return requested_year

        earliest = self.constituent_years[0]
        latest = self.constituent_years[-1]

        if requested_year < earliest:
            return earliest

        # Choose the most recent available year that does not exceed the request to avoid look-ahead
        prior_years = [y for y in self.constituent_years if y <= requested_year]
        if prior_years:
            return prior_years[-1]

        # If only future years exist (should not happen with above check), fall back to the earliest
        return latest

    def _get_constituents_for_date(self, dt: pd.Timestamp) -> list[str]:
        """Retourne la liste des constituants à une date donnée sans modifier l'état interne.

        Contrairement à update_stock_list(), cette méthode est sans effet de bord :
        elle ne modifie ni self.stock_list ni self.year.
        """
        effective_year = self._effective_constituent_year(dt)
        selected_year = self._select_constituent_year(effective_year)
        filepath = self.constituent_dir / f"{selected_year}.csv"
        df = pd.read_csv(filepath, dtype={"permno": str})["permno"]
        if selected_year != effective_year:
            print(
                f"⚠️ Constituents for {effective_year} unavailable; using {selected_year} "
                f"(for date {dt.date()})."
            )
        return df.tolist()
