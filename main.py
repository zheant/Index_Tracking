import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
from prafa.portfolio import Portfolio
from prafa.universe import Universe


def Main():
    # Set the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                    default='financial_data')

    parser.add_argument('--result_path', type=str, 
                    default='results') # default=os.getcwd()+'/results'
    parser.add_argument('--solution_name', type=str,
                    default='quob')#,] choices=['quob_cor', 'quob',  'gurobi', 'lagrange_backward'])

    parser.add_argument('--cardinality', type=int, default=30)

    # Select the Data to Use
    parser.add_argument('--start_date', type=str, default="2014-01-02")
    parser.add_argument('--end_date', type=str, default="2025-01-02")
    parser.add_argument('--index', type=str,
                    default='sp500')#, choice=['sp500', 'russel, nikkei])

    parser.add_argument(
        '--replicator_cores',
        type=int,
        default=1,
        help='Nombre de cœurs à allouer par contrôleur ReplicaTOR (puissance de 2).'
    )

    parser.add_argument('--replicator_bin', type=str, default=None,
                    help='Chemin vers le binaire ReplicaTOR (par défaut ~/or_tool/cmake-build/ReplicaTOR ou la variable REPLICATOR_BIN)')
    parser.add_argument('--replicator_time_limit', type=float, default=300.0,
                    help="Durée maximale (en secondes) accordée à ReplicaTOR pour chaque fenêtre de rebalancement")

    parser.add_argument(
        '--gurobi_time_limit',
        type=float,
        default=10800.0,
        help="Durée maximale (en secondes) accordée à Gurobi pour chaque fenêtre de rebalancement",
    )
    parser.add_argument(
        '--gurobi_log_dir',
        type=str,
        default='logs',
        help=(
            "Répertoire dans lequel stocker les journaux détaillés de Gurobi. "
            "Laissez vide pour désactiver la sauvegarde automatique."
        ),
    )


    #nombre de jours 
    parser.add_argument('--T', type=int, default=3, help="nombre d'année pour l'entrainement")
    parser.add_argument('--rebalancing', type=int, default=12, help="Month increment for rebalancing")
    parser.add_argument(
        '--filter_inactive',
        dest='filter_inactive',
        action='store_true',
        help="Supprime les titres inactifs ou constants avant l'optimisation (activé par défaut).",
    )
    parser.add_argument(
        '--keep_inactive',
        dest='filter_inactive',
        action='store_false',
        help="Conserve les titres inactifs ou constants dans l'univers (comportement précédent).",
    )
    parser.set_defaults(filter_inactive=True)
    args = parser.parse_args()
    
  

    
    #fenetre d'entrainement
    portfolio_duration = relativedelta(years=args.T)

    #pour le rebalancement
    time_increment = relativedelta(months=args.rebalancing)

    #initialisation des object necessaire pour extraire les portefeuilles dans le temps
    universe = Universe(args)
    portfolio = Portfolio(universe)

    data_start = universe.get_data_start_date()

    # With a rolling window of ``T`` years we need at least that amount of
    # history before the first rebalance can take place.  Otherwise the
    # training slice collapses to a single day which in turn removes every
    # security during the variance screening stage.
    earliest_rebalance = data_start + portfolio_duration

    start_date = max(pd.to_datetime(args.start_date), earliest_rebalance)
    end_date = pd.to_datetime(args.end_date)

    if start_date >= end_date:
        raise ValueError(
            "La période disponible est insuffisante pour un rebalance : "
            "ajustez --start_date/--end_date ou réduisez la fenêtre --T."
        )

    dates = [start_date]
    current_date = start_date + time_increment

    while current_date < end_date:
        dates.append(current_date)
        current_date += time_increment

    total_rebalances = len(dates)

    for idx, rebalancing_date in enumerate(dates, start=1):
        start_datetime = rebalancing_date - portfolio_duration
        if start_datetime < data_start:
            start_datetime = data_start
        print(
            f"🔁 Rebalancement {idx}/{total_rebalances} : "
            f"fenêtre {start_datetime.date()} → {rebalancing_date.date()}"
        )
        portfolio.rebalance_portfolio(start_datetime, rebalancing_date)


    return None


if __name__ == "__main__":
    Main()


