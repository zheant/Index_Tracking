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

    parser.add_argument('--replicator_bin', type=str, default=None,
                    help='Chemin vers le binaire ReplicaTOR (par défaut ~/or_tool/cmake-build/ReplicaTOR ou la variable REPLICATOR_BIN)')
    parser.add_argument('--replicator_time_limit', type=float, default=300.0,
                    help="Durée maximale (en secondes) accordée à ReplicaTOR pour chaque fenêtre de rebalancement")


    #nombre de jours 
    parser.add_argument('--T', type=int, default=3, help="nombre d'année pour l'entrainement")
    parser.add_argument('--rebalancing', type=int, default=12, help="Month increment for rebalancing")
    args = parser.parse_args()
    
  

    
    #fenetre d'entrainement
    portfolio_duration = relativedelta(years=args.T)

    #pour le rebalancement
    time_increment = relativedelta(months=args.rebalancing)

    #liste des dates de rebalancement
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    
    # Construire la liste des dates
    dates = [start_date]
    current_date = start_date + time_increment

    while current_date < end_date:
        dates.append(current_date)
        current_date += time_increment
    
    #initialisation des object necessaire pour extraire les portefeuilles dans le temps
    portfolio = Portfolio(Universe(args))
    for rebalancing_date in dates:
        start_datetime = rebalancing_date - portfolio_duration
        portfolio.rebalance_portfolio(start_datetime, rebalancing_date)
        print(f"Rebalancing from {start_datetime.date()} to {rebalancing_date.date()}")

    
    return None


if __name__ == "__main__":
    Main()


