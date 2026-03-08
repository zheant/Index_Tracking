import argparse
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
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

    parser.add_argument('--cardinality', type=int, default=50)

    parser.add_argument('--replicator_cores', type=int, default=int(os.environ.get("REPLICATOR_CORES", 8)),
                    help='Number of OpenMP threads for ReplicaTOR (overrides REPLICATOR_CORES env)')

    parser.add_argument('--time_limit', type=float, default=300.0,
                    help='Time limit (seconds) applied to QUOB/ReplicaTOR and Gurobi solves')

    parser.add_argument('--d_scale', type=float, default=1.0,
                    help='D_scale_factor for ReplicaTOR: scales dispersion between selected medoids (default 1.0)')

    parser.add_argument('--strata_large_size', type=int, default=1000,
                    help='Number of top stocks (by market cap) in the large-cap stratum for stratified QUOB '
                         '(default 1000: Russell 1000/2000 institutional boundary)')

    parser.add_argument('--distance_method', type=str, choices=['dcor', 'pearson'], default='dcor',
                    help='Distance metric for solver matrices (dcor or pearson)')

    parser.add_argument('--missing_policy', type=str, choices=['auto', 'strict', 'legacy'], default='auto',
                    help='Missing-data handling: auto uses legacy for SP500 and strict otherwise; override to force a policy')

    parser.add_argument('--reconstitution_month', type=int, default=7,
                    help='Premier mois où la nouvelle composition de l\'indice est active (défaut 7 = juillet pour Russell 3000)')

    parser.add_argument('--max_missing_frac', type=float, default=0.10,
                    help='Fraction maximale de jours manquants pour conserver un stock (défaut 0.10)')
    parser.add_argument('--min_trading_frac', type=float, default=0.50,
                    help='Fraction minimale de jours à rendement non nul pour conserver un stock — filtre de liquidité (défaut 0.50)')
    parser.add_argument('--winsor_sigma', type=float, default=3.0,
                    help='Seuil de winsorisation en nombre de σ par titre (0 pour désactiver, défaut 3.0)')
    parser.add_argument('--hard_clip', type=float, default=1.0,
                    help='Clip absolu des rendements aberrants avant winsorisation, ex. 1.0 = ±100%% par jour (0 pour désactiver, défaut 1.0)')
    parser.add_argument('--exclude_pool_b_capweight', action='store_true', default=False,
                    help='Phase 16 : exclure Pool B du cap-weighting (Pool A uniquement, renormalisé). '
                         'Élimine le biais de liquidité au coût d\'ignorer ~15-20%% du poids de l\'indice.')
    parser.add_argument('--no_stratification', action='store_true', default=False,
                    help='Désactiver la stratification dans quob_stratified : un seul QUOB sur l\'ensemble de Pool A '
                         'suivi d\'un cap-weighting global. Combiné avec --min_trading_frac 0.0 → Phase 17 sans strates.')
    parser.add_argument('--phase18_qp_index', action='store_true', default=False,
                    help='Phase 18 : remplacer le cap-weighting par un QP ciblant r_index directement. '
                         'Minimise ||R_medoids @ w - r_index||² s.t. Σw=1, w≥0. '
                         'Pool B non détenu mais son influence passe par la cible index.')

    # Select the Data to Use
    parser.add_argument('--start_date', type=str, default="2014-01-02")
    parser.add_argument('--end_date', type=str, default="2025-01-02")
    parser.add_argument('--index', type=str,
                    default='sp500')#, choice=['sp500', 'russel, nikkei])


    #nombre de jours 
    parser.add_argument('--T', type=int, default=3, help="nombre d'année pour l'entrainement")
    parser.add_argument('--rebalancing', type=int, default=12, help="Month increment for rebalancing")
    args = parser.parse_args()

    if args.missing_policy == "auto":
        args.missing_policy = "legacy" if args.index.lower() == "sp500" else "strict"
    
  

    
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

    # S'assurer que end_date est incluse comme dernière fenêtre d'entraînement
    if dates[-1] != end_date:
        dates.append(end_date)

    #initialisation des object necessaire pour extraire les portefeuilles dans le temps
    portfolio = Portfolio(Universe(args))
    for rebalancing_date in dates:
        start_datetime = rebalancing_date - portfolio_duration
        portfolio.rebalance_portfolio(start_datetime, rebalancing_date)
        print(f"Rebalancing from {start_datetime.date()} to {rebalancing_date.date()}")

    portfolio.save_portfolio()
    return None


if __name__ == "__main__":
    Main()


