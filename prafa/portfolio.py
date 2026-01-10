import numpy as np
from prafa.universe import Universe
from prafa.quob import QUOB
from prafa.gurobi import Gurobi
from datetime import datetime
import time
import pandas as pd
import json
import pickle
from scipy.optimize import minimize


from datetime import datetime
from dateutil.relativedelta import relativedelta

"""
Cette classe centralise le code. Elle qui mets a jour l'univers et ensuite calcule L'optimisation et stack la reponse ici avec la date 
Faire bien attention de bien ajuster universe avant optimization parce l'universe est réutiliser dans L'optimisation , mais on la récupere 
a partir de portfolio
"""


class Portfolio:
    def __init__(self, universe: Universe):
        self.universe = universe
        
        self.portfolios = {}  # Dictionnaire pour stocker les portefeuilles par date (le portfeuille est un dictionnaire de poids)
    
    def rebalance_portfolio(self,
        start_datetime : datetime,
        end_datetime : datetime
    )   :
        #la fenetre de temps est celle de l'entrainement donc, on regarde composition de la end_date et se sert des
        #données passées pour résoudre le probleme d'optimisation et ainsi trouver les poids optimiaux
        self.universe.new_universe(start_datetime, end_datetime)
        sol = Solution(self)

        weights = sol.solve()
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        elif not isinstance(weights, pd.Series):
            weights = pd.Series(weights, index=self.universe.get_stock_namme_in_order())

        audit = getattr(self.universe, "last_cleaning_stats", {})
        self.portfolios[end_datetime] = {
            "weights": weights,
            "calendar_hash": audit.get("calendar_hash"),
            "calendar_count": audit.get("calendar_count"),
        }
        return self.portfolios[end_datetime] # dictionnire contenant poids
       

    def get_universe(self) -> Universe:
        return self.universe
    
    def save_portfolio(self):
        path = f'{self.universe.args.result_path}/portfolio_{self.universe.args.index}_{self.universe.args.solution_name}_{self.universe.args.cardinality}.json'
        with open(path, 'wb') as f:
            pickle.dump(self.portfolios, f)
            print('les portefeuilles ont été enregistrés!! pret a réalisé le backtest')
    
    
    def __del__(self):
        self.save_portfolio()
    



"""
La classe solution est la classe permettant de résoudre le problème d'optimisation.
"""

    
class Solution:
    
    def __init__(
        self,
        portfolio : Portfolio,
        ):
        
        self.portfolio = portfolio
        self.universe =  portfolio.get_universe()
        self.solution_name = self.universe.args.solution_name
        self.num_assets = self.universe.get_number_of_stocks()
        self.K = self.universe.args.cardinality
        self.distance_method = getattr(self.universe.args, "distance_method", "dcor")
        self.simple_corr = self.distance_method == "pearson"

        # Preserve explicit *_cor solution names while allowing CLI flag override.
        if self.solution_name.endswith("_cor"):
            self.simple_corr = True
        
        #self.new_return = np.array(self.universe.get_stocks_returns())
        #self.new_index = np.array(self.universe.get_index_returns())
        
        self.new_return = self.universe.get_stocks_returns().values
        self.new_index = self.universe.get_index_returns().values
        self.stock_list = self.universe.get_stock_namme_in_order()
        
        
        self.eps = 0.0001
        self.coefficient = 1000
        
        # print(self.new_index)
        # print(self.new_return)
        # raise Exception("Finish")
        # 3. Vérification manuelle de cohérence poids <-> action
        

    
    def objective_function(
        self,
        weight : list,
    ) -> list :
        # 누적 수익률
        error = self.new_return @ weight - self.new_index
        error = np.sum(error**2)
        
        return error# + 1000 * self.penalty(weight) #! ablation 
    
    
    def weight_sum_constraint(
        self,
        weight : list,
    ) -> list :
        # sorted_weights = sorted(weight)
        # return np.sum(sorted_weights[:self.K]) - 1
        
        return np.sum(weight) - 1
    
    
   

    def cardinality_constraint2(
        self,
        weight : list,
    ):
        
        # print("weight sum :", np.sum(weight))
        # print("weight :", weight)
        weight = 1 / ( 1 + np.e ** ( - self.coefficient * ( weight - self.eps) ) ) # before 1e-4
        # print("binary :", weight)
        # print("num of stocks :", np.sum(weight))
        # print("min num of stocks :", self.K)
        return - np.sum(weight) + self.K 
        
    

    def lagrange_full_replication(
        self
    ) -> dict :
        # Define initial weight
        initial_weight = np.ones(self.num_assets)
        initial_weight /= initial_weight.sum()  
        bounds = [(0, 1) for _ in range(self.num_assets)]

        # Define Constraints    
        constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}#, 'jac': self.weight_sum_jac}
        # constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}
        
        # Optimization
        result = minimize(self.objective_function, initial_weight, method = 'SLSQP', constraints=constraint, bounds=bounds)
        weights = pd.Series(result.x, index=self.new_return.columns)
        return weights 

    
    def lagrange_partial_ours(
        self
    ) -> dict :
        seed_value = 42
        np.random.seed(seed_value)
        
        trial = 1
        while(1):
            start_time = time.time()
            # Define initial weight
            initial_weight = np.random.rand(self.num_assets)
            initial_weight /= initial_weight.sum()  
            bounds = [(0, 1) for _ in range(self.num_assets)]

            # Define Constraints
            # constraint = [{'type': 'eq', 'fun': self.weight_sum_constraint},
            #               {'type': 'ineq', 'fun': self.cardinality_constraint}]
            #! ablation 
            constraint = [{'type': 'eq', 'fun': self.weight_sum_constraint},# 'jac': self.weight_sum_jac},
                        {'type': 'ineq', 'fun': self.cardinality_constraint2}]#, 'jac': self.cardinality_jac}]
            

            # Optimization
            result = minimize(self.objective_function, initial_weight, method = 'SLSQP', constraints=constraint, bounds=bounds, options={'maxiter': 200})#, tol=1e-6)
            self.optimal_weight = result.x
            self.stock2weight = {}
            for i in range(len(self.stock_list)):
                if result.x[i] < self.eps:
                    self.stock2weight[self.stock_list[i]] = 0
                else:
                    self.stock2weight[self.stock_list[i]] = result.x[i]
            """"
            # Update Portfolio & Calculate Error
            self.portfolio.update_portfolio(self.stock2weight)
            
            error = self.new_return @ self.optimal_weight - self.new_index
            error = np.sum(error**2)
            #! ablation 
            # self.optimal_error = self.objective_function(self.optimal_weight)
            self.optimal_error = error
            print(f"Calculated error : {self.optimal_error}")
            """
            # Calculate Top K weight sum
            topK_weight_sum = 0
            sorted_weights = sorted(self.stock2weight.items(), key=lambda x: x[1], reverse=True)
            for stock, weight in sorted_weights[:self.K]:
                topK_weight_sum += weight
            
            count = 0
            for stock, weight in sorted_weights:
                if (weight >= 0.0001):
                    count += 1
                
                
            # To avoid local optima
            if trial > 50:
                print("No portfolio satisfies the constraints")
                break
            #! ablation 
            if topK_weight_sum < 0.97 or topK_weight_sum > 1.01:
                print("topK weight sum", topK_weight_sum)
                trial += 1
                continue
            if (count > self.K):
                print("count:", count)
                trial += 1
                continue
            # # ! ablation 
            # if self.cardinality_constraint2(self.optimal_weight) < 0:
            #     w = 1 / ( 1 + np.e ** ( - self.coefficient * ( self.optimal_weight - self.eps) ) ) # before 1e-4
            #     print(f"calculated cardinality: {np.sum(w)}")
            #     print(f"K: {self.K}")
            #     print("Cardinality does not satisfied")
            #     break
            else:
                print("trial : ", trial)
                print(f'sec : {time.time() - start_time}')
                print(f'min : {(time.time() - start_time)/60}')
                # print(result)
                break
            raise Exception("")
        #return self.stock2weight #, self.optimal_error
        return np.array(list(self.stock2weight.values()))
    
    

    def quob(self):
        obj = QUOB(
            self.new_return,
            self.new_index,
            self.universe.args.cardinality,
            simple_corr=self.simple_corr,
            replicator_cores=self.universe.args.replicator_cores,
            time_limit=self.universe.args.time_limit,
        )
        return obj.get_weights()

    def quob_cor(self):
        obj = QUOB(
            self.new_return,
            self.new_index,
            self.universe.args.cardinality,
            simple_corr=True,
            replicator_cores=self.universe.args.replicator_cores,
            time_limit=self.universe.args.time_limit,
        )
        return obj.get_weights()

    def gurobi(self):
        obj = Gurobi(
            self.new_return,
            self.new_index,
            self.universe.args.cardinality,
            simple_corr=self.simple_corr,
            time_limit=self.universe.args.time_limit,
        )
        weights = obj.get_weights()
        if weights is None or obj.idx is None or len(obj.idx) == 0:
            status = getattr(obj, "status", None)
            runtime = getattr(obj, "runtime", None)
            selected = getattr(obj, "selection_cardinality", None)
            print(
                "Warning: Gurobi did not return any weights (status=%s, runtime=%s, "
                "selected=%s); using zero portfolio." % (status, runtime, selected)
            )
            return np.zeros(self.new_return.shape[1])

        full = np.zeros(self.new_return.shape[1])
        for idx, weight in zip(obj.idx, weights[obj.idx] if weights.shape[0] == self.new_return.shape[1] else weights):
            full[idx] = weight
        return full

    def gurobi_cor(self):
        obj = Gurobi(
            self.new_return,
            self.new_index,
            self.universe.args.cardinality,
            simple_corr=True,
            time_limit=self.universe.args.time_limit,
        )
        weights = obj.get_weights()
        if weights is None or obj.idx is None or len(obj.idx) == 0:
            status = getattr(obj, "status", None)
            runtime = getattr(obj, "runtime", None)
            selected = getattr(obj, "selection_cardinality", None)
            print(
                "Warning: Gurobi (correlation) did not return any weights (status=%s, runtime=%s, "
                "selected=%s); using zero portfolio." % (status, runtime, selected)
            )
            return np.zeros(self.new_return.shape[1])

        full = np.zeros(self.new_return.shape[1])
        for idx, weight in zip(obj.idx, weights[obj.idx] if weights.shape[0] == self.new_return.shape[1] else weights):
            full[idx] = weight
        return full

    def lagrange_partial_forward(
        self
    ) -> dict :
        num_assets = self.num_assets
        new_return = self.new_return
        new_index = self.new_index
        largest_weight = []
        largest_stocks = []
        stock_list = self.stock_list
        K = self.K
        
        while len(largest_weight) < K :  
            # Define initial weight 
            initial_weight = np.ones(num_assets)
            initial_weight /= initial_weight.sum()  
            bounds = [(0, 1) for _ in range(num_assets)]
            
            # Define Objective & Constratins 
            objective = lambda weight: np.sum((new_return @ weight - new_index)**2)
            constraint = {'type': 'eq', 'fun': self.weight_sum_constraint} #'jac': self.weight_sum_jac}
            # constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}
        
            # Optimization
            result = minimize(objective, initial_weight, method = 'SLSQP', constraints=constraint, bounds=bounds)
            
            # Find Largest Weight
            max_idx = np.argmax(result.x)
            max_weight = result.x[max_idx]
            max_weight_stock = stock_list[max_idx]
            largest_weight.append(max_weight)
            largest_stocks.append(max_weight_stock)
            print("largest weight:", max_weight)
            print("largest weight stock:", max_weight_stock)
            
            # Remove Largest Weight
            new_return = np.delete(new_return, max_idx, axis=1)
            stock_list = np.delete(stock_list, max_idx)
            num_assets -= 1
            
        # Finally QP with K stocks
        initial_weight = np.ones(K)
        initial_weight /= initial_weight.sum()  
        bounds = [(0, 1) for _ in range(K)]
        # Define Largest Return data
        df_new_return = self.universe.df_return[largest_stocks]
        new_return = np.array(df_new_return)
        # Define Objective & Constratins & Problem
        objective = lambda weight: np.sum((new_return @ weight - new_index)**2)
        constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}#, 'jac': self.weight_sum_jac}
        # Optimization
        result = minimize(objective, initial_weight, method = 'SLSQP', constraints=constraint, bounds=bounds)
        self.optimal_weight = result.x
        self.stock2weight = {}
        for i in range(len(largest_stocks)):
            self.stock2weight[largest_stocks[i]] = result.x[i]
        for i in range(len(stock_list)):
            if stock_list[i] not in self.stock2weight:
                self.stock2weight[stock_list[i]] = 0
        
        """
        # Update Portfolio & Calculate Error
        self.portfolio.update_portfolio(self.stock2weight)
        self.optimal_error = sum((new_return @ self.optimal_weight - new_index)**2)
        print(f"Calculated error : {self.optimal_error}")
        """
        return self.stock2weight #, self.optimal_error
    
    
    def lagrange_partial_backward(
        self,
    ) -> dict :
        num_assets = self.num_assets
        new_return = self.new_return
        new_index = self.new_index
        smallest_weight = []
        smallest_stocks = []
        stock_list = self.stock_list
        K = self.K
        
        new_stock_list = stock_list
        while num_assets >= K :  
            # Define initial weight 
            initial_weight = np.ones(num_assets)
            initial_weight /= initial_weight.sum()  
            bounds = [(0, 1) for _ in range(num_assets)]
            
            # Define Objective & Constratins 
            objective = lambda weight: np.sum((new_return @ weight - new_index)**2)
            constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}#, 'jac': self.weight_sum_jac}
            # constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}
        
            # Optimization
            result = minimize(objective, initial_weight, method = 'SLSQP', constraints=constraint, bounds=bounds)
            
            if num_assets != K:
                # Find Smallest Weight
                min_idx = np.argmin(result.x)
                min_weight = result.x[min_idx]
                min_weight_stock = new_stock_list[min_idx]
                smallest_weight.append(min_weight)
                smallest_stocks.append(min_weight_stock)
                print("smallest weight:", min_weight)
                print("smallest weight stock:", min_weight_stock)
                
                # Remove Smallest Weight
                new_return = np.delete(new_return, min_idx, axis=1)
                new_stock_list = np.delete(new_stock_list, min_idx)
                num_assets -= 1
            else:
                break
        
        self.optimal_weight = result.x
        self.stock2weight = {}
        for i in range(len(new_stock_list)):
            self.stock2weight[new_stock_list[i]] = self.optimal_weight[i]
        for i in range(len(stock_list)):
            if stock_list[i] not in self.stock2weight:
                self.stock2weight[stock_list[i]] = 0
        
        """
        # Update Portfolio & Calculate Error
        self.portfolio.update_portfolio(self.stock2weight)
        self.optimal_error = sum((new_return @ self.optimal_weight - new_index)**2)
        print(f"Calculated error : {self.optimal_error}")
        """
        return self.stock2weight #, self.optimal_error
        
        
    
  
    def solve(
        self,
    ) -> dict :
        solution_name = self.solution_name

        if solution_name == 'lagrange_full':
            weights = self.lagrange_full_replication()
        elif solution_name == 'lagrange_ours':
            weights = self.lagrange_partial_ours()
        elif solution_name == 'quob':
            weights = self.quob()
        elif solution_name == 'quob_cor':
            weights = self.quob_cor()
        elif solution_name == 'gurobi':
            weights = self.gurobi()
        elif solution_name == 'gurobi_cor':
            weights = self.gurobi_cor()
        elif solution_name == 'lagrange_forward':
            weights = self.lagrange_partial_forward()
        elif solution_name == 'lagrange_backward':
            weights = self.lagrange_partial_backward()
        
        return weights
        
