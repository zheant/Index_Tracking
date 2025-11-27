
import gurobipy as gp
from gurobipy import GRB
import dcor
import numpy as np
import pandas as pd
from scipy.optimize import minimize


params = {
    "WLSACCESSID": "ee1b9da6-6290-460c-aa8c-d8071e9ddaf0",
    "WLSSECRET": "7e4a8961-92b1-4b35-bc63-094505d6bb3f",
    "LICENSEID": 2677740,
}

#env = gp.Env(params=params) 


class Gurobi:
    def __init__(self, stocks_returns, index_returns, K, simple_corr=False, time_limit=300):
        #matrice et vecteur numpy
        self.stocks_returns = stocks_returns
        self.index_returns = index_returns
        self.K = K #cardinalité!!
        self.idx = None #liste d'indice des stonks choisit
        self.simple_corr = simple_corr
        self.time_limit = time_limit
        
        


    def matrix_dcor(self):

        Welsch_function = lambda x : 1 - np.exp(-0.5 * x)

        n = self.stocks_returns.shape[1]
        dcor_mat = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                dcor_val = dcor.distance_correlation(self.stocks_returns[:, i], self.stocks_returns[:, j])
                dist = 1 - dcor_val
                dcor_mat[i, j] = dcor_mat[j, i] = Welsch_function(dist) #Welsch_function(dist)

        dcor_mat = np.nan_to_num(dcor_mat, nan=1.0, posinf=1.0, neginf=1.0)
        np.fill_diagonal(dcor_mat, 0.0)
        dcor_mat = np.clip(dcor_mat, 0.0, 1.0)

        return dcor_mat
        


    def matrix_simplecor(self):
        Welsch_function = lambda x : 1 - np.exp(-0.5 * x)
        distance_func = lambda di : np.sqrt(0.5*(1 - di))

        n = self.stocks_returns.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_matrix = np.corrcoef(self.stocks_returns, rowvar=False)

        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        base_distance = 0.5 * (1 - corr_matrix)
        base_distance = np.maximum(base_distance, 0.0)
        distance_matrix = distance_func(base_distance)
        distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=1.0)

        welsch_distance = np.nan_to_num(Welsch_function(distance_matrix), nan=1.0, posinf=1.0, neginf=1.0)
        np.fill_diagonal(welsch_distance, 0.0)
        welsch_distance = np.clip(welsch_distance, 0.0, 1.0)

        return welsch_distance


    def stock_picking(self, n):
        #résolution du probleme d'optimisation 
        #retourne une liste d'indice des stonks sélectionné
        #construire ma matrice de distance
        if self.simple_corr:
            D = self.matrix_simplecor()
        else:
            D = self.matrix_dcor()

        n = D.shape[0]
        alpha = 1 / self.K
        beta = 1 / n

        ones = np.ones(n)
        c = beta * (D @ ones)   # c = β Δ 1

        #m = gp.Model("BQO_compact")
        #m.Params.NonConvex = 2
        with gp.Env(params=params) as env, gp.Model(env=env) as m:
            m.setParam("TimeLimit", self.time_limit)

            z = m.addMVar(n, vtype=GRB.BINARY, name="z")

            # Objectif entièrement matriciel
            m.setObjective(c @ z - 0.5 * alpha * (z @ D @ z), GRB.MINIMIZE)

            # Contrainte de cardinalité
            m.addConstr(ones @ z == self.K, name="card")

            m.optimize()
            return z.X


    def calc_weights(self):
        stock_pick_binary = self.stock_picking(self.stocks_returns.shape[1])
        self.idx = np.where(stock_pick_binary == 1)[0].tolist()
        subset_returns = self.stocks_returns[:, self.idx]
        
        initial_weight = np.ones(len(self.idx))
        initial_weight /= initial_weight.sum()  
        bounds = [(0, 1) for _ in range(len(self.idx))]

        # Define Constraints    
        constraint = {'type': 'eq', 'fun':lambda weight : np.sum(weight) - 1}
        objective_function = lambda weight : np.sum((subset_returns @ weight - self.index_returns)**2)
        
        # Optimization
        result = minimize(objective_function, initial_weight, method = 'SLSQP', constraints=constraint, bounds=bounds)
        return result.x

    
    def get_weights(self):
        #retourne numpy array sparse des poids
        
        weight_global = np.zeros(self.stocks_returns.shape[1])

        micro_weight = self.calc_weights()
        for i in range(len(micro_weight)):
            weight_global[self.idx[i]] = micro_weight[i]

        return weight_global
