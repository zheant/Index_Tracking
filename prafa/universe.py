import pandas as pd
from datetime import datetime
import numpy as np





class Universe():
    
    
    def __init__(
        self,
        args,
    ) :
        self.args = args
        
        #données sur toutes l'historique
        self.initialisation_donnes()

        #timeseries sur la periode en cours
        self.df_return = None
        self.df_index = None

        self.stock_list = self.update_stock_list(None)
    

    def initialisation_donnes(self):
        #données sur toutes l'historique
        self.df_return_all = pd.read_csv(f"financial_data/{self.args.index}/returns_stocks.csv")  #return des stocks 
        #self.df_return_all.columns = [col.split()[0].replace('/', '.') for col in self.df_return_all.columns]
        self.df_return_all['date'] = pd.to_datetime(self.df_return_all['date'])
        self.df_return_all.set_index('date', inplace=True)

        self.df_index_all = pd.read_csv(f"financial_data/{self.args.index}/returns_index.csv")   #return de l'indice
        self.df_index_all['Date'] = pd.to_datetime(self.df_index_all['Date'])
        self.df_index_all.set_index('Date', inplace=True)

    
    def update_stock_list(self, datetime : datetime = None):
        #ce code va aller chercher la compositon
        #df = lambda year : pd.read_csv(f"financial_data/{self.args.index}/constituants/{year}.csv",usecols=["Ticker"]).str.split().str[0].str.replace("/", ".")
        df = lambda year: pd.read_csv(
                f"financial_data/{self.args.index}/constituants/{year}.csv", dtype={'permno': str})["permno"]
        
        if datetime is None:
            #appelle dans le constructeur premier universe
            self.year = int(self.args.start_date[0:4])
            self.stock_list = df(self.year).tolist()

        elif datetime.year != self.year:
            #puisque le rebalancement se fait par an,
            #on va chercher la liste des stocks pour l'année en cours
            #sinon on va chercher les stocks pour la nouvelle année
            self.year = datetime.year
            self.stock_list = df(self.year).tolist()

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
      
        
        if type(start_datetime) != type(pd.Timestamp('now')):
            start_datetime = pd.Timestamp(start_datetime)
        if type(end_datetime) != type(pd.Timestamp('now')):
            end_datetime = pd.Timestamp(end_datetime)
        
        #ajustement des stocks dans l'univers
        if training:
            self.update_stock_list(end_datetime)
        else:
            self.update_stock_list(start_datetime)
        
        # ⚠️ À mettre dans la méthode new_universe juste avant d'extraire les rendements :
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

        # Aligne explicitement les calendriers pour éviter les NaN liés aux jours non communs
        common_index = self.df_return.index.intersection(self.df_index.index)
        self.df_return = self.df_return.loc[common_index]
        self.df_index = self.df_index.loc[common_index]

        self.data_cleaning()
        self.stock_list = list(self.df_return.columns)

    

    
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
    
    
    def data_cleaning(self):
        # Nombre de NaN avant tout traitement
        nan_avant = self.df_return.isna().sum().sum()

        # Remplissage avant/arrière avec fenêtre limitée pour combler les petits trous
        self.df_return.ffill(limit=5, inplace=True)
        self.df_return.bfill(limit=5, inplace=True)

        nan_apres_remplissage = self.df_return.isna().sum().sum()
        valeurs_remplies = nan_avant - nan_apres_remplissage
        print(f"Filled {valeurs_remplies} missing values with limited ffill/bfill.")

        # Restreindre l'univers aux titres ayant des données sur toute la fenêtre
        colonnes_avant = self.df_return.shape[1]
        self.df_return = self.df_return.dropna(axis=1, how="any")
        self.stock_list = self.df_return.columns.to_list()
        colonnes_apres = self.df_return.shape[1]
        print(
            f"Removed {colonnes_avant - colonnes_apres} columns lacking full window coverage."
        )

        # Supprimer les lignes avec au moins un NaN restant (synchronisation avec l'index)
        lignes_avant = self.df_return.shape[0]

        self.df_index.dropna(inplace=True)
        common_index = self.df_return.index.intersection(self.df_index.index)
        self.df_return = self.df_return.loc[common_index]
        self.df_index = self.df_index.loc[common_index]

        self.df_return.dropna(inplace=True)
        self.df_index = self.df_index.loc[self.df_return.index]

        lignes_apres = self.df_return.shape[0]
        print(f"Removed {lignes_avant - lignes_apres} rows due to missing values.")
