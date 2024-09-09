import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    HoltWinters,
    CrostonClassic as Croston, 
    HistoricAverage,
    DynamicOptimizedTheta,
    SeasonalNaive,
    AutoARIMA,
    AutoETS,
    AutoTBATS,
     MSTL
)

from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from mlforecast.utils import PredictionIntervals
from window_ops.expanding import expanding_mean
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression


def create_partitions(input, index, partitions, overlap, test_length, train_length = None):
    
    df = None

    for i in range(partitions, 0, -1):
        if train_length is None:
            s = 1
        else:
            s = len(input) - train_length - i * test_length + overlap * (i -1) - 1

        e = len(input) - i * test_length  + overlap * (i -1) - 1
        train_start = input[index].iloc[s]
        train_end = input[index].iloc[e]
        test_start = input[index].iloc[e + 1]
        test_end = input[index].iloc[e + test_length]
        
        p_df = {"partition": partitions - i + 1,
                "train_start": [train_start], 
                "train_end" : [train_end],
                "test_start": [test_start], 
                "test_end" : [test_end],
                }
        if df is None:
            df = pd.DataFrame(p_df)
        else:
            temp = pd.DataFrame(p_df)
            df = df._append(temp)

    df = df.sort_values(by = ["partition"])
    df.reset_index(inplace = True, drop = True)
    return df


def method_mapping(methods, partitions):
    d = None
    for m in methods["methods"].keys():
        models_list = methods["methods"][m]["models"]
        for md in models_list:
            temp = None
            temp = pd.DataFrame({"partition": range(1, partitions + 1, 1)})
            temp["label"] = m
            temp["method"] = methods["methods"][m]["method"]
            temp["model"] = md
            if d is None:
                d = temp
            else:
                d = d._append(temp)
    d = d.sort_values(by = ["partition"])
    d.reset_index(inplace = True, drop = True)

    return d    


def fc_to_long(forecast, models):
    f = None
    for i in models:
        temp = forecast[["unique_id", "ds", i, i + "-lo-0.95", i + "-hi-0.95"]]
        temp = temp.rename(columns = {i: "mean", i + "-lo-0.95": "lower", i + "-hi-0.95": "upper" })
        temp["model"] = i
        if f is None:
            f = temp
        else:
            f = f._append(temp)

    f.reset_index(inplace = True, drop = True)

    return f
    
    



class create_backtesting:
    def __init__(self):
        self.object = "backtesting"

    def add_input(self, input, index, var):
        self.input = input
        self.index = index
        self.var = var

    def add_bkt_params(self, partitions, overlap, test_length, train_length = None):
        self.partitions = partitions
        self.overlap = overlap
        self.test_length = test_length
        self.train_length = train_length
    
    def add_mlflow_settings(self, experiment_name, mlflow_path, tags, overwrite = False):
        self.experiment_name = experiment_name
        self.mlflow_path = mlflow_path
        self.tags = tags
        self.overwrite = overwrite

    def add_methods(self, methods):
        self.methods = methods

    def run_backtesting(self):
        self.grid = create_partitions(input = self.input, 
                                index = self.index,
                                partitions = self.partitions, 
                                overlap = self.overlap,
                                train_length = self.train_length,
                                test_length = self.test_length)
        methods_grid = method_mapping(methods = self.methods,
                                      partitions = self.partitions)
        d = self.input
        d_index = self.index
        bkt_test = None
        for index, row in self.grid.iterrows():
            p = row["partition"]
            train_start = row["train_start"]
            train_end = row["train_end"]

            test_start = row["test_start"]
            test_end = row["test_end"]

            train = d[(d[d_index] >= train_start) & (d[d_index] <= train_end)]
            test = d[(d[d_index] >= test_start) & (d[d_index] <= test_end)]
            h = len(test)
            freq = self.methods["freq"]
            pi = self.methods["pi"]
            cores = self.methods["cores"]
            methods = self.methods["methods"]
            prediction_intervals = self.methods["prediction_intervals"]

            if prediction_intervals["method"] == "conformal_distribution":
                n_windows = prediction_intervals["n_windows"]

            for m in methods:
                method = methods[m]["method"]
                if method == "MLForecast":

                    fc = create_mlforecast(input = train, 
                                      method = methods[m], 
                                      freq = freq, 
                                      cores = cores, 
                                      h = h, 
                                      pi = pi, 
                                      prediction_intervals = prediction_intervals)
                    
                    fc["partition"] = p 
                    fc["method"] = method
                    fc["label"] = m

                    if bkt_test is None:
                        bkt_test = fc
                    else:
                        bkt_test = bkt_test._append(fc)


                elif method == "StatsForecast":
                    print("a")

        self.backtesting = bkt_test
                
        

         
def create_mlforecast(input, method, freq, cores, h, pi, prediction_intervals):
    models_list = method["models"]
    models = [] 

    for md in models_list:
        models.append(eval(f"{md}()"))
    

    if "lags" not in method.keys() and "date_features" not in method.keys():
        print("Error: Neither the lags nor date_features arguments were specify")
        return
    if "lags" not in method.keys():
        lags = None
    else:
        lags = method["lags"]
    if "date_features" not in method.keys():
        date_features = None
    else:
        date_features = method["date_features"]
    mlf = MLForecast(models = models,
                     freq = freq,
                     lags = lags,
                     date_features = date_features,
                     num_threads= cores)
    if prediction_intervals["method"] == "conformal_distribution":
        mlf.fit(df = input,  fitted=True, 
                prediction_intervals=PredictionIntervals(n_windows = prediction_intervals["n_windows"], 
                                                         h = h, 
                                                         method="conformal_distribution"))
    fc = mlf.predict(h, level  = [pi])
    f = fc_to_long(forecast = fc, models = models_list)
    return f

def create_statsforecast(input, method, freq, cores, h, pi, prediction_intervals):
    models_list = method["models"]
    models = [] 

    for md in models_list:
        models.append(eval(f"{md}()"))
    

    if "lags" not in method.keys() and "date_features" not in method.keys():
        print("Error: Neither the lags nor date_features arguments were specify")
        return
    if "lags" not in method.keys():
        lags = None
    else:
        lags = method["lags"]
    if "date_features" not in method.keys():
        date_features = None
    else:
        date_features = method["date_features"]
    mlf = MLForecast(models = models,
                     freq = freq,
                     lags = lags,
                     date_features = date_features,
                     num_threads= cores)
    if prediction_intervals["method"] == "conformal_distribution":
        mlf.fit(df = input,  fitted=True, 
                prediction_intervals=PredictionIntervals(n_windows = prediction_intervals["n_windows"], 
                                                         h = h, 
                                                         method="conformal_distribution"))
    fc = mlf.predict(h, level  = [pi])
    f = fc_to_long(forecast = fc, models = models_list)
    return f
    



bkt = create_backtesting()
methods = {
    "prediction_intervals": {
        "method": "conformal_distribution",
        "n_windows": 5
    },
    "freq": "h",
    "pi": 0.95,
    "cores": 4,
    "methods": {
    "model1": {
        "method": "MLForecast",
        "models": ["LGBMRegressor", "XGBRegressor", "LinearRegression"],
        "lags": [1,2,3, 24, 48, 168],
        "date_features": ["month", "day", "dayofweek", "hour"]
       
    },
        "model2": {
        "method": "StatsForecast",
        "models": {"HoltWinters", 
                   "Croston", 
                   "SeasonalNaive": {"season_length": [24, 168]}, 
                   "HistoricAverage"}


        }
       
    }
    }
import datetime
data = pd.read_csv("data/us48.csv")
data["ds"] = pd.to_datetime(data["period"])
end = data["ds"].max().floor(freq = "d") - datetime.timedelta(hours = 1)
start = end - datetime.timedelta(hours = 24 * 30 * 25)
data = data[(data["ds"] <= end) & (data["ds"] >= start)]
data = data.sort_values(by = "ds")
data = data[["ds", "value"]]
data = data.rename(columns = {"ds": "ds", "value": "y"})
data["unique_id"] = 1


bkt.add_input(input = data, 
              var = "y", 
              index = "ds")

bkt.add_bkt_params(partitions = 2,
                   overlap = 0, 
                   test_length = 24, 
                   train_length = None)

bkt.add_mlflow_settings(experiment_name = "us48", 
                        mlflow_path = "file:///mlruns", 
                        tags = "v0.0.1", 
                        overwrite = True)
bkt.add_methods(methods = methods)
bkt.run_backtesting()