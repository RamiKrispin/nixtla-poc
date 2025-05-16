from statistics import mean

def models_reformat(models):
  for i in range(len(models)):
    if isinstance(models[i], str):
      models[i] = eval(models[i])


def mape(y, yhat):
    mape = mean(abs(y - yhat)/ y) 
    return mape

def rmse(y, yhat):
    rmse = (mean((y - yhat) ** 2 )) ** 0.5
    return rmse

def coverage(y, lower, upper):
    coverage = sum((y <= upper) & (y >= lower)) / len(y)
    return coverage