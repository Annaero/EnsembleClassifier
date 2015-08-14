import sys
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from EnsembleClassifier import EnsembleClassifier, OMEnsembleClassifier, dtw_measurement as dtw
from regres import read_data, read_ens_coeffs
from SelectionStrategy import NoneStrategy     
from statistics import median, mean  
from math import sqrt  
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
from dateutil import parser
from dtw import dtw
from dateutil.rrule import rrule, HOURLY
from sklearn import linear_model

from itertools import product
from numpy.linalg import lstsq
       
def rmse(actual, predicted):
    ln = min(len(predicted), len(actual))
    rmse = sqrt(mean_squared_error(actual[:ln], predicted[:ln]))
    return rmse   
    
def calibrate_ensemble(models, measurements, max_tm=None):
    modelsPredictions = models  
      
    predictors = [list() for mdl in range(3)]
    target = list()
    #predictors.append([1] * len(predictors[0]))
    rng = max_tm if max_tm else len(models[0])
    for tm in range(rng):
        msm = measurements[tm*6: tm*6 + 48]
        for currentPrediction, predictor \
                in zip([prd[tm] for prd in modelsPredictions], predictors):
#            print(len(currentPrediction[:48]))
            predictor.extend(currentPrediction[:48])
#        print(len(msm`))
        target.extend(msm)
        
#    print(*MODELS)
        for ens_map in reversed(list(product([1,0], repeat = 3))):
#            lm = linear_model.LinearRegression()
            
            ensemble_predictors = \
                    [[a*b for a,b in zip(point, ens_map)] for point in zip(*predictors)]
#            lm.fit(ensemble_predictors, target)
#            lm.get_params() 
        
            ensemble_predictors = [pred+[1] for pred in ensemble_predictors]
            coefs = lstsq(ensemble_predictors, target)[0]
            yield coefs
    
if __name__ == "__main__":
    path = sys.argv[1]
    path2 = sys.argv[2]

    MODEL = "S1"

    measurementsFile = os.path.join(path, "2011080100_measurements_{}_2623.txt".format(MODEL))
    noswanFile = os.path.join(path, "2011080100_noswan_{}_48x434.txt".format(MODEL))
    swanFile = os.path.join(path, "2011080100_swan_{}_48x434.txt".format(MODEL))
    hirombFile = os.path.join(path, "2011080100_hiromb_{}_60x434.txt".format(MODEL))
    coeffsFile = os.path.join(path, "ens_coefs.txt")
    
    measurements = read_data(measurementsFile)
    noswan = read_data(noswanFile)
    swan = read_data(swanFile)
    hiromb = read_data(hirombFile)

    modelsPredictions = [hiromb, swan, noswan]    
      
    predictors = [list() for mdl in range(3)]
    target = list()
    #predictors.append([1] * len(predictors[0]))
    for tm in range(100):#range(int(len(measurements)/6)-8):
        msm = measurements[tm*6 : tm*6 + 48]
        for currentPrediction, predictor \
                in zip([prd[tm] for prd in modelsPredictions], predictors):
#            print(len(currentPrediction[:48]))
            predictor.extend(currentPrediction[:48])
#        print(len(msm))
        target.extend(msm)
        
#    print(*MODELS)
    with open(os.path.join(path2, "ens_coefs.txt"), "w+") as ens_coef_file:
        for ens_map in reversed(list(product([1,0], repeat = 3))):
#            lm = linear_model.LinearRegression()
            
            ensemble_predictors = \
                    [[a*b for a,b in zip(point, ens_map)] for point in zip(*predictors)]   
#            lm.fit(ensemble_predictors, target)
#            lm.get_params() 
        
            ensemble_predictors = [pred+[1] for pred in ensemble_predictors]
            coefs = lstsq(ensemble_predictors, target)[0]
            form_str = "\t".join(["{{{0}:.3f}}".format(i) for i in range(4)])
            coef_str = form_str.format(*coefs)
            ens_coef_file.write(coef_str+"\n")
            print(coef_str)
            
#            print(lstsq(ensemble_predictors, target)[0])