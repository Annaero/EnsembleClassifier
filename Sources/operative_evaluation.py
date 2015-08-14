# -*- coding: utf-8 -*-
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
from math import sqrt
       
def rmse(actual, predicted):
    ln = min(len(predicted), len(actual))
    rmse = sqrt(mean_squared_error(actual[:ln], predicted[:ln]))
    return rmse   
    
if __name__ == "__main__":
    path = sys.argv[1]
    path2 = sys.argv[2]

    MODEL = "S1"

    measurementsFile = os.path.join(path, "2011080100_measurements_{}_2623.txt".format(MODEL))
    noswanFile = os.path.join(path, "2011080100_noswan_{}_48x434.txt".format(MODEL))
    swanFile = os.path.join(path, "2011080100_swan_{}_48x434.txt".format(MODEL))
    hirombFile = os.path.join(path, "2011080100_hiromb_{}_60x434.txt".format(MODEL))
    coeffsFile = os.path.join(path2, "ens_coefs.txt")
#    coeffsFile = "../Data/NewEns/ens_coefs.txt"
    
    measurements = read_data(measurementsFile)
    noswan = read_data(noswanFile)
    swan = read_data(swanFile)
    hiromb = read_data(hirombFile)
    
    coefs = list(read_ens_coeffs(coeffsFile))
     
    cnt = len(noswan)
    learn_len = 50
    validate_len = cnt - learn_len

    classifier = OMEnsembleClassifier([hiromb, swan, noswan],
                                      coefs, measurements, error_measurement=rmse)
    classifier.prepare(1)
        
    errors = []
    forecasts = []
    level = []
    
    operative_time = list(range(50, cnt))    
    for i in operative_time:  
        classifier.train(i, 50)
        pred_fcst, full_fcst = classifier.get_forecasts(i)
        pred, pred_err = classifier.predict_best_ensemble(i)
        full, full_err = classifier.get_biggest_ensemble(i)
        
        best, best_err = classifier.get_best_ensemble(i)
        level.append(best)
                
        rel_err = full_err - pred_err
#        if rel_err > 0:
        if True:
            swan_rmse = rmse(swan[i], measurements[i*6 : i*6 + 48])
            hiromb_rmse = rmse(hiromb[i], measurements[i*6 : i*6 + 48])
            selected_rmse = rmse(pred_fcst, measurements[i*6 : i*6 + 48])
#            _, sec_ens_rmse = classifier.get_ensemble(i, 2)
            full_rmse = rmse(full_fcst, measurements[i*6:i*6+48])
            errors.append([classifier.get_ensemble(i, n)[1] for n in range(8)]+ [pred_err])
#            if swan_rmse >= selected_rmse:
#            if full_rmse<=swan_rmse:
            forecasts.append((rel_err, measurements[i*6 : i*6 + 48],
                              pred_fcst, full_fcst, [h-34 for h in hiromb[i]], swan[i], noswan[i]))
    good_forecasts = list(sorted(forecasts, key = lambda x: x[0]))[-10:]
    
    xs = list(range(len(good_forecasts[0][2])))
#    for forecast in good_forecasts:
#        if len(forecast[1]) != len(xs):
#            continue
#        plt.figure(figsize=[14, 10])
#        mline, = plt.plot(xs, forecast[1], "g-", linewidth=4.0, label="Measurments")
#        seline, = plt.plot(xs, forecast[2], "b-", linewidth=2.0, label = "Selected ensemble")
#        fline, = plt.plot(xs, forecast[3], "r-", linewidth=2.0, label = "Full ensemble")
#        hline, = plt.plot(xs, forecast[4], ":", linewidth=2.0, label="HIROMB")
#        sline, = plt.plot(xs, forecast[5], "--", linewidth=2.0, label="BSM-SWAN")
#        nsline, = plt.plot(xs, forecast[6], "-.", linewidth=2.0, label="BSM-NOSWAN")
#        plt.legend(handles=[mline, seline, fline, hline, sline, nsline], fontsize = 15)
#        plt.xlabel("Time, h", fontsize = 17)
#        plt.ylabel("Water level, sm", fontsize = 17)
#        plt.show()
#        plt.close()
        
    errors_by_time = zip(*errors)
    mean_errors = [mean(e) for e in errors_by_time]
    print(mean_errors)
        
        
#    plt.figure(figsize=[12,8])
#    plt.plot(operative_time, level)
    enss = ["NOSWAN", "SWAN", "NOSWAN, SWAN", "HIROMB", "HIROMB, NOSWAN", "HIROMB, SWAN","HIROMB, SWAN, NOSWAN"]
#    plt.yticks(range(0,7), enss )
#    plt.ylim((-1, 7))
#    plt.show()
#    plt.close()
    
    plt.figure(figsize=[10,10])
    counts, bins, patches = plt.hist(level, bins=8, normed=1, rwidth=0.8, alpha=0.5)
    plt.xticks(range(8))
    
    plt.show()
    plt.close()