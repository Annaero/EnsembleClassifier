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
from collections import Counter

from utils import rmse, correct_forecast, detect_peaks
    
if __name__ == "__main__":
    path = sys.argv[1]
    path2 = sys.argv[2]

    MODEL = "S1"

    measurementsFile = os.path.join(path, "2011080100_measurements_{}_2623.txt".format(MODEL))
    noswanFile = os.path.join(path, "2011080100_noswan_{}_48x434.txt".format(MODEL))
    swanFile = os.path.join(path, "2011080100_swan_{}_48x434.txt".format(MODEL))
    hirombFile = os.path.join(path, "2011080100_hiromb_{}_60x434.txt".format(MODEL))
    coeffsFile = os.path.join(path2, "ens_coefs.txt")
    peak_coeff_file = os.path.join(path2, "peak_ens_coefs.txt")
#    coeffsFile = "../Data/NewEns/ens_coefs.txt"
    
    measurements = read_data(measurementsFile)
    noswan = read_data(noswanFile)
    swan = read_data(swanFile)
    hiromb = read_data(hirombFile)
    hiromb = [[h-34 for h in forecast] for forecast in hiromb]
    
    coefs = list(read_ens_coeffs(coeffsFile))
    T_ens, H_ens = list(read_ens_coeffs(peak_coeff_file))
     
    cnt = len(noswan)
    learn_len = 45
    validate_len = cnt - learn_len

    classifier = OMEnsembleClassifier([hiromb, swan, noswan],
                                      coefs, measurements, error_measurement=rmse)
    classifier.prepare(1)
        
    errors = []
    forecasts = []
    level = []
    sscores=[]
    operative_time = list(range(learn_len, cnt))    
    for i in operative_time:  
        classifier.train(i, learn_len)
        pred_fcst, full_fcst = classifier.get_forecasts(i)
#        pred, pred_err = classifier.predict_best_ensemble(i)
#        full, full_err = classifier.get_biggest_ensemble(i)
        
#        best, best_err = classifier.get_best_ensemble(i)
#        level.append(best)
                
#        rel_err = full_err - pred_err
#        sscores.append(1-pred_err/full_err)
#        errors.append([classifier.get_ensemble(i, n)[1] for n in range(8)]+ [pred_err])
        peaks = detect_peaks(full_fcst)
        if peaks:
            corr_fcst = correct_forecast(full_fcst, [hiromb[i], swan[i], noswan[i]], (T_ens, H_ens))
            if corr_fcst != full_fcst:          
                forecasts.append((measurements[i*6 : i*6 + 48], corr_fcst, full_fcst))
#        if rel_err > 0:
#        if True:
#            swan_rmse = rmse(swan[i], measurements[i*6 : i*6 + 48])
#            hiromb_rmse = rmse(hiromb[i], measurements[i*6 : i*6 + 48])
#            selected_rmse = rmse(pred_fcst, measurements[i*6 : i*6 + 48])
#            _, sec_ens_rmse = classifier.get_ensemble(i, 2)
#            full_rmse = rmse(full_fcst, measurements[i*6:i*6+48])
            
#            if swan_rmse >= selected_rmse:
#            if full_rmse<=swan_rmse:
#            forecasts.append((rel_err, measurements[i*6 : i*6 + 48],
#                              pred_fcst, full_fcst, [h-34 for h in hiromb[i]], swan[i], noswan[i]))
#    good_forecasts = list(sorted(forecasts, key = lambda x: x[0]))[-20:]
    
    xs = list(range(len(forecasts[0][2])))
    for forecast in forecasts:
#        if len(forecast[1]) != len(xs):
#            continue
        plt.figure(figsize=[14, 10])
        mline, = plt.plot(xs, forecast[0], "ko", linewidth=6.0, label="Measurments")
        plt.plot(xs, forecast[1], "k-", linewidth=2.0)
        seline, = plt.plot(xs, forecast[1], "b-", linewidth=2.0, label = "Corrected peaks")
        fline, = plt.plot(xs, forecast[2], "r-", linewidth=2.0, label = "Full ensemble")
#        hline, = plt.plot(xs, forecast[4], ":", linewidth=2.0, label="HIROMB")
#        sline, = plt.plot(xs, forecast[5], "--", linewidth=2.0, label="BSM-SWAN")
#        nsline, = plt.plot(xs, forecast[6], "-.", linewidth=2.0, label="BSM-NOSWAN")
        plt.legend(handles=[mline, seline, fline], fontsize = 15)
        plt.xlabel("Time, h", fontsize = 17)
        plt.ylabel("Water level, cm", fontsize = 17)
        plt.show()
        plt.close()
        
#    errors_by_time = list(zip(*errors))
#    mean_errors = [mean(e) for e in errors_by_time]
#    
#    def skill(ref, score):
#        return 1-score/ref
#    
#    skills = [list(map(lambda x: skill(e[0], x), e)) for e in errors]
#    skills_by_time = list(zip(*skills))
#    mean_skills = [mean(s) for s in skills_by_time]
#    print(mean_errors)
#    print(mean_skills)
        
#        
#    enss = ["", "NOSWAN", "BSM-WOWC-HIRLAM", "BSM-BS-HIRLAM,\nBSM-WOWC-HIRLAM", "HIROMB", "HIROMB,\nBSM-BS-HIRLAM", "HIROMB,\nBSM-WOWC-HIRLAM","HIROMB,\nBSM-WOWC-HIRLAM,\nBSM-BS-HIRLAM"]
    
    
#    c = Counter(level)    
#    plt.figure(figsize=[14,5])
#    fig, ax1 = plt.subplots(figsize=[12,5])
#    ax1.bar(list(c)[1:], list(c.values())[1:], alpha=0.5)
#    ax1.set_xticks([i+0.4 for i in range(8)])
#    ax1.set_xticklabels(enss)
#    ax1.set_xlim((0.5,8)) 
#    
#    ax1.set_ylabel("")
#    ax1.set_xlabel("Ensemble", fontsize=17)    
#    
#    ax2 = ax1.twinx()
#    line, = ax2.plot([i+0.4 for i in range(1,8)], mean_skills[1:-1], "ro-", label="RMSE-SS")
#    ax2.set_ylabel("Ensemble mean RMSE-SS, cm", fontsize=15)
#    ax2.set_ylim([0.55,0.65])
#    ax2.legend(handles=[line])
#    
#    plt.show()
#    plt.close()