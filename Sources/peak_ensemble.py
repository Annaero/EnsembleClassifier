# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:49:33 2015

@author: Annaero
"""
import sys, os

import matplotlib.pyplot as plt
from regres import read_data, read_ens_coeffs
from numpy import mean

from utils import window, correct_forecast, calibrate_peak_ensemble,\
                  rmse, detect_peaks, dtw, find_corresponding_peak
from EnsembleClassifier import ensemble_predictions
     
if __name__=="__main__":
    path = sys.argv[1]
    path2 = sys.argv[2]
    MODEL = "GI"
    
    measurementsFile = os.path.join(path,"measurements")
    noswanFile = os.path.join(path, "BSM-BS-HIRLAM")
    swanFile = os.path.join(path, "BSM-WOWC-HIRLAM")
    hirombFile = os.path.join(path, "HIROMB")
    coeffsFile = os.path.join(path, "ens_coefs.txt")
    peakCoeffsFile = os.path.join(path, "peak_ens_coefs.txt")
    
    measurements = read_data(measurementsFile)
    noswan = read_data(noswanFile)
    swan = read_data(swanFile)
    hiromb = read_data(hirombFile)
    coeffs = list(read_ens_coeffs(coeffsFile))
#    peak_coeffs = list(read_ens_coeffs(peakCoeffsFile))
    
    models = [hiromb, swan, noswan]
    forecasts = ensemble_predictions(models, coeffs[-1])
    
    err = rmse    
    
    cnt = len(noswan)
    mean_errors = []
    for thr in range (2, 48):
        rmses = []
        peaks_on = []
        for i in range(cnt):
            msm = measurements[i*6:i*6+48]
            forecast = forecasts[i]
            forecast_error = err(forecast, msm)        
            if len(peaks_on) >= thr:
                window_models = [list(), list(), list()]
                t_msm = []
                for p in peaks_on[-thr:]:
                    window_models[0].append(hiromb[p])
                    window_models[1].append(swan[p])
                    window_models[2].append(noswan[p])
                    t_msm.extend(measurements[p*6:p*6+48])
                peak_coeffs = calibrate_peak_ensemble(window_models, t_msm)
                corrected_forecast = correct_forecast(forecast, [hiromb[i], swan[i], noswan[i]], peak_coeffs)
                corrected_error = err(corrected_forecast, msm)
            else:
                corrected_error = forecast_error
                
#            xs = range(len(msm))
#            if corrected_error < forecast_error:                 
#                plt.figure(figsize=[14, 10])
#                mline, = plt.plot(xs, msm, "ko", linewidth=6.0, label="Measurments")
#                plt.title("{} forecast".format(i))            
#                plt.plot(xs, msm, "k-", linewidth=2.0)
#                seline, = plt.plot(xs, corrected_forecast, "b-", linewidth=2.0, label = "Corrected peaks")
#                fline, = plt.plot(xs, forecast, "r-", linewidth=2.0, label = "Full ensemble")
#                hline, = plt.plot(xs, hiromb[i], ":", linewidth=2.0, label="HIROMB")
#                sline, = plt.plot(xs, swan[i], "--", linewidth=2.0, label="BSM-SWAN")
#                nsline, = plt.plot(xs, noswan[i], "-.", linewidth=2.0, label="BSM-NOSWAN")
#                plt.legend(handles=[mline, seline, fline], fontsize = 15)
#                plt.xlabel("Time, h", fontsize = 17)
#                plt.ylabel("Water level, cm", fontsize = 17)
#                plt.show()
#                plt.close()            
                
            measured_peaks = detect_peaks(msm)
            models_forecasts = [prd[i] for prd in models]
            forecasts_peaks = [detect_peaks(fcst) for fcst in models_forecasts]
            forecasts_peaks_cor = [list(map(lambda x: find_corresponding_peak(x, forecast_peaks),
                                          measured_peaks)) for forecast_peaks in forecasts_peaks]
            if any([all(cor_peaks) for cor_peaks in zip(*forecasts_peaks_cor)]):
                peaks_on.append(i)
                    
                    
            rmses.append((forecast_error, corrected_error))
        print([mean(e) for e in zip(*rmses)])
        mean_errors.append([mean(e) for e in zip(*rmses)])
            
        
    corr_errors = list(zip(*mean_errors))[1]
    plt.plot(range (2, 48), corr_errors)
        
    