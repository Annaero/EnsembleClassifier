# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:49:33 2015

@author: Annaero
"""
import sys, os

import matplotlib.pyplot as plt
from regres import read_data, read_ens_coeffs

from utils import window, correct_forecast, calibrate_peak_ensemble, rmse, detect_peaks, dtw
from EnsembleClassifier import ensemble_predictions
     
if __name__=="__main__":
    path = sys.argv[1]
    path2 = sys.argv[2]
    MODEL = "GI"
    
    measurementsFile = os.path.join(path,"measurments")
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
    
    cnt = len(noswan)
    rmses = []
    peaks_on = []
    for i in range(cnt):
        msm = measurements[i*6:i*6+48]
        forecast = forecasts[i]
        forecast_error = rmse(forecast, msm)        
        if len(peaks_on) > 9:
            window_models = []
            t_msm = []
            for p in peaks_on[-5:]:
                window_models.append((hiromb[p], swan[p], noswan[p]))
                t_msm.extend(measurements[p*6:p*6+48])
            peak_coeffs = calibrate_peak_ensemble(list(zip(*window_models)), t_msm)
            corrected_forecast = correct_forecast(forecast, [hiromb[i], swan[i], noswan[i]], peak_coeffs)
            corrected_error = rmse(corrected_forecast, msm)
        else:
            corrected_error = forecast_error
            
        xs = range(len(msm))
        if corrected_error > forecast_error:
            plt.figure(figsize=[14, 10])
            mline, = plt.plot(xs, msm, "ko", linewidth=6.0, label="Measurments")
            plt.plot(xs, msm, "k-", linewidth=2.0)
            seline, = plt.plot(xs, corrected_forecast, "b-", linewidth=2.0, label = "Corrected peaks")
            fline, = plt.plot(xs, forecast, "r-", linewidth=2.0, label = "Full ensemble")
    #        hline, = plt.plot(xs, forecast[4], ":", linewidth=2.0, label="HIROMB")
    #        sline, = plt.plot(xs, forecast[5], "--", linewidth=2.0, label="BSM-SWAN")
    #        nsline, = plt.plot(xs, forecast[6], "-.", linewidth=2.0, label="BSM-NOSWAN")
            plt.legend(handles=[mline, seline, fline], fontsize = 15)
            plt.xlabel("Time, h", fontsize = 17)
            plt.ylabel("Water level, cm", fontsize = 17)
            plt.show()
            plt.close()            
            
        peaks = detect_peaks(forecast)
        if peaks:
            peaks_on.append(i)
        rmses.append((forecast_error, corrected_error))
            
        
        
        
    