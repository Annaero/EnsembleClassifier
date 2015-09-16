# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:49:33 2015

@author: Annaero
"""
import sys, os
from os.path import join as path_join

import matplotlib.pyplot as plt
from regres import read_data, read_ens_coeffs
from numpy import mean

from utils import window, correct_forecast, calibrate_peak_ensemble,\
                  rmse, detect_peaks, dtw, find_corresponding_peak, calculate_peaks_errors
from EnsembleClassifier import ensemble_predictions
     
from enscalibration import MODELS     
     
if __name__=="__main__":
    path = sys.argv[1]
    path2 = sys.argv[2]
    GEO_POINT = "GI"
    
    measurementsFile = path_join(path,"measurements")
    models_files = [path_join(path, model) for model in MODELS]
    coeffsFile = path_join(path, "ens_coefs.txt")
    peakCoeffsFile = path_join(path, "peak_ens_coefs.txt")
    
#    measurementsFile = os.path.join(path, "2011080100_measurements_{}_2623.txt".format(GEO_POINT))
#    model_file_names = ["2011080100_noswan_{}_48x434.txt", 
#                        "2011080100_swan_{}_48x434.txt",
#                        "2011080100_hiromb_{}_60x434.txt"]
#    models_files = [os.path.join(path, model_file_str.format(GEO_POINT)) for model_file_str in model_file_names]
#    coeffsFile = os.path.join(path2, "ens_coefs.txt")
    
    measurements = read_data(measurementsFile)
    coeffs = list(read_ens_coeffs(coeffsFile))
#    peak_coeffs = list(read_ens_coeffs(peakCoeffsFile))
    
    models_count = len(MODELS)
    models = [read_data(model_file) for model_file in models_files]
    forecasts = ensemble_predictions(models, coeffs[-1])
    
    err = rmse    
    
    cnt = len(models[0])
    mean_errors = []
    H_errors = []
    T_errors = []
    max_thr = 15
    peak_level = 80
#    peak_coeffs = calibrate_peak_ensemble(models, measurements)

    for thr in range(2, max_thr):
        rmses = []
        peaks_errors = []
        cor_peaks_errors = []
        peaks_on = []
        mf_peaks_on = []
        imp_corrections = 0
        dec_corrections = 0
        for i in range(cnt):
            msm = measurements[i*6:i*6+48]
            forecast = forecasts[i]
            forecast_error = err(forecast, msm)  
            if len(peaks_on) >= max_thr:
                window_models = [list() for m in range(models_count)]
                t_msm = []
                for p in peaks_on[-thr:]:
                    for m in range(models_count):
                        window_models[m].append(models[m][p])
                    t_msm.extend(measurements[p*6:p*6+48])
                peak_coeffs = calibrate_peak_ensemble(window_models, t_msm, peak_level=peak_level)
                corrected_forecast = correct_forecast(forecast, [f[i] for f in models],
                                                      peak_coeffs, peak_level=peak_level)
                corrected_error = err(corrected_forecast, msm)
                
                peaks_errors.extend(calculate_peaks_errors(msm, forecast, peak_level=peak_level))
                cor_peaks_errors.extend(calculate_peaks_errors(msm, corrected_forecast, peak_level=peak_level))
            else:
                corrected_error = forecast_error 
                peaks_errors.extend(calculate_peaks_errors(msm, forecast, peak_level=peak_level))
                cor_peaks_errors.extend(calculate_peaks_errors(msm, forecast, peak_level=peak_level))
             
            if thr == 24:
                xs = range(len(msm))
                if corrected_error < forecast_error:                 
                    plt.figure(figsize=[14, 10])
                    mline, = plt.plot(xs, msm, "ko", linewidth=6.0, label="Measurments")
                    plt.title("{} forecast".format(i))            
                    plt.plot(xs, msm, "k-", linewidth=2.0)
                    seline, = plt.plot(xs, corrected_forecast, "b-", linewidth=2.0, label = "Corrected peaks")
                    fline, = plt.plot(xs, forecast, "r-", linewidth=2.0, label = "Full ensemble")
#                    hline, = plt.plot(xs, hiromb[i], ":", linewidth=2.0, label="HIROMB")
#                    sline, = plt.plot(xs, swan[i], "--", linewidth=2.0, label="BSM-SWAN")
#                    nsline, = plt.plot(xs, noswan[i], "-.", linewidth=2.0, label="BSM-NOSWAN")
                    plt.legend(handles=[mline, seline, fline], fontsize = 15)
                    plt.xlabel("Time, h", fontsize = 17)
                    plt.ylabel("Water level, cm", fontsize = 17)
#                    plt.show()
                    plt.savefig(path_join(path2, "peak_correction", "{}.png".format(i)))
                    plt.close()  
            
            if corrected_error> forecast_error:
                dec_corrections +=1
            if corrected_error < forecast_error:
                imp_corrections += 1
                
            measured_peaks = detect_peaks(msm, peak_level=peak_level)
            ensemble_peaks = detect_peaks(forecast, peak_level=peak_level)
            models_forecasts = [prd[i] for prd in models]
            forecasts_peaks = [detect_peaks(fcst, peak_level=peak_level) for fcst in models_forecasts]
            forecasts_peaks_cor = [list(map(lambda x: find_corresponding_peak(x, forecast_peaks),
                                          measured_peaks)) for forecast_peaks in forecasts_peaks]                                        
            if any([all(cor_peaks) for cor_peaks in zip(*forecasts_peaks_cor)]):
                peaks_on.append(i)
            if any(map(lambda x: find_corresponding_peak(x, ensemble_peaks), measured_peaks)):
                mf_peaks_on.append(i)
                            
            rmses.append((forecast_error, corrected_error))
        print([mean(e) for e in zip(*rmses)], 
               imp_corrections, dec_corrections, len(mf_peaks_on),
               [mean(e) for e in zip(*peaks_errors)],
               [mean(e) for e in zip(*cor_peaks_errors)])
        mean_errors.append([mean(e) for e in zip(*rmses)])
            
        
    corr_errors = list(zip(*mean_errors))[1]
    plt.plot(range(2, max_thr), corr_errors)
    plt.plot([2, max_thr], [mean_errors[-1][0], mean_errors[-1][0]])
        
    