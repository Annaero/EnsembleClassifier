# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:05:18 2015

@author: Annaero
"""
import pickle
import sys
import os.path
import os
import matplotlib.pyplot as plt

from collections import OrderedDict
from dateutil import parser
from dateutil.rrule import rrule, HOURLY
from sklearn import linear_model
from numpy.linalg import lstsq

from itertools import product
from numpy.linalg import lstsq

from utils import calibrate_ensemble, calibrate_peak_ensemble

#MODELS = ["BALTP_FORCE_2m", "BALTP_FORCE_90m", "BALTP_GFS60_2m", 
#          "BALTP_GFS60_90m", "BALTP_GFS192_2m", "BALTP_HIRLAM_2m",
#          "BALTP-90M-GFS", "BALTP-90M-HIRLAM", "BSM_FORCE_BankeSmithAss",
#          "BSM_FORCE_WowcSwanAss", "BSM_GFS60_BankeSmithAss", 
#          "BSM_GFS192_BankeSmithAss", "BSM-BS-HIRLAM", "BSM-WOWC-HIRLAM", 
#          "HIROMB"]

TABED_MODELS = ["BSM-BS-HIRLAM", "BALTP-90M-GFS", "BALTP-90M-HIRLAM",
                "BSM-WOWC-HIRLAM", "HIROMB"]

#MODELS = ["BALTP_FORCE_2m", "BALTP_FORCE_90m", "BALTP_GFS60_2m",
#          "BALTP_GFS60_90m", "BALTP_GFS192_2m", "BALTP-90M-GFS",
#          "BSM_FORCE_WowcSwanAss", "BSM-BS-HIRLAM"]

#MODELS = ["BSM-WOWC-HIRLAM", "BSM_GFS60_BankeSmithAss", "BALTP_HIRLAM_2m", "HIROMB"]
MODELS = ["HIROMB", "BSM-WOWC-HIRLAM", "BSM-BS-HIRLAM"]
          
          
DTFORMAT = "%Y-%m-%d %H:%M:%S"
MSM_BEGINING = parser.parse("2013-09-01 00:00:00")

def read_meserments(path, fileName):
    s1DumpPath = os.path.join(path, "S1_dump") 
    grnDumpPath = os.path.join(path, "GRN_dump")  
    filePath = os.path.join(path, fileName)
    
    if not os.path.exists(s1DumpPath):
        GRN = list()
        S1 = list()
        with open(filePath) as mesermentsFile:
            for line in mesermentsFile:
                tokens = line.strip().split("\t")
                GRN.append(float(tokens[0]))
                S1.append(float(tokens[2]))
        pickle.dump(S1, open(s1DumpPath, "wb" ))
        pickle.dump(GRN, open(grnDumpPath, "wb" ))
    else:
        S1 = pickle.load(open(s1DumpPath, "rb" ))
        GRN = pickle.load(open(grnDumpPath, "rb" ))
        
    return GRN, S1
    
def get_msm_index(dt):
    tdelta = dt - MSM_BEGINING
    return int(tdelta.total_seconds() / 3600)
    
def read_forecasts(path, model, point):  
    print(model)
    filePath = os.path.join(path, point, "{0}-{1}.txt".format(model, point))  
    
    times = list()
    predictions = OrderedDict()    
    with open(filePath) as modelFile:
        for line in modelFile:
            if model not in TABED_MODELS:
                tokens = line.strip().split(" ")
                timestr = tokens[0] + " " + tokens[1]
                pr = tokens[2:]
            else:
                tokens = line.strip().split("\t")
                timestr = tokens[0]
                pr = tokens[1:]
            dt = parser.parse(timestr, dayfirst=True, yearfirst=True)
            times.append(dt)
            if model=="HIROMB":    
                predictions[dt] = [float(l) - 34.356 for l in pr]
            else:
                predictions[dt] = [float(l) for l in pr]
            predictLen = len(tokens[2:])
    return predictions, times, predictLen
    
    
if __name__ == "__main__":
    path = sys.argv[1]
    path_to_aligned = sys.argv[2]
    
    GRN, S1 = read_meserments(path, \
        "GI_C1NB_C1FG_SHEP_restored_20130901000000_01.txt")

    POINT = "S1"
    POINT_MSM = S1

    models_forecasts_full = list()
    modeling_times = list()
    forecast_lengths = list()
    for model in MODELS:
        forecasts, times, forecast_len = read_forecasts(path, model, POINT)
        models_forecasts_full.append(forecasts)
        modeling_times.append(times)
        forecast_lengths.append(forecast_len)
        
    #Get max prediction length
    minPredLen = min(forecast_lengths)

    minTime = max([times[0] for times in modeling_times])
    maxTime = min([times[-1] for times in modeling_times])
    times = list(rrule(HOURLY, interval = 6, dtstart=minTime, until=maxTime))
 
    predictors = [list() for mdl in MODELS]
    target = list()
    
    bg = get_msm_index(times[0])
    measurements = POINT_MSM[bg:]
    
    models_forecasts = list(map(lambda model: [model[tm] for tm in times], models_forecasts_full))
    
#    for tm in times:
#        msmIndex = get_msm_index(tm)
#        msm = POINT_MSM[msmIndex : msmIndex + minPredLen]
#        target_len = len(msm)
#        
#        for currentPrediction, predictor \
#                in zip([prd[tm] for prd in models_forecasts], predictors):
#            predictor.extend(currentPrediction[:target_len])
#        target.extend(msm)
             
        
    #def calibrate_ensemble(models_forecasts, ):        
        
    # Find ensembles coefficients and save them to file
    save_coefs_to = os.path.join(path_to_aligned, "ens_coefs.txt")
    with open(save_coefs_to, "w+") as ens_coef_file:
        for coefs in calibrate_ensemble(models_forecasts, measurements):
            form_str = "\t".join(["{{{0}:.3f}}".format(i) for i in range(len(MODELS) + 1)])
            coef_str = form_str.format(*coefs)
            ens_coef_file.write(coef_str + "\n")
            print(coef_str)
            
    with open(os.path.join(path_to_aligned, "peak_ens_coefs.txt"), "w+") as peak_ens_coef_file:
        t, h = calibrate_peak_ensemble(models_forecasts, measurements)
        t_coef_str = form_str.format(*t)
        peak_ens_coef_file.write(t_coef_str + "\n")
        h_coef_str = form_str.format(*h)
        peak_ens_coef_file.write(h_coef_str + "\n")
            
        print(t_coef_str+"\n"+h_coef_str)
            
#            ensemble_predictors = [pred + [1] for pred in ensemble_predictors]

    # Save forecasts aligned by time to new files
    for model, forecasts in zip(MODELS, models_forecasts):
        if not os.path.exists(path_to_aligned):
            os.makedirs(path_to_aligned)
                
        with open(os.path.join(path_to_aligned, model), "w+") as aligned_model_file:
            for forecast in forecasts:
                predStr = "\t".join([str(p) for p in forecast])
                aligned_model_file.write(predStr)
                aligned_model_file.write("\n")
                
    #Save measurment aligned by time
    save_measurments_to = os.path.join(path_to_aligned, "measurments") 
    with open(save_measurments_to, "w+") as aligned_mes_file:
        msms = POINT_MSM[get_msm_index(times[0]):get_msm_index(times[-1]) + forecast_len]
        msms_str = "\n".join([str(m) for m in msms])        
        aligned_mes_file.write(msms_str)
