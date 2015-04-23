# -*- coding: utf-8 -*-

import sys
import os
import os.path
import matplotlib.pyplot as plt

from dateutil.rrule import rrule, HOURLY
from enscalibration import read_meserments, read_predictions, get_msm_index
from regres import read_ens_coeffs, DistMesurement, ensemble_predict

def read_data(fileName):
    data = list()
    with open(fileName, "r") as dataFile:
        line = dataFile.readline()
        while line:
            levels = line.strip().split("\t")
            levels =  [float(lvl) for lvl in levels]
            if len(levels) == 1:
                levels = levels[0]
            else:
                levels = levels[1:] #TODO: Use full length
            data.append(levels) 
            line = dataFile.readline()
    return data

MODELS = ["BSM-WOWC-HIRLAM", "BSM_GFS60_BankeSmithAss", "BALTP_HIRLAM_2m"]#, "HIROMB"]

if __name__ == "__main__":
    path = sys.argv[1]
    
    predictions = []
    for model in MODELS:
        m_file = os.path.join(path, model)
        prediction = read_data(m_file)
        predictions.append(prediction)
    
    mesurments = read_data(os.path.join(path, "mesur"))
    
    cnt = len(predictions[0])
    pred_len = len(predictions[0][0])
    
#    rmseByMod = list()  
#    for prediction in predictions:
#        errorsByTime = []        
#        for tm in range(cnt-1):
#            pred = prediction[tm]
#            actual = mesurments[tm*6+1:]
#            error = DistMesurement(pred, actual)
#            errorsByTime.append(error)
#        rmseByMod.append(errorsByTime)
#        
#    plt.figure(figsize=(20,5))
#
 
#    handles = []
#    for errors, model in zip(rmseByMod, MODELS):
#        lbl, = plt.plot(range(cnt-1), errors, label = model)
#        handles.append(lbl)
#    plt.legend(handles=handles)
#    plt.savefig("error_model.tmp.png")
#        
#    plt.show()  
      
    rmseByEns = list()
    for coeffs in read_ens_coeffs(os.path.join(path, "ens_coef.txt")):
        #ensCount += 1
        rmses = list()    
        for i in range(cnt-1):
            pred = list([p[i] for p in predictions])
            pred.append([1]*pred_len)
            predicted = ensemble_predict(pred, coeffs)
            actual = mesurments[i*6 + 1:]
            rmses.append(DistMesurement(predicted, actual))
        rmseByEns.append(rmses)
    rmseByTime = zip(*rmseByEns)
    
#    plt.figure(figsize=(20,5))
#    for errors in rmseByEns:
#        plt.plot(range(cnt-1), errors)
#    plt.show()
    
    level = list()
    for pred in rmseByTime:
        best = min(pred)
        currentLevel = pred.index(best)
        level.append(currentLevel)
    
#    plt.figure(figsize=(20,5))
#    for errors in rmseByEns:
#        plt.plot(range(cnt-1), level)
#    plt.show() 
    
    
    
#    level = list()
#    for pred in rmseByTime:
#        best = min(pred)
#        currentLevel = pred.index(best)
#        level.append(currentLevel)
    
#    ensCount = 0
#    rmseByEns = list()
#    for coeffs in read_ens_coeffs(coeffsFile):
#        ensCount += 1
#        rmses = list()    
#        for i in range(cnt):
#            ensembles = [hiromb[i], swan[i], noswan[i], add]
#            predicted = ensemble_predict(ensembles, coeffs)
#            actual = meserments[i*6 + 1:]
#            rmses.append(DistMesurement(predicted, actual))
#        rmseByEns.append(rmses)
#    rmseByTime = zip(*rmseByEns) 
 
#    predictors = [list() for mdl in MODELS]
#    target = list()
#    for tm in times:
#        for currentPrediction, predictor \
#                in zip([prd[tm] for prd in modelsPredictions], predictors):
#            predictor.extend(currentPrediction[:minPredLen])
#        msmIndex = get_msm_index(tm)
#        target.extend(msm_base[msmIndex : msmIndex + minPredLen])