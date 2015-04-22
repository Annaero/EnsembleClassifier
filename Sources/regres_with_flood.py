# -*- coding: utf-8 -*-

import sys

from dateutil.rrule import rrule, HOURLY
from enscalibration import read_meserments, read_predictions, get_msm_index
from regres import read_ens_coeffs

MODELS = ["BSM-WOWC-HIRLAM", "BSM_GFS60_BankeSmithAss", "BALTP_HIRLAM_2m", "HIROMB"]

if __name__ == "__main__":
    path = sys.argv[1]
    
    GRN, S1 = read_meserments(path, \
        "GI_C1NB_C1FG_SHEP_restored_20130901000000_01.txt")  
    msm_base = S1
    
    modelsPredictionsRaw = list()
    modelTimes = list()
    predictLengths = list()
    for model in MODELS:
        predictions, times, predictLen = read_predictions(path, model, "S1")
        modelsPredictionsRaw.append(predictions)
        modelTimes.append(times)
        predictLengths.append(predictLen)
        
    minPredLen = min(predictLengths)
    
    minTime = max([times[0] for times in modelTimes])
    maxTime = min([times[-1] for times in modelTimes])
    times = list(rrule(HOURLY, interval=6, dtstart=minTime, until=maxTime))
    
    
    
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