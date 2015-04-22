# -*- coding: utf-8 -*-

import sys
import os
import os.path
import matplotlib.pyplot as plt

from dateutil.rrule import rrule, HOURLY
from enscalibration import read_meserments, read_predictions, get_msm_index
from regres import read_ens_coeffs, read_data, DistMesurement

MODELS = ["BSM-WOWC-HIRLAM", "BSM_GFS60_BankeSmithAss", "BALTP_HIRLAM_2m", "HIROMB"]

if __name__ == "__main__":
    path = sys.argv[1]
    
    predictions = []
    for model in MODELS:
        m_file = os.path.join(path, model)
        prediction = read_data(m_file)
        predictions.append(prediction)
    
    mesurments = read_data(os.path.join(path, "mesur"))
    
    rmseByEns = list()
    cnt = len(predictions[0])
    for prediction in predictions:
        errorsByTime = []        
        for tm in range(cnt):
            pred = prediction[tm]
            actual = mesurments[tm*6+1:]
            error = DistMesurement(pred, actual)
            errorsByTime.append(error)
        rmseByEns.append(errorsByTime)
        
    plt.figure(figsize=(20,5))

    for errors in rmseByEns:
        plt.plot(range(cnt), errors)
        
    plt.show()
    
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