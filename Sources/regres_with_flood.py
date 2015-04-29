# -*- coding: utf-8 -*-

import sys
import os
import os.path
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
from sklearn import preprocessing

from sklearn import linear_model

from dateutil.rrule import rrule, HOURLY
from enscalibration import read_meserments, read_predictions, get_msm_index, MODELS
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
#            else:
#                levels = levels[1:] #TODO: Use full length
            data.append(levels) 
            line = dataFile.readline()
    return data

#MODELS = ["BSM-WOWC-HIRLAM", "BSM_GFS60_BankeSmithAss", "BALTP_HIRLAM_2m"]#, "HIROMB"]

if __name__ == "__main__":
    path = sys.argv[1]
    artifacts_path = sys.argv[2]
    
    predictions = []
    for model in MODELS:
        m_file = os.path.join(path, model)
        prediction = read_data(m_file)
        predictions.append(prediction)
    
    measurements = read_data(os.path.join(path, "mesur"))
    
    cnt = len(predictions[0])
    pred_len = len(predictions[0][0])
    
    
#    rmseByMod = list()  
#    for prediction in predictions:
#        errorsByTime = []        
#        for tm in range(cnt-1):
#            pred = prediction[tm]
#            actual = measurements[tm*6+1:]
#            error = DistMesurement(pred, actual)
#            errorsByTime.append(error)
#        rmseByMod.append(errorsByTime)
#        
#    plt.figure(figsize=(20,5))
##
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
    ens_coeff = list(read_ens_coeffs(os.path.join(path, "ens_coef.txt")))[:-1]
    ensCount = len(ens_coeff)
    for coeffs in ens_coeff:
        rmses = list()    
        for i in range(cnt):
            pred = list([p[i] for p in predictions])
            pred.append([1]*pred_len)
            predicted = ensemble_predict(pred, coeffs)
            actual = measurements[i*6:]
            rmses.append(DistMesurement(predicted, actual))
        rmseByEns.append(rmses)
    rmseByTime = zip(*rmseByEns)
    
#    plt.figure(figsize=(20,5))
#    patches = []
#    for errors, coeff in zip(rmseByEns, ens_coeff):
#        patch, = plt.plot(range(cnt-1), errors, label=str(coeff))
#        patches.append(patch)
#    plt.legend(handles=patches)  
#    plt.savefig(os.path.join(os.path.join(artifacts_path, "new_ens_errors.png")))
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
#    

    learnCnt = 200
    validateStart = 220
    
    def get_x(i):
        fst = i * 6 - 4
        end = i * 6
        msm = measurements[fst : end + 1]     
        return msm
       
#    lms = [linear_model.LinearRegression(normalize = True) for n in range(ensCount)]
    lms = [SVR(kernel='rbf', C=1e3, gamma=0.3) for n in range(ensCount)]
#    lms = [SVR(kernel='poly', C=1e3, degree=2) for n in range(ensCount)]
    Xs = [get_x(i) for i in range(cnt)]
#    Xs = preprocessing.scale(Xs)
    
    for lm, rmses in zip(lms, rmseByEns):
        lm.fit(Xs[1:learnCnt], rmses[1:learnCnt])    
    
#    cl = SVC(kernel='linear')#, C=1e3, gamma=0.6) 
#    cl.fit(Xs[1:learnCnt], level[1:learnCnt])
    
#    def best_predict(X):
#        return cl.predict(X)
        
    def best_predict(X, lms):
        p_rmses = [lm.predict(X) for lm in lms]
        min_p_rmse = min(p_rmses)
        return p_rmses.index(min_p_rmse)
        
    bestPred = list()
    worstPred = list() 
    mlPred = list()
    ensPred = rmseByEns[0][validateStart:]
    
    for i in range(validateStart, cnt):
        X = get_x(i)
        
#        mlLvl = best_predict(X)
        mlLvl = best_predict(X, lms)
        bestLvl = level[i]
        
        bestPred.append(rmseByEns[bestLvl][i])
        mlPred.append(rmseByEns[mlLvl][i])
          
    plt.figure(figsize=(10,5))  
    mean = sum(ensPred) / (cnt-validateStart)
    ensL, = plt.plot(range(validateStart, cnt), ensPred, "r",label = "ensamble {:.3}".format(mean))
    
    mean = sum(bestPred) / (cnt-validateStart)
    bestL, = plt.plot(range(validateStart, cnt), bestPred, "*", label = "best {:.2}".format(mean))
    
    mean = sum(mlPred) / (cnt-validateStart)
    mlL, = plt.plot(range(validateStart, cnt), mlPred, "c-", label = "classified {:.3}".format(mean))
    
    plt.legend(handles=[ensL, bestL, mlL])
    plt.savefig(os.path.join(artifacts_path, "new_ens_pred_err.png"))
    plt.show()
    plt.close()