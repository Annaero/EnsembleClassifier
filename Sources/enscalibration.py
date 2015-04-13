# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:05:18 2015

@author: Annaero
"""
import pickle
import sys
import os.path
import matplotlib.pyplot as pyplot

from collections import OrderedDict
from dateutil import parser
from math import sqrt
from dtw import dtw
from dateutil.rrule import rrule, HOURLY
from sklearn import linear_model

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

MODELS = ["BSM-WOWC-HIRLAM", "BSM_GFS60_BankeSmithAss", "BALTP_HIRLAM_2m", "HIROMB"]
          
          
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
    return int(tdelta.seconds / 3600)
    
def read_predictions(path, model, point):  
    print(model)
    filePath = os.path.join(path, point, "{0}-{1}.txt".format(model, point))  
    
    times = list()
    predictions = dict()    
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
            predictions[dt] = [float(l) for l in pr]
            predictLen = len(tokens[2:])
    return predictions, times, predictLen
            
def RMSE(predicted, actual):
    rootErr = sum([ (p-a)**2 for p,a in zip(predicted, actual)]) / len(predicted)   
    return sqrt(rootErr)
    
def dist_mesurement(predicted, actual):
    dist, cost, path = dtw(predicted, actual)
    return dist
    #return RMSE(predicted, actual)
    
def get_combinations(models):
    
    
if __name__ == "__main__":
    path = sys.argv[1]
    
    GRN, S1 = read_meserments(path, \
        "GI_C1NB_C1FG_SHEP_restored_20130901000000_01.txt")


    modelsPredictions = list()
    modelTimes = list()
    predictLengths = list()
    for model in MODELS:
        predictions, times, predictLen = read_predictions(path, model, "S1")
        modelsPredictions.append(predictions)
        modelTimes.append(times)
        predictLengths.append(predictLen)
        
    #Get max prediction length
    minPredLen = min(predictLengths)

    #Append free component
    #modelsPredictions.insert(0, [1]*minPredLen)
       
    #Get time period avaible for files
    minTime = max([times[0] for times in modelTimes])
    maxTime = min([times[-1] for times in modelTimes])
    times = list(rrule(HOURLY, interval=6, dtstart=minTime, until=maxTime))
 
    predictors = [list() for mdl in MODELS]
    target = list()
    for tm in times:
        for currentPrediction, predictor \
                in zip([prd[tm] for prd in modelsPredictions], predictors):
            predictor.extend(currentPrediction[:minPredLen])
        msmIndex = get_msm_index(tm)
        target.extend(S1[msmIndex : msmIndex + minPredLen])
        
    #predictors.append([1] * len(predictors[0]))

    lm = linear_model.LinearRegression(normalize = True)
    lm.fit(list(zip(*predictors)), target)
    lm.get_params() 
    
    lm.predict
    
    ### list(itertools.product([1,0], repeat=4))
        
    
#    msmIndex = get_msm_index(times[0])
#    meserments = S1[msmIndex:]
#    
#    i = 10
#
#    pred = prediction[times[i]] 
#    msm = meserments[i*6:i*6+len(pred)]
#    
#    pyplot.figure()
#
#    pyplot.plot(range(len(msm)), msm, color="green")
#    pyplot.plot(range(len(pred)), pred, color="red")
#    print(dist_mesurement(msm, pred))
#    
#    pyplot.show()
#    pyplot.close()
    