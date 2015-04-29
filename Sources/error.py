# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:24:02 2015

@author: magistr
"""

# -*- coding: utf-8 -*-
import sys
from math import sqrt
import os.path
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
import numpy as np
from scipy.stats import norm

from sklearn.svm import SVR
from sklearn.svm import SVC

from dtw import dtw

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
                levels = levels[0:48] #TODO: Use full length
            data.append(levels) 
            line = dataFile.readline()
    return data
    
def read_ens_coeffs(fileName):
    with open(fileName, "r") as dataFile:
        line = dataFile.readline()
        while line:
            coefs = line.strip().split("\t")
            yield [float(e) for e in coefs]
            line = dataFile.readline()

def ensemble_predict(ensembles, coeffs):
    pred = [list(ens) for ens in zip(*ensembles)]
    return [ sum([ p*c for p,c in zip(ens,coeffs)]) for ens in pred ]  

def RMSE(predicted, actual):
    rootErr = sum([ (p-a)**2 for p,a in zip(predicted, actual)]) / len(predicted)   
    return sqrt(rootErr)
    
def DistMesurement(predicted, actual):
    ln = min(len(predicted), len(actual))
    dist, cost, path = dtw(predicted[:ln], actual[:ln])
    return dist
#    return RMSE(predicted, actual)

#TODO: 
        
if __name__ == "__main__":
    path = sys.argv[1]
#    path2 = sys.argv[2]

    MODEL = "S1"

    measurementsFile = os.path.join(path, "2011080100_measurements_{}_2623.txt".format(MODEL))
    noswanFile = os.path.join(path, "2011080100_noswan_{}_48x434.txt".format(MODEL))
    swanFile = os.path.join(path, "2011080100_swan_{}_48x434.txt".format(MODEL))
    hirombFile = os.path.join(path, "2011080100_hiromb_{}_60x434.txt".format(MODEL))
    coeffsFile = os.path.join(path, "ens_coefs.txt")
    
    meserments = read_data(measurementsFile)
    noswan = read_data(noswanFile)
    swan = read_data(swanFile)
    hiromb = read_data(hirombFile)
    
    cnt = len(swan)
    ensCount = 0
    add = [1] * cnt
    
    plt.figure(figsize=(20,5))
    
    rmseByEns = list()
    for coeffs in read_ens_coeffs(coeffsFile):
        ensCount += 1
        rmses = list()
        for i in range(cnt):
            ensembles = [hiromb[i], swan[i], noswan[i], add]
            predicted = ensemble_predict(ensembles, coeffs)
            actual = meserments[i*6:]
            rmses.append(DistMesurement(predicted, actual))
        rmseByEns.append(rmses)
    rmseByTime = zip(*rmseByEns)
 
 
    level = list()
    for pred in rmseByTime:
        best = min(pred)
        currentLevel = pred.index(best)
        level.append(currentLevel)
      
    maxLearnCnt  = 320
    validationStart = maxLearnCnt
    
    plt.figure(figsize=(20,10))  
    plt.xticks(range(0, maxLearnCnt, 10))  
    
    hd = []
    
    for dd in range(0, 3):
        def get_x(i):
            fst = i * 6 - dd
            end = i * 6
            msm = meserments[fst : end+1]     
            X = msm
            return X    
        
        means = []
        ensMeans = []
        for learnCnt in range(2, maxLearnCnt):    
        #    lms = [linear_model.LinearRegression(normalize = True) for n in range(ensCount)]
            lms = [SVR(kernel='rbf', C=1e3, gamma=0.3) for n in range(ensCount)]
            Xs = [get_x(i) for i in range(1, cnt+1)]
        #    Xs = preprocessing.scale(Xs)
            
#            cl = SVC(kernel='rbf', C=1e3, gamma=0.3)    
            
            for lm, rmses in zip(lms, rmseByEns):
                lm.fit(Xs[1:learnCnt], rmses[1:learnCnt])
            
        #    cl.fit(Xs[1:learnCnt], level[1:learnCnt])    
            
            def best_predict(X, lms):      
                p_rmses = [lm.predict(X) for lm in lms]
                min_p_rmse = min(p_rmses)
                return p_rmses.index(min_p_rmse)
                
        #    def best_predict(X):
        #        return cl.predict(X)
        
            bestPred = list()
            worstPred = list() 
            mlPred = list()
    
            better_count = 0
            same_count = 0
            
            for i in range(validationStart, cnt):
                X = get_x(i)
                
                mlLvl = best_predict(X, lms)
                bestLvl = level[i]
                
                bestPred.append(rmseByEns[bestLvl][i])
                mlPred.append(rmseByEns[mlLvl][i])
    
               
            ensMeans.append(sum(rmseByEns[6][validationStart:cnt]) / (cnt-validationStart))
            mean = sum(mlPred) / (cnt-validationStart)
            means.append(mean)
                

        mlL, = plt.plot(range(2, maxLearnCnt), means, label = "{} points".format(dd+1))
        hd.append(mlL)
        
    plt.plot(range(2, maxLearnCnt), ensMeans, "r-")
    plt.legend(handles=hd)
    plt.show()
    plt.close()
    