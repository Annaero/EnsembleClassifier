# -*- coding: utf-8 -*-
import sys
from math import sqrt
import os.path
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
import numpy as np
from scipy.stats import norm


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
                levels = levels[1:48] #TODO: Use full length
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
    #ln = min(len(predicted), len(actual))
    #dist, cost, path = dtw(predicted[:ln], actual[:ln])
    #return dist
    return RMSE(predicted, actual)

#TODO: 
        
if __name__ == "__main__":
    path = sys.argv[1]

    measurementsFile = os.path.join(path, "2011080100_measurements_S1_2623.txt")
    noswanFile = os.path.join(path, "2011080100_noswan_S1_48x434.txt")
    swanFile = os.path.join(path, "2011080100_swan_S1_48x434.txt")
    hirombFile = os.path.join(path, "2011080100_hiromb_S1_60x434.txt")
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
            actual = meserments[i*6 + 1:]
            rmses.append(DistMesurement(predicted, actual))
        rmseByEns.append(rmses)
    rmseByTime = zip(*rmseByEns) 
 
    learnCnt = 320 
 
    level = list()
    for pred in rmseByTime:
        best = min(pred)
        currentLevel = pred.index(best)
        level.append(currentLevel)

    transition_count_matrix = [[1] * ensCount for i in range(ensCount)]
    lastLvl = -1
    for lvl in level[0:learnCnt]:
        if lastLvl >= 0:
            transition_count_matrix[lastLvl][lvl] += 1
        lastLvl = currentLevel

    transition_matrix = list()
    for transitions in transition_count_matrix:
        total = sum(transitions)
        probability = [count/float(total) for count in transitions]
        transition_matrix.append(probability)
    
    #meserments = preprocessing.normalize(meserments, norm='l2')
            
    def get_x(i):
        fst = i * 6 - 5
        end = i * 6
        msm = meserments[fst : end+1]     
        X = msm
        return X
       
    lms = [linear_model.LinearRegression(normalize = True) for n in range(ensCount)]
    Xs = [get_x(i) for i in range(1, learnCnt)]
    Xs = preprocessing.scale(Xs)
    
    #Ys = preprocessing.normalize(rmseByEns)
    for lm, rmses in zip(lms, rmseByEns):
        lm.fit(Xs, rmses[1:learnCnt])
    
    def best_predict(Xs, lms, prev):
        cfs = transition_matrix[prev] if prev >=0 else [1] * 7       
        p_rmses = [( lm.predict(Xs) * cf )  for lm, cf in zip(lms, cfs)]
        min_p_rmse = min(p_rmses)
        return p_rmses.index(min_p_rmse)

    bestPred = list()
    worstPred = list() 
    mlPred = list()
    ensPred = rmseByEns[6][learnCnt:]
    
    prevLevel = -1
    for i in range(learnCnt, cnt):
        X = get_x(i)
        
        mlLvl = best_predict(X, lms, prevLevel)
        bestLvl = level[i]
        prevLevel = bestLvl
        
        bestPred.append(rmseByEns[bestLvl][i])
        mlPred.append(rmseByEns[mlLvl][i])
          
    plt.figure(figsize=(20,10))  
#   # assert len(range(learnCnt, cnt)) == len(ensPred)
#    
    mean = sum(ensPred) / (cnt-learnCnt)
    ensL, = plt.plot(range(learnCnt, cnt), ensPred, "r",label = "ensamble {:.3}".format(mean))
    
    mean = sum(bestPred) / (cnt-learnCnt)
    bestL, = plt.plot(range(learnCnt, cnt), bestPred, "*", label = "best {:.2}".format(mean))
    
    mean = sum(mlPred) / (cnt-learnCnt)
    mlL, = plt.plot(range(learnCnt, cnt), mlPred, "c-", label = "classified {:.3}".format(mean))
    
    plt.legend(handles=[ensL, bestL, mlL])
    plt.savefig(os.path.join(path, "S1_fig3_DTW_100_by5.png"))
    plt.show()
    plt.close()
    
    
    binsc = 20
    plt.figure(figsize=(15,10))
    
    bins = np.histogram(mlPred+bestPred+ensPred, bins = binsc)[1]
        
    plt.hist(ensPred, bins, color='blue', normed=1, alpha=1, histtype='step', label="ens") 
    plt.hist(bestPred, bins, color='green' ,normed=1, alpha=0.4, histtype='bar', label="best")
    plt.hist(mlPred, bins, color='red', normed=1, alpha=1, histtype='step', label="ml")
    
    plt.savefig(os.path.join(path, "S1_fig4_DTW_100_by5.png"))
    plt.show()
    plt.close()
    
    plt.figure(figsize=(15,10))
    x = np.linspace(0, 35, binsc)

    params = norm.fit(ensPred)
    pdf_f = norm.pdf(x, loc=params[0],scale=params[1])
    ensL, = plt.plot(x, pdf_f, color='blue', label="ensemble 7")
    
    params = norm.fit(bestPred)
    pdf_f = norm.pdf(x, loc=params[0],scale=params[1])
    bestL, = plt.plot(x, pdf_f, color='green', label="best")

    params = norm.fit(mlPred)
    pdf_f = norm.pdf(x, loc=params[0],scale=params[1])
    mlL, =  plt.plot(x, pdf_f, color='red', label="predicted")

    plt.legend(handles=[ensL, bestL, mlL])
    plt.savefig(os.path.join(path, "S1_fig5_DTW_100_by5.png"))

    plt.show()
    plt.close()
    