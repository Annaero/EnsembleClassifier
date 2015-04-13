# -*- coding: utf-8 -*-
import sys
from math import sqrt
import os.path
import matplotlib.pyplot as plt
from functools import reduce
from sklearn import svm
from nbaes import NBaes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

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

#TODO: 
        
if __name__ == "__main__":
    path = sys.argv[1]

    measurementsFile = os.path.join(path, "2011080100_measurements_GI_2623.txt")
    noswanFile = os.path.join(path, "2011080100_noswan_GI_48x434.txt")
    swanFile = os.path.join(path, "2011080100_swan_GI_48x434.txt")
    hirombFile = os.path.join(path, "2011080100_hiromb_GI_60x434.txt")
    coeffsFile = os.path.join(path, "ens_coefs.txt")
    
    meserments = read_data(measurementsFile)
    noswan = read_data(noswanFile)
    swan = read_data(swanFile)
    hiromb = read_data(hirombFile)
    
    cnt = len(swan)
    add = [1] * cnt
    
    plt.figure(figsize=(20,5))
    
    rmseByEns = list()
    for coeffs in read_ens_coeffs(coeffsFile):
        rmses = list()    
        for i in range(cnt):
            ensembles = [hiromb[i], swan[i], noswan[i], add]
            predicted = ensemble_predict(ensembles, coeffs)
            actual = meserments[i*6 + 1:]
            rmses.append(RMSE(predicted, actual))
        plt.plot(range(cnt), rmses)
        rmseByEns.append(rmses)
        
    plt.savefig(os.path.join(path, "fig1.png"))
    
    level = list()
    wlevel = list()
    for pred in zip(*rmseByEns):
        best = min(pred)
        worst = max(pred)
        level.append(pred.index(best) + 1)
        wlevel.append(pred.index(worst) + 1)
            
    plt.close()
    plt.figure(figsize=(20,5))
    plt.ylim([0, 8])
    plt.plot(range(cnt), level)
    plt.savefig(os.path.join(path, "fig2.png"))
    plt.close()
    
    learnCnt = 150
    
    one = min(meserments)
    nr = 1-one
    meserments =  list([el+nr for el in meserments])     
    
    def get_predictors(msm):
        for i in range(len(msm)-1, 0,-1):
            #yield msm[i]-msm[i-1] #difference
            yield (msm[i]-msm[i-1])/msm[i] #difference in persents of previous
            
    def get_lm_predictors(msm):
        predictors = [[x] for x in range(len(msm))]
        clm = linear_model.LinearRegression(normalize = True)
        clm.fit(predictors, msm)
       # print(list(clm.coef_) + [clm.intercept_])
        return list(clm.coef_) + [clm.intercept_]
        
    def normalize(vctr):
        lt = min(vctr)
        return [el - lt for el in vctr]
        
    def get_x(i):
        fst = i*6 - 5
        end = i*6
        msm = meserments[fst : end+1]     
        #assert len(msm) == 3
        
        X=list(get_predictors(msm))
        #X = meserments[fst : end+1]
        #msm = list(get_predictors(normalize(msm)))
        #X = get_lm_predictors(normalize(msm))
        #X = normalize(meserments[fst : end+1])
        return X
        
        
    Ys = [lvl-1 for lvl in level[1:learnCnt]] #we can't do prediction for first point
    Xs = list()
    for i in range(1, learnCnt):
        X = get_x(i)
        Xs.append(X)
    #clf = NBaes()
    VS = list(zip(Xs, Ys))
    for i in range(1, 7):
        vls = list([v[0] for v in VS if v[1] == i ]   ) 
        #print(vls)
       # plt.plot(list([x[0] for x in vls]), list([x[1] for x in vls]), "*")
#    plt.show()  
#    plt.close()      
        
    clf = svm.SVC(kernel='linear')
    clf.fit(Xs, Ys)   
    print(clf.score)
        
    bestPred = list()
    worstPred = list() 
    mlPred = list()
    ensPred = rmseByEns[6][learnCnt:]
    for i in range(learnCnt, cnt):
        X = get_x(i)
        
        mlLvl = clf.predict(X)
        bestLvl = level[i]
        worstLvl = wlevel[i]
        
        bestPred.append(rmseByEns[bestLvl-1][i])
        worstPred.append(rmseByEns[worstLvl-1][i])
        mlPred.append(rmseByEns[mlLvl][i])
          
    plt.figure(figsize=(10,5))  
   # assert len(range(learnCnt, cnt)) == len(ensPred)
    
    mean = reduce(lambda x, y: x + y, ensPred) / (cnt-learnCnt)
    ensL, = plt.plot(range(learnCnt, cnt), ensPred, label = "ensamble {:.3}".format(mean))
    
    mean = reduce(lambda x, y: x + y, bestPred) / (cnt-learnCnt)
    bestL, = plt.plot(range(learnCnt, cnt), bestPred, "o", label = "best {:.2}".format(mean))
    
    mean = reduce(lambda x, y: x + y, mlPred) / (cnt-learnCnt)
    mlL, = plt.plot(range(learnCnt, cnt), mlPred, "c-", label = "classified {:.3}".format(mean))
    
    mean = reduce(lambda x, y: x + y, worstPred) / (cnt-learnCnt)
    mlW, = plt.plot(range(learnCnt, cnt), worstPred, "r-", label = "worst {:.2}".format(mean))
    
    plt.legend(handles=[ensL, bestL, mlL, mlW])
    #plt.savefig(os.path.join(path, "fig3_baes.png"))
    plt.show()
    plt.close()
    
