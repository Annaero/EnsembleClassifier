# -*- coding: utf-8 -*-
import sys
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from EnsembleClassifier import EnsembleClassifier

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
                levels = levels[0:48]
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
       
if __name__ == "__main__":
    path = sys.argv[1]
    path2 = sys.argv[2]

    MODEL = "S1"

    measurementsFile = os.path.join(path, "2011080100_measurements_{}_2623.txt".format(MODEL))
    noswanFile = os.path.join(path, "2011080100_noswan_{}_48x434.txt".format(MODEL))
    swanFile = os.path.join(path, "2011080100_swan_{}_48x434.txt".format(MODEL))
    hirombFile = os.path.join(path, "2011080100_hiromb_{}_60x434.txt".format(MODEL))
    coeffsFile = os.path.join(path, "ens_coefs.txt")
    
    measurements = read_data(measurementsFile)
    noswan = read_data(noswanFile)
    swan = read_data(swanFile)
    hiromb = read_data(hirombFile)
    
    coefs = list(read_ens_coeffs(coeffsFile))
    classifier = EnsembleClassifier([hiromb, swan, noswan], coefs, measurements)
    classifier.prepare(3) 

    cnt = len(noswan)
    learnCnt = 280
     
    classifier.train(range(1, learnCnt))
    
    bestPred = list()
    mlPred = list()
    ensPred = list()
    
    for i in range(learnCnt, cnt):
        _, mlErr = classifier.predict_best_ensemble(i)
        _, bestErr = classifier.get_best_ensemble(i)
        _, ensErr = classifier.get_biggest_ensemble(i)

        bestPred.append(bestErr)
        mlPred.append(mlErr)
        ensPred.append(ensErr)
                     
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
    plt.savefig(os.path.join(path2, "{}_fig3_DTW_SVR_by4.png".format(MODEL)))
    plt.show()
    plt.close()
    
    binsc = 20
    plt.figure(figsize=(15,10))
    
    bins = np.histogram(mlPred+bestPred+ensPred, bins = binsc)[1]
        
    plt.hist(ensPred, bins, color='blue', normed=1, alpha=1, histtype='step', label="ens") 
    plt.hist(bestPred, bins, color='green' ,normed=1, alpha=0.4, histtype='bar', label="best")
    plt.hist(mlPred, bins, color='red', normed=1, alpha=1, histtype='step', label="ml")
    
    plt.savefig(os.path.join(path2, "{}_fig4_DTW_SVR_by4.png".format(MODEL)))
    plt.show()
    plt.close()
    
    plt.figure(figsize=(15,10))
    x = np.linspace(0, 10, binsc)

    params = norm.fit(ensPred)
    pdf_f = norm.pdf(x, loc=params[0],scale=params[1])
    ensL, = plt.plot(x, pdf_f, color='blue', label="ensemble")
    
    params = norm.fit(bestPred)
    pdf_f = norm.pdf(x, loc=params[0],scale=params[1])
    bestL, = plt.plot(x, pdf_f, color='green', label="best")

    params = norm.fit(mlPred)
    pdf_f = norm.pdf(x, loc=params[0],scale=params[1])
    mlL, =  plt.plot(x, pdf_f, color='red', label="predicted")

    plt.legend(handles=[ensL, bestL, mlL])
    plt.savefig(os.path.join(path2, "{}_fig5_DTW_SVR_by4.png".format(MODEL)))

    plt.show()
    plt.close()
    