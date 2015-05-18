# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:41:58 2015

@author: magistr
"""

import sys
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from EnsembleClassifier import EnsembleClassifier, get_elements
from dtw import dtw
from sklearn.metrics import mean_squared_error, mean_absolute_error

from itertools import combinations

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
       
def dist_mesurement(ts1, ts2):
    ln = min(len(ts1), len(ts2))
    dist, cost, path = dtw(ts1[:ln], ts2[:ln])
    return dist  
#    return mean_squared_error(ts1, ts2)  
#    return mean_absolute_error(ts1, ts2)
       
if __name__ == "__main__":
    path = sys.argv[1]
#    path2 = sys.argv[2]

    MODEL = "S1"

    measurementsFile = os.path.join(path, "2011080100_measurements_{}_2623.txt".format(MODEL))
    noswanFile = os.path.join(path, "2011080100_noswan_{}_48x434.txt".format(MODEL))
    swanFile = os.path.join(path, "2011080100_swan_{}_48x434.txt".format(MODEL))
    hirombFile = os.path.join(path, "2011080100_hiromb_{}_60x434.txt".format(MODEL))
    coeffsFile = os.path.join(path, "ens_coefs.txt")
    
    measurements = read_data(measurementsFile)
    noswan = read_data(noswanFile)
    swan = read_data(swanFile)
    hiromb = [[p - 37.356 for p in predictions] for predictions in read_data(hirombFile)]
    hiromb1 = read_data(hirombFile)
    
    coefs = list(read_ens_coeffs(coeffsFile))
    classifier = EnsembleClassifier([hiromb1, swan, noswan], coefs, measurements)
#    classifier = EnsembleClassifier([hiromb, swan, noswan], [[1,0,0], [0,1,0], [0,0,1]], measurements)
    classifier.prepare(5)
    
#    ensembles = classifier._ensembles
    ensembles = [noswan, swan, hiromb]
    disanses = dict()
    for e1, e2 in combinations(range(len(ensembles)), 2):
        pairs = zip(ensembles[e1], ensembles[e2])
        disanses[(e1,e2)] = [dist_mesurement(ts1, ts2) for ts1, ts2 in pairs]
    
#    plt.figure(figsize=[100, 100]) 
    plt.figure(figsize=[16, 16])
#    plt.title("MAE")
    i=1
    models=["hiromb", "swan", "noswan"]
    for (dist1, dist2) in combinations(disanses.keys(), 2):
#        plt.subplot(21,10,i)
        plt.subplot(2,2,i)
        
        enss = [[list(), list()] for e in coefs]
        for t in range(len(hiromb)):
            e = classifier._best_ensemble_by_time[t]
            enss[e][0].append(disanses[dist1][t])
            enss[e][1].append(disanses[dist2][t])
        
        lines = []
        for e, c in zip(enss, range(len(coefs))):
            if(len(e[0]) > 40):
                label = ", ".join([models[m] for m in range(len(models)) if coefs[c][m]])
                line, = plt.plot(e[0], e[1], "o", markersize=5, label=label)
                lines.append(line)
        
        plt.legend(handles=lines)
        plt.xlim(0,8)
        plt.ylim(0,8)
        plt.xlabel(", ".join([models[dist1[0]], models[dist1[1]]]), fontsize=12)
        plt.ylabel(", ".join([models[dist2[0]], models[dist2[1]]]), fontsize=12)
        i+=1
    plt.show()
    plt.close()
           
    