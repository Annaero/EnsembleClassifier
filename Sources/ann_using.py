# -*- coding: utf-8 -*-
import sys
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from EnsembleClassifier import ANNEnsembleClassifier
from regres import read_data, read_ens_coeffs   
from statistics import median, mean  

       
def get_dist_fun(distf):
    def dist_mesurement(predicted, actual):
        ln = min(len(predicted), len(actual))
        dist = distf(predicted[:ln], actual[:ln])
        if(type(dist) is tuple):
            dist = dist[0]
        return dist         
    return dist_mesurement
       
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
     
    cnt = len(noswan)
    learn_len = 200
    validate_len = cnt - learn_len
    operative_time = list(range(learn_len, cnt))
    
    errors = []
    classifier = ANNEnsembleClassifier([hiromb, swan, noswan], coefs, measurements)
    classifier.prepare(1)
    for i in operative_time:
        training_set = range(i - learn_len, i)
        classifier.train(training_set)
        
        ml, mlErr = classifier.predict_best_ensemble(i)
        best, bestErr = classifier.get_best_ensemble(i)
        ens, ensErr = classifier.get_biggest_ensemble(i) 
        errors.append((bestErr, ensErr, mlErr))
             
    error_lists = list(zip(*errors))
      
    fig = plt.figure(figsize=[10, 10])
        
    
    plt.boxplot(error_lists, whis = 'range', showmeans = True)
    plt.axhline(y=median(error_lists[-1]), linewidth = 1, color = '0.25',
             linestyle = "--")
    plt.axhline(y=mean(error_lists[-1]), linewidth = 1, color = '0.25',
             linestyle = "--")
    plt.xticks([1,2,3], ["best", "ens", "ml"], fontsize = 11)

    plt.xlabel("Ensemble selection approach", fontsize = 13)
    plt.ylabel("Error", fontsize=13)
    
    plt.show()
    plt.close()