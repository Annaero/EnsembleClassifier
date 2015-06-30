# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:54:50 2015

@author: magistr
"""

import sys
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from EnsembleClassifiertmp import AssimilationEnsembleClassifier, EnsembleClassifier
from regres import read_data, read_ens_coeffs
from SelectionStrategy import SimpleSelectionStrategy 
from sklearn.metrics import mean_squared_error as MSR    
from statistics import median, mean  
from math import sqrt  
from sklearn.svm import SVR
       
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
#    dist = get_dist_fun(lambda x, y: sqrt(MSR(x,y)))
    classifier = AssimilationEnsembleClassifier([hiromb, swan, noswan], coefs, measurements)
    classifier.prepare(1)  
    svr_classifier = classifier.copy()
    
#    strat1 = NoneStrategy(classifier)
#    strat2 = SimpleSelectionStrategy(classifier2, 0.6)
    
    svm_model = lambda : SVR(kernel='rbf', C=1e6, gamma=0.3)     
    
    cnt = len(noswan)
    learn_len = 70
    start = 35
    pred_len = len(noswan[0])

    mean_errors = []
    ml_mean_errors = []
    correctness_rates = []
    ml_correctness_rates = []
    svr_ml_mean_errors = []
    svr_ml_correctness_rates = []
    
    for kl in range(start, pred_len+1):
        print(kl)
        classifier.find_nearest(knowledge_len = kl)
        svr_classifier.find_nearest(knowledge_len = kl)
        
        correct = 0
        correct_ml = 0
        correct_svr_ml = 0
        
        errors = []
        ml_errors = []
        svr_errors = []
        
        operative_time = list(range(learn_len, cnt))
        for i in operative_time:
            training_set = range(i - learn_len, i)
            classifier.train(training_set)
            svr_classifier.train(training_set, svm_model)
            
            ens, err = classifier.get_nearest_ensemble(i)
            ml, ml_err = classifier.predict_best_ensemble(i)
            svr_ml, svr_ml_err = svr_classifier.predict_best_ensemble(i)
            best_ens, best_err = classifier.get_best_ensemble(i)
            
            errors.append(err)
            ml_errors.append(ml_err) 
            svr_errors.append(svr_ml_err) 
            
            if abs(err-best_err) < 0.0001: #ens==best_ens:
                correct+=1
            if abs(ml_err-best_err) < 0.0001:#ml==best_ens:
                correct_ml+=1
            if abs(svr_ml_err - best_err) < 0.0001:
                correct_svr_ml += 1
        correctness_rate = correct / len(operative_time)
        ml_correctness_rate = correct_ml / len(operative_time)
        svr_ml_correctness_rate = correct_svr_ml / len(operative_time)
        
        correctness_rates.append(correctness_rate)
        ml_correctness_rates.append(ml_correctness_rate)
        svr_ml_correctness_rates.append(svr_ml_correctness_rate)
        
        mean_errors.append(mean(errors))
        ml_mean_errors.append(mean(ml_errors))
        svr_ml_mean_errors.append(mean(svr_errors))
        
    correct_ml = 0
    correct_svr_ml = 0

    ml_errors = []    
    svr_errors = []
    
    classifier = EnsembleClassifier([hiromb, swan, noswan], coefs, measurements)
    classifier.prepare(1)
    svr_classifier = classifier.copy()
    for i in operative_time:
        training_set = range(i - learn_len, i)
        svr_classifier.train(training_set, svm_model)
            
        classifier.train(training_set)
        ml, ml_err = classifier.predict_best_ensemble(i)
        ml_errors.append(ml_err)
        svr_ml, svr_ml_err = svr_classifier.predict_best_ensemble(i)
        svr_errors.append(svr_ml_err)
        best, best_err = classifier.get_best_ensemble(i)
        if abs(ml_err-best_err) < 0.0001:#ml==best_ens:
            correct_ml+=1
        if abs(svr_ml_err - best_err) < 0.0001:
            correct_svr_ml += 1
    cr = correct_ml / len(operative_time)
    srv_cr = correct_svr_ml / len(operative_time)
        
    plt.figure(figsize=[20, 10])
    plt.subplot(1,2,1)
    plt.ylim((0,1))
#    ax1.plot(range(1, pred_len), mean_errors)
    plt.xlabel("Known data length", fontsize = 12)
#    ax1.set_ylabel("Mean error", fontsize = 12)
#    
#    ax2 = ax1.twinx()
    s_line, = plt.plot(range(start, pred_len+1), correctness_rates, c="g", label="Simple")  
    ml_line, = plt.plot(range(start, pred_len+1), ml_correctness_rates, c="r", label="LR")
    svr_line, = plt.plot(range(start, pred_len+1), svr_ml_correctness_rates, c="b", label="SVR")
    
    plt.plot([6,6], [0,1], ":", c="0.5")    
    sml_line, = plt.plot([0, 50],[cr,cr], "--", c="r", label="LR (simple)")
    ssvr_line, = plt.plot([0, 50],[srv_cr, srv_cr], "--", c="b", label="SVR (simple)")
    
    plt.legend(handles=[s_line, ml_line, svr_line, sml_line, ssvr_line], loc=2)
    
    plt.ylabel("Correctness rate")
    
    plt.subplot(1,2,2)      

    plt.xlabel("Known data length", fontsize = 12)

    s_line, = plt.plot(range(start, pred_len+1), mean_errors, c="g", label="Simple")  
    ml_line, = plt.plot(range(start, pred_len+1), ml_mean_errors, c="r", label="LR")
    svr_line, = plt.plot(range(start, pred_len+1), svr_ml_mean_errors, c="b", label="SVR")
      
    mr = mean(ml_errors) 
    svrmr = mean(svr_errors) 
    sml_line, = plt.plot([0, 50],[mr,mr], "--", c="r", label="LR (simple)")
    ssvr_line, = plt.plot([0, 50],[svrmr, svrmr], "--", c="b", label="SVR (simple)")    
    
#    plt.legend(handles=[s_line, ml_line, svr_line, sml_line, ssvr_line], loc=2)
    
    plt.ylabel("Mean error")
    
    plt.show()
    plt.close()