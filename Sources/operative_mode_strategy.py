# -*- coding: utf-8 -*-
import sys
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from EnsembleClassifier import EnsembleClassifier
from regres import read_data, read_ens_coeffs
from SelectionStrategy import SimpleSelectionStrategy     
from statistics import median, mean    
       
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
    classifier.prepare(2) 

    cnt = len(noswan)
    learn_len = 70
    validate_len = cnt - learn_len
     
    
    byth = []
    errors_by_th = []
    for threshold in np.linspace(0, 1.5, 60):
        errors = []
        strategy = SimpleSelectionStrategy(classifier, threshold)
        for i in range(learn_len, cnt):
            training_set = list(range(i-learn_len, i))
            strategy.retrain_classifier(training_set)
            err = strategy.get_next_ensemble(i)
            errors.append(err)
            
        errors_by_th.append(errors)
        errors_by_time = list(zip(*errors))
        byth.append((threshold, mean(errors_by_time[-1]), mean(errors_by_time[-2])))
                
        plt.figure(figsize=(10,12))
#        plt.title("Error dis".format(threshold) )

        plt.axhline(y=mean(errors_by_time[-1]), linewidth = 1, color = '0.25', 
             linestyle = "--")
        
        plt.boxplot(errors_by_time, whis='range', showmeans=True)
        plt.xticks([1, 2, 3, 4], ['best', 'ens', 'ml', "cons-ml"])
    
        path3 = os.path.join(path2, "strat")
        plt.savefig(os.path.join(path3, "100_{}.png".format(threshold)))
        
#        plt.show()
        plt.close()
        
    [time, mn, ml] = list(zip(*byth))

    plt.figure(figsize=(7,7))           
     
    mnl, = plt.plot(list(time), list(mn), label="Conservative")
    mdl, = plt.plot(list(time), list(ml), label="Non-conservative")     
    
    plt.ylabel("Mean error", fontsize=15)
    plt.xlabel("Threshold", fontsize=15)  
    
    mnl.set_antialiased(True)
    
    plt.legend(handles=[mnl, mdl], fontsize=12)
    
#    plt.title("Mean error for different threshold", fontsize=15) 
    
#    plt.savefig(os.path.join(path2, "mean_strat_100.png"))    
    
    plt.show()
    plt.close()
#    
#    plt.figure(figsize=[10, 7])
#
#    
#    res = plt.boxplot(errors_by_th, whis='range', showmeans=True)   
##    plt.xticks(range(30), np.linspace(0, 1.5, 30), fontsize=12)    
#    
##    plt.plot(np.linspace(0, 1.5, 30), res["means"])    
##    
##    plt.axhline(y=min([mean(e) for e in errors_by_th]), 
##                    linewidth=1, color='0.75', linestyle="--")
#                    
#    plt.xlabel("Threshold", fontsize=15)
#    plt.ylabel("Error", fontsize=15)
#    
#    plt.show()
#    plt.close()
#    