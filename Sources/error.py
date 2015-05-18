# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:24:02 2015

@author: magistr
"""

# -*- coding: utf-8 -*-
import sys
import os.path
import matplotlib.pyplot as plt

from EnsembleClassifier import EnsembleClassifier
from regres import read_data, read_ens_coeffs 
from sklearn.cross_validation import ShuffleSplit
from statistics import mean, median
from SelectionStrategy import NoneStrategy

     
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
    hiromb = read_data(hirombFile)
    
    coefs = list(read_ens_coeffs(coeffsFile))
    classifier = EnsembleClassifier([hiromb, swan, noswan], coefs, measurements)
    
    classifier.prepare(1) 
    strategy = NoneStrategy(classifier)
    
    total = len(hiromb)
    max_learn_count = 200
    validate_count = total - max_learn_count
    variants = 20

    ml_errors = []
    ens_errors = []
    best_errors = []
    for learn_count in range(1, max_learn_count):
        ml_list = []
        ens_list = []
        best_list = []
        
        shuffle = ShuffleSplit(total, variants, validate_count, learn_count)
        for (training_set, validate_set) in shuffle:
            strategy.retrain_classifier(training_set)
            errors = [strategy.get_next_ensemble(i) for i in validate_set]

            errors_by_time = list(zip(*errors))
            [bestErr, ensErr, mlErr, _] = errors_by_time
            ml_list+=list(mlErr)
            ens_list+=list(ensErr)
            best_list+=list(bestErr)
            
        ml_errors.append(mean(ml_list))
        ens_errors.append(mean(ens_list))
        best_errors.append(mean(best_list))
        
    plt.figure(figsize=(10, 10))

    plt.title("Validation set size={}".format(validate_count), fontsize = 15)

    ml_line, = plt.plot(range(max_learn_count-1), ml_errors, label="Predicted") 
    ens_line, = plt.plot(range(max_learn_count-1), ens_errors, label="Ensemble")
    best_lb, = plt.plot(range(max_learn_count-1), best_errors, label="Best")
        
    plt.legend(handles=[ml_line, ens_line, best_lb], fontsize = 12)

    plt.ylabel("Mean error", fontsize = 12)
    plt.xlabel("Training set size", fontsize = 12)

    plt.show()
    plt.close()
    
#        errors_by_points.append(ml_errors[0])
    
#    plt.figure(figsize=[7,7])    
#    plt.title("Mean error by history length", fontsize=15)
#    plt.plot(range(1,len(errors_by_points)+1), errors_by_points)
#    plt.xlabel("Points", fontsize=12)
#    plt.ylabel("Mean error", fontsize=12)    
#    plt.show()