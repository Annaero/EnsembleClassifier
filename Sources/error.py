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
    
    from EnsembleClassifier import EnsembleClassifier
    coefs = list(read_ens_coeffs(coeffsFile))
    classifier = EnsembleClassifier([hiromb, swan, noswan], coefs, measurements)
    classifier.prepare(3) 

    total = len(hiromb)
    max_learn_count = 280
    validate_count = total - max_learn_count
    variants = 20

    ml_errors = []
    ens_errors = []
    best_errors = []
    for learn_count in range(2, max_learn_count):
        print("Learn count {}".format(learn_count))
        ml_avg_list = []
        ens_avg_list = []
        best_avg_list = []
        
        shuffle = ShuffleSplit(total, variants, validate_count, learn_count)
        for (train_set, validate_set) in shuffle:
            ts = list(train_set)
            vs = list(validate_set)
            
            classifier.train(ts)
            
            ml = [classifier.predict_best_ensemble(i)[1] for i in vs]
            ml_avg = sum(ml) / len(vs)
            ml_avg_list.append(ml_avg)
            
            ens = [classifier.get_biggest_ensemble(i)[1] for i in vs]
            ens_avg = sum(ens) / len(vs)
            ens_avg_list.append(ens_avg)
            
            best = [classifier.get_best_ensemble(i)[1] for i in vs]
            best_avg = sum(best) / len(vs)
            best_avg_list.append(best_avg)
            
        ml_errors.append((sum(ml_avg_list)/len(ml_avg_list), min(ml_avg_list), max(ml_avg_list)))
        ens_errors.append((sum(ens_avg_list)/len(ens_avg_list), min(ens_avg_list), max(ens_avg_list)))
        best_errors.append((sum(best_avg_list)/len(best_avg_list), min(best_avg_list), max(best_avg_list)))
    
    plt.figure(figsize=(15,10))
    for errors in [ml_errors, ens_errors, best_errors]:
        mlErrors = list(zip(*errors))
        err = mlErrors[0]
#        for i in range(0, len(mlErrors[0])):
#            me = sum(mlErrors[0][i:i+5])/len(mlErrors[0][i:i+5])
#            err.append(me)
        
        asym = [mlErrors[1], mlErrors[2]]
        
        
        plt.plot(range(2, max_learn_count), err)
    plt.show()
    plt.close()
        
#    plt.errorbar(range(2, max_learn_count), err, asym)
      
#    bestPred = list()
#    mlPred = list()
#    ensPred = list()              
#        
#    plt.plot(range(2, maxLearnCnt), ensMeans, "r-")
#    plt.legend(handles=hd)
#    plt.show()
#    plt.close()
    