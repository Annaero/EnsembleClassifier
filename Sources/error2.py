# -*- coding: utf-8 -*-

import sys
import os
import os.path
import matplotlib.pyplot as plt
from enscalibration import MODELS
from regres import read_ens_coeffs
from EnsembleClassifier import EnsembleClassifier 
from SelectionStrategy import NoneStrategy 

from sklearn.cross_validation import ShuffleSplit
from math import sqrt


def read_data(fileName):
    data = list()
    with open(fileName, "r") as dataFile:
        line = dataFile.readline()
        while line:
            levels = line.strip().split("\t")
            levels =  [float(lvl) for lvl in levels]
            if len(levels) == 1:
                levels = levels[0]
#            else:
#                levels = levels[1:] #TODO: Use full length
            data.append(levels) 
            line = dataFile.readline()
    return data

if __name__ == "__main__":
    path = sys.argv[1]
    artifacts_path = sys.argv[2]
    
    predictions = []
    for model in MODELS:
        m_file = os.path.join(path, model)
        prediction = read_data(m_file)
        predictions.append(prediction)
    
#    coeffsFile = os.path.join("../Data/", "ens_coefs.txt")
    coeffsFile = os.path.join(path, "ens_coef.txt")
    
    measurements = read_data(os.path.join(path, "mesur"))
    
    coefs = list(read_ens_coeffs(coeffsFile))
    coefs.reverse()
    
    
    cnt=len(predictions[0])
    
    classifier = EnsembleClassifier(predictions, coefs, measurements)
    classifier.prepare(1)
    strategy = NoneStrategy(classifier)

    errors = []
    ml_errors = []
    pta_errors = [] #predicted-to-actual
    
    learn_len = 10
    operative_time = list(range(learn_len, cnt))    
    for i in operative_time:
        training_set = range(i - learn_len, i-1)
            
#        classifier.train(training_set)
#        ml, mlErr = classifier.predict_best_ensemble(i)
#        best, bestErr = classifier.get_best_ensemble(i)
#        ens, ensErr = classifier.get_biggest_ensemble(i) 
#        errors.append((bestErr, ensErr, mlErr))
            
        strategy.retrain_classifier(training_set)
        err = strategy.get_next_ensemble(i)
        pta = classifier.get_predict_to_actual_error(i)
        errors.append(err[:3])
        ml_errors.append(err[-2])
        pta_errors.append(pta)
                
            
        errors_by_time = list(zip(*errors))
#        tmp.append(errors_by_time)
        [bestErr, ensErr, mlErr] = errors_by_time
    
        pta_by_ens = list(zip(*pta_errors))
#        css.append(classifier)
#    errs.append(ml_errors)
    
    plt.figure(figsize=[14,12])
    qs = abs(sqrt(len(coefs)))
    plt.suptitle("Predicted-to-actual error biplots", fontsize=15)
    for ens, i in zip(pta_by_ens, range(len(pta_by_ens))):                
        plt.subplot(qs,qs,i+1)
        models = MODELS
        plt.title(", ".join([models[m] for m in range(len(models)) if coefs[i][m]]), fontsize=12)

        plt.xlabel("predicted")
        plt.ylabel("actual")        
        
        plt.xlim(0,6)
        plt.ylim(0,6)
        
        [p,a] = list(zip(*ens))
        plt.plot([0,10], [0,10], c="0.5")
        plt.plot(p, a, "*", c="b")        
    plt.show()
    

    plt.figure(figsize=(30,12))      
    
    operative_count = int(len(operative_time) / 3 - 20)
    ml_line, = plt.plot(operative_time,
                        list(errors_by_time[2]), label="Predicted") 
    ens_line, = plt.plot(operative_time,
                         list(errors_by_time[1]), label="Ensemble")
    best_lb, = plt.plot(operative_time,
                        list(errors_by_time[0]), "*", label="Best")
    
    ml_line.set_antialiased(True)
    ens_line.set_antialiased(True)    
    
    plt.ylabel("Mean error", fontsize=15)
    plt.xlabel("Time", fontsize=15)    
    
    plt.legend(handles=[ml_line, ens_line, best_lb], fontsize=12)
    plt.title("Operative mode simulation with training set size=70", fontsize=15) 
    
    plt.show()
    plt.close()
    
    #Error boxplots
    plt.figure(figsize=[7, 7])
#    plt.suptitle("Error distribution", fontsize = 15)

    plt.boxplot(errors_by_time, showmeans = True) #, whis = 'range')
    plt.xticks([1,2,3], ["best", "ens", "ml"], fontsize = 11)
        
    plt.xlabel("Ensemble selection approach", fontsize = 13)
    plt.ylabel("Error", fontsize=13)
    
    plt.axhline(mean(errors_by_time[2]), linestyle="--")
#    plt.yticks(range(0,10))
    plt.show()
    plt.close()