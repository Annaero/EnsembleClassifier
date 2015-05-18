# -*- coding: utf-8 -*-
import sys
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from EnsembleClassifier import EnsembleClassifier
from regres import read_data, read_ens_coeffs
from SelectionStrategy import NoneStrategy     
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
#    errs = []
    
    classifier.prepare(2) 

    cnt = len(noswan)
    learn_len = 70
    validate_len = cnt - learn_len
    
    errors = []
    ml_errors = []
    pta_errors = [] #predicted-to-actual
    strategy = NoneStrategy(classifier)
    operative_time = list(range(learn_len, cnt))
    
    for i in operative_time:
        training_set = range(i-learn_len, i)
        strategy.retrain_classifier(training_set)
        err = strategy.get_next_ensemble(i)
        pta = classifier.get_predict_to_actual_error(i)
        errors.append(err)
        ml_errors.append(err[-2])
        pta_errors.append(pta)
            
    errors_by_time = list(zip(*errors))
    [bestErr, ensErr, mlErr, _] = errors_by_time

    pta_by_ens = list(zip(*pta_errors))
    
#    errs.append(ml_errors)
     
#    errors_by_points = [mean(e) for e in errs]
#    plt.figure(figsize=[7,7])  
#    plt.title("Mean error by history length", fontsize=15)
#    plt.plot(range(1,len(errors_by_points)+1), errors_by_points)
#    plt.xlabel("Points", fontsize=12)
#    plt.ylabel("Mean error", fontsize=12)    
#    plt.show()
    
    plt.figure(figsize=[14,12])
    plt.suptitle("Predicted-to-actual error biplots", fontsize=15)
    for ens, i in zip(pta_by_ens, range(7)):                
        plt.subplot(3,3,i+1)
        models = ["hiromb", "swan", "noswan"]
        plt.title(", ".join([models[m] for m in range(len(models)) if coefs[i][m]]), fontsize=12)

        plt.xlim(0,8)
        plt.ylim(0,8)
        
        [p,a] = list(zip(*ens))
        plt.plot([0,8], [0,8], c="0.5")
        plt.plot(p, a, "*", c="b")        
    plt.show()
    

    plt.figure(figsize=(15,12))      
    
    operative_count = int(len(operative_time) / 3 - 20)
    ml_line, = plt.plot(operative_time[:operative_count], list(mlErr)[:operative_count], label="Predicted") 
    ens_line, = plt.plot(operative_time[:operative_count], list(ensErr)[:operative_count], label="Ensemble")
    best_lb, = plt.plot(operative_time[:operative_count], list(bestErr)[:operative_count], "*", label="Best")
    
    ml_line.set_antialiased(True)
    ens_line.set_antialiased(True)    
    
    plt.ylabel("Mean error", fontsize=15)
    plt.xlabel("Time", fontsize=15)    
    
    plt.legend(handles=[ml_line, ens_line, best_lb], fontsize=12)
    plt.title("Operative mode simulation with training set size=70", fontsize=15) 
    
    plt.show()
    plt.close()
    
    #Error boxplots
    fig = plt.figure(figsize=[7, 7])
    plt.title("Error distribution for operative mode")

    res = plt.boxplot(errors_by_time[:3], whis = 'range', showmeans = True)   
    plt.xticks([1,2,3], ["ml", "ens", "best"], fontsize = 12)
    
    plt.xlabel("Ensemble selection approach", fontsize = 15)
    plt.ylabel("Error", fontsize=15)    
    
#    y2 = host.twinx()    
    plt.axhline(y=median(mlErr), linewidth = 1, color = '0.75', 
             linestyle = "--", label=median(mlErr))
             
    plt.annotate("{0:.2f}".format(median(mlErr)),
                 xy=(3, median(mlErr)), xytext=(3.33, median(mlErr)+0.15)) 
    
    plt.show()
    plt.close()
    
    print("Mean error {}".format(mean(mlErr)))
        
    