# -*- coding: utf-8 -*-
import sys
import os.path
import matplotlib.pyplot as plt
from EnsembleClassifier import EnsembleClassifier, OMEnsembleClassifier
from regres import read_data, read_ens_coeffs
from SelectionStrategy import NoneStrategy     
from statistics import  mean  
from sklearn.metrics import mean_squared_error
from math import sqrt
       
def rmse(actual, predicted):
    ln = min(len(predicted), len(actual))
    rmse = sqrt(mean_squared_error(actual[:ln], predicted[:ln]))
    return rmse
       
if __name__ == "__main__":
    path = sys.argv[1]
    path2 = sys.argv[2]

    MODEL = "GI"
    
    measurementsFile = os.path.join(path, "2011080100_measurements_{}_2623.txt".format(MODEL))
    noswanFile = os.path.join(path, "2011080100_noswan_{}_48x434.txt".format(MODEL))
    swanFile = os.path.join(path, "2011080100_swan_{}_48x434.txt".format(MODEL))
    hirombFile = os.path.join(path, "2011080100_hiromb_{}_60x434.txt".format(MODEL))
    coeffsFile = os.path.join(path2, "ens_coefs.txt")
    
    measurements = read_data(measurementsFile)
    noswan = read_data(noswanFile)
    swan = read_data(swanFile)
    hiromb = read_data(hirombFile)
    
    coefs = list(read_ens_coeffs(coeffsFile))
     
    cnt = len(noswan)
    learn_len = 45
    validate_len = cnt - learn_len

    classifier = OMEnsembleClassifier([hiromb, swan, noswan], 
                                      coefs, measurements,  error_measurement=rmse)
    
    ml_errors_by_point = []  
    for points in range(1, 10):    
        classifier.prepare(points)
            
        errors = []
        ml_errors = []
        pta_errors = [] #predicted-to-actual
        level = []
        strategy = NoneStrategy(classifier)
        operative_time = list(range(learn_len, cnt))  
       
        
        for i in operative_time:
    #        training_set = range(i - learn_len, i-1)
                
            classifier.train(i, learn_len)
    #        classifier.train(training_set)
            ml, mlErr = classifier.predict_best_ensemble(i)
            best, bestErr = classifier.get_best_ensemble(i)
            ens, ensErr = classifier.get_biggest_ensemble(i) 
            errors.append((bestErr, ensErr, mlErr))
            ml_errors.append(mlErr)
                
    #        strategy.retrain_classifier(i)
    #        err = strategy.get_next_ensemble(i)
            pta = classifier.get_predict_to_actual_error(i)
    #        errors.append(err[:3])
    #        ml_errors.append(err[-2])
            pta_errors.append(pta)
                    
            level.append(ml)
        ml_errors_by_point.append(mean(ml_errors))
    
    plt.figure(figsize=[5,5])
    plt.plot(range(1,10), ml_errors_by_point, "o-")
    plt.xlim([0,10])
    plt.ylim([15.5, 17])
    plt.xlabel("Number of regression members", fontsize=15)
    plt.ylabel("Mean RMS error, cm", fontsize=15) 
    plt.show()
    
            
    errors_by_time = list(zip(*errors))
#        tmp.append(errors_by_time)
#        [bestErr, ensErr, mlErr, _] = errors_by_time
    
    pta_by_ens = list(zip(*pta_errors))
#        css.append(classifier)
#    errs.append(ml_errors)
    
#    plt.figure(figsize=[15,13])
#    plt.suptitle("Predicted-to-actual error biplots", fontsize=15)
#    for ens, i in zip(pta_by_ens[1:], range(1,8)):                
#        plt.subplot(3,3,i)
#        models = ["HIROMB", "BSM-SWAN", "BSM-NOSWAN"]
#        plt.title(", ".join([models[m] for m in range(len(models)) if coefs[i][m]]), fontsize=17)
#
##        plt.xlabel("Predicted RMSE, cm")
##        plt.ylabel("Actual RMSE, cm")
#        
#        mmax = 15        
#        
#        plt.xlim(0 ,mmax)
#        plt.ylim(0, mmax)
#        
#        [p,a] = list(zip(*ens))
#        plt.plot([0, mmax], [0, mmax], c="0.5")
#        plt.plot(p, a, "*", c="b")        
#    plt.show()
    
    [_, e, m] = list(zip(*errors))
    plt.figure(figsize=(7,7))
    plt.scatter(e,m)
    plt.xlim(0,30)
    plt.ylim(0,30)
    plt.plot([0,30],[0,30], "k-", linewidth=1.0)
    plt.xlabel("Full ensemble RMS error, cm", fontsize=17)
    plt.ylabel("Selected ensemble RMS error, cm", fontsize=17)
    plt.show()

#    plt.figure(figsize=(15,12))      
#    
#    operative_count = int(len(operative_time) / 3 - 20)
#    ml_line, = plt.plot(operative_time[:operative_count],
#                        list(mlErr)[:operative_count], label="Predicted") 
#    ens_line, = plt.plot(operative_time[:operative_count],
#                         list(ensErr)[:operative_count], label="Ensemble")
#    best_lb, = plt.plot(operative_time[:operative_count],
#                        list(bestErr)[:operative_count], "*", label="Best")
#    
#    ml_line.set_antialiased(True)
#    ens_line.set_antialiased(True)    
#    
#    plt.ylabel("Mean error", fontsize=15)
#    plt.xlabel("Time", fontsize=15)    
#    
#    plt.legend(handles=[ml_line, ens_line, best_lb], fontsize=12)
#    plt.title("Operative mode simulation with training set size=70", fontsize=15) 
#    
#    plt.show()
#    plt.close()
    
    #Error boxplots
    plt.figure(figsize=[7, 7])
#    plt.suptitle("Error distribution", fontsize = 15)

    plt.boxplot(errors_by_time, whis = 'range', showmeans = True)
    plt.xticks([1,2,3], ["best", "Full", "Selected"], fontsize = 17)
        
#    plt.xlabel("Ensemble", fontsize = 17)
    plt.ylabel("RMSE", fontsize=17)
    
#    plt.axhline(mean(errors_by_time[2]), linestyle="--")
#    plt.yticks(range(0,35))
    plt.ylim([0, 35])
    plt.show()
    plt.close()
    
    plt.hist(level, 8)
   
#    print("Mean error {}".format(mean(mlErr)))
            