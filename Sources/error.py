# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:24:02 2015

@author: magistr
"""

# -*- coding: utf-8 -*-
import sys
import os.path
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from EnsembleClassifier import EnsembleClassifier
from EnsembleClassifier import ANNEnsembleClassifier
from regres import read_data, read_ens_coeffs 
from sklearn.cross_validation import ShuffleSplit
from statistics import mean, median
from SelectionStrategy import NoneStrategy

from numpy import percentile, std

     
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
    #classifier = EnsembleClassifier([hiromb, swan, noswan], coefs, measurements)
    classifier = EnsembleClassifier([hiromb, swan, noswan], coefs, measurements)
    classifier.prepare(1)
   # classifier2 = classifier.copy()
    
    total = len(hiromb)    
    
    #svm_model = lambda : SVR(kernel='rbf', C=1e6, gamma=0.3)    
        
#        learn_count = 30
    lin = []
    bst = []
    ns = []
    ts_sizes = range(1,250) #250) #[30, 70, 120, 145]
    for learn_count in ts_sizes:
        validate_count = total - 250
        variants = 40    
        
        predicted = []
        best = []
        ens = []
        shuffle = ShuffleSplit(total, variants, validate_count, learn_count)
        for (training_set, validate_set) in shuffle:
            classifier.train(training_set)
                #classifier2.train(training_set, regression_model=svm_model)
                
            predicted += [classifier.predict_best_ensemble(i)[1] for i in validate_set] 
            best += [classifier.get_best_ensemble(i)[1] for i in validate_set] 
            ens += [classifier.get_biggest_ensemble(i)[1] for i in validate_set]
               # svr += [classifier2.predict_best_ensemble(i)[1] for i in validate_set]
           # errors_by_points_svr.append(mean(svr))
        #errors_by_ts.append((errors_by_points_linear, errors_by_points_svr))
        lmn = mean(predicted)
        lin.append((lmn, min(predicted), max(predicted)))
        bst.append((mean(best), min(best), max(best)))
        ns.append((mean(ens), min(ens), max(ens)))

#        lin.append((lmn, lmn-std(predicted)/2, lmn+std(predicted)/2))
#        bst.append((mean(best), mean(best)-std(best)/2, mean(best)+std(best)/2))
#        ns.append((mean(ens), mean(ens)-std(ens)/2, mean(ens)+std(ens)/2))        
        
#        bst.append((mean(best), percentile(best, 25), percentile(best, 75)))
        
    [lmean, l25, l75] = list(zip(*lin))  
    [bmean, b25, b75] = list(zip(*bst))  
    [emean, e25, e75] = list(zip(*ns))    
        
    plt.figure(figsize=[10,10])
    plt.plot(ts_sizes, lmean, "r-", label="Linear regression")
    plt.fill_between(ts_sizes, l25, l75, facecolor="red", alpha=0.2)    
    
    plt.plot(ts_sizes, emean, "b-", label="Linear regression")
    plt.fill_between(ts_sizes, e25, e75, facecolor="blue", alpha=0.5)    
    
#    plt.plot(ts_sizes, bmean, "g-", label="Linear regression")
#    plt.fill_between(ts_sizes, b25, b75, facecolor="green", alpha=0.5)  
    plt.show()
    plt.close()
    
#    plt.suptitle("Mean error by history length\nvalidation set size={0}".format(validate_count),
#                        fontsize = 15)
#    for (errors_by_points_linear, errors_by_points_svr), i in zip(errors_by_ts, ts_sizes):
#        #plt.subplot(2,2, ts_sizes.index(i))    
#        plt.title("Training set size={0}".format(i),
#                      fontsize=15)    
#        
#        l_line, = plt.plot(range(1, max_points), errors_by_points_linear, "-o", label="Linear regression")
#        #svr_line, = plt.plot(range(1, max_points), errors_by_points_svr, "-*", label="SVR")
#    
#        #plt.legend(handles=[l_line, svr_line])   
#        
#        plt.ylim(3.25, 3.75)   
#        
#        plt.xlabel("Points", fontsize=12)
#        plt.ylabel("Mean error", fontsize=12)    
#    plt.show()
#        ens_errors.append(mean(ens_list))
#        best_errors.append(mean(best_list))
        
#    fig = plt.figure(figsize=[7, 7])
#    plt.title("Error distribution for operative mode")
#
#    diffun = mean
#
#    plt.axhline(y=diffun(ml_list), linewidth = 1, color = '0.25', 
#             linestyle = "--", label=median(ml_list))
#             
#    plt.annotate("{0:.2f}".format(diffun(ml_list)),
#                 xy=(3, diffun(ml_list)), xytext=(3.33, diffun(ml_list)+0.1)) 
#
#    plt.axhline(y=diffun(ens_list), linewidth = 1, color = '0.25', 
#             linestyle = "--", label=diffun(ml_list))
#             
#    plt.annotate("{0:.2f}".format(diffun(ens_list)),
#                 xy=(3, diffun(ens_list)), xytext=(3-0.1, 0.1))
#
#    res = plt.boxplot(errors_by_time[:3], whis = 'range', showmeans = True)   
#    plt.xticks([1,2,3], ["ml", "ens", "best"], fontsize = 12)
#    
#    plt.xlabel("Ensemble selection approach", fontsize = 15)
#    plt.ylabel("Error", fontsize=15)     
#    plt.show()
#    plt.close()        
        
#    plt.figure(figsize=(10, 10))
#
#    plt.title("Validation set size={}".format(validate_count), fontsize = 15)
#
#    ml_line, = plt.plot(range(max_learn_count-1), ml_errors, label="Predicted") 
#    ens_line, = plt.plot(range(max_learn_count-1), ens_errors, label="Ensemble")
#    best_lb, = plt.plot(range(max_learn_count-1), best_errors, label="Best")
#        
#    plt.legend(handles=[ml_line, ens_line, best_lb], fontsize = 12)
#
#    plt.ylabel("Mean error", fontsize = 12)
#    plt.xlabel("Training set size", fontsize = 12)
#
#    plt.show()
#    plt.close()
    
