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
from EnsembleClassifier import EnsembleClassifier,OMEnsembleClassifier
from EnsembleClassifier import ANNEnsembleClassifier
from regres import read_data, read_ens_coeffs 
from sklearn.cross_validation import ShuffleSplit
from sklearn.utils import shuffle

from statistics import mean, median
from SelectionStrategy import NoneStrategy

from numpy import percentile, std 
from collections import defaultdict

from enscalibration_3 import calibrate_ensemble
from sklearn.metrics import mean_squared_error

from math import sqrt
        
def rmse(actual, predicted):
    ln = min(len(predicted), len(actual))
    rmse = sqrt(mean_squared_error(actual[:ln], predicted[:ln]))
    return rmse
    
if __name__ == "__main__":
    path = sys.argv[1]
    path2 = sys.argv[2]

    MODEL = "S1"

    measurementsFile = os.path.join(path, "2011080100_measurements_{}_2623.txt".format(MODEL))
    noswanFile = os.path.join(path, "2011080100_noswan_{}_48x434.txt".format(MODEL))
    swanFile = os.path.join(path, "2011080100_swan_{}_48x434.txt".format(MODEL))
    hirombFile = os.path.join(path, "2011080100_hiromb_{}_60x434.txt".format(MODEL))
    coeffsFile = os.path.join(path2, "ens_coefs.txt")
#    coeffsFile = "../Data/NewEns/ens_coefs.txt"
    
    measurements = read_data(measurementsFile)
    noswan = read_data(noswanFile)
    swan = read_data(swanFile)
    hiromb = read_data(hirombFile)
    
    coefs = list(read_ens_coeffs(coeffsFile))
    classifier = OMEnsembleClassifier([hiromb, swan, noswan], 
                                              coefs, measurements, error_measurement=rmse)
    classifier.prepare(1)
    #classifier = EnsembleClassifier([hiromb, swan, noswan], coefs, measurements)
    
   # classifier2 = classifier.copy()
    
    total = len(hiromb)    
    
    #svm_model = lambda : SVR(kernel='rbf', C=1e6, gamma=0.3)    
        
#        learn_count = 30
    lin = []
    lin_s = []
    bst = []
    ns = []
    ptas = []
    ptas_by_ens = []
    ens_by_place = []
    best2selected_pred_dist = []
    best2selected_act_dist = []
    
    validate_count = total - 250
    variants = 40
    cross_set = list(ShuffleSplit(total, variants, validate_count, 250))
    
    max_size = 100
    ts_sizes = range(1, max_size) #250) #[30, 70, 120, 145]
    for learn_count in ts_sizes:
#        coefs = list(calibrate_ensemble([hiromb, swan, noswan],
#                                            measurements, learn_count))
#        classifier = OMEnsembleClassifier([hiromb, swan, noswan], 
#                                              coefs, measurements)
#        classifier.prepare(1)
        
        predicted = []
        spred = []
        best = []
        ens = []
        pta = []
        
        ens_places = [0] * len(coefs)
        
        best2selected_pred = []
        best2selected_act = []
        
#        for (training_set, validate_set) in cross_set:
        for t in range(max_size, total):
           #ts = training_set[:learn_count]
#            ts = list(range(t-learn_count, t-8))           
            classifier.train(t, learn_count)
            validate_set = [t]
                
            pred_ens, pred_err = classifier.predict_best_ensemble(t)
            predicted.append(pred_err)
            
            best_ens, best_err = classifier.get_best_ensemble(t)
            best.append(best_err)
            
            ens.append(classifier.get_biggest_ensemble(t)[1])
            pta.append(list(map(lambda x: abs(x[0]-x[1]), 
                                classifier.get_predict_to_actual_error(t))))

            ensemble_prediction, actual_errors = classifier.get_ens_ranged_by_prediction(t)
            ens_places[ensemble_prediction.index(best_ens)] += 1
            spred.append(actual_errors[1:])
            
            pred_pred_err, best_pred_err = classifier.get_predicted_selected_to_best_error(t)
            best2selected_pred.append(pred_pred_err - best_pred_err)
            best2selected_act.append(pred_err - best_err)
            
#            full2selected.append(pred_err-)

        lmn = mean(predicted)
        lin.append((lmn, min(predicted), max(predicted)))
        lin_s.append([mean(p) for p in zip(*spred)])
        bst.append((mean(best), min(best), max(best)))
        ns.append((mean(ens), min(ens), max(ens)))
        ptas.append((mean(map(mean, pta)), std(list(map(mean, pta)))))
        pta_by_ens = list(zip(*pta))
        ptas_by_ens.append((list(map(mean, pta_by_ens)), 
                            list(map(std, pta_by_ens)),
                            list(map(max, pta_by_ens)),
                            list(map(min, pta_by_ens))))

        ens_by_place.append(ens_places)

        best2selected_pred_dist.append(mean(best2selected_pred))
        best2selected_act_dist.append(mean(best2selected_act))
#        lin.append((lmn, lmn-std(predicted)/2, lmn+std(predicted)/2))
#        bst.append((mean(best), mean(best)-std(best)/2, mean(best)+std(best)/2))
#        ns.append((mean(ens), mean(ens)-std(ens)/2, mean(ens)+std(ens)/2))        
        
#        bst.append((mean(best), percentile(best, 25), percentile(best, 75)))
        
    [lmean, l25, l75] = list(zip(*lin))
    [bmean, b25, b75] = list(zip(*bst))
    [emean, e25, e75] = list(zip(*ns))
    
    mean_err_by_place = list(zip(*lin_s))
    
    
    ###Mean error by training size    
    plt.figure(figsize=[10,10])
    plt.title("Mean error by training size", fontsize=15)
    pline, = plt.plot(ts_sizes, lmean, "r-", label="Predicted ensemble", linewidth=2.0)
#    plt.fill_between(ts_sizes, l25, l75, facecolor="red", alpha=0.2)    
    fline, = plt.plot(ts_sizes, emean, "b-", label="Full ensemble", linewidth=2.0)
#    plt.fill_between(ts_sizes, e25, e75, facecolor="blue", alpha=0.5)    
    bline, = plt.plot(ts_sizes, bmean, "g-", label="Best ensemble", linewidth=2.0)
#    plt.fill_between(ts_sizes, b25, b75, facecolor="green", alpha=0.5) 
    
    lines = []
#    for place, i in zip(mean_err_by_place, range(2, len(mean_err_by_place))):
#        line, = plt.plot(ts_sizes, place, "-", label="Predicted ensemble on {}".format(i))
#        lines.append(line)
    
    plt.ylabel("Mean RMS error, cm", fontsize=15)
    plt.xlabel("Training set size, forecast", fontsize=15)
    plt.legend(handles = [pline, fline, bline] + lines, fontsize=15)
    plt.show()
    plt.close()

        
    ###Mean error by training size
    [pta_mean, pta_std] = list(zip(*ptas))
    plt.figure(figsize=[10,10])
    
    plt.title("Mean error prediction error by training size")
    mean_line, = plt.plot(ts_sizes, pta_mean, "b-", label="Mean")
    pta_lower = [m-s for m,s in zip(pta_mean, pta_std)]
    pta_upper = [m+s for m,s in zip(pta_mean, pta_std)]
    plt.fill_between(ts_sizes[2:], pta_lower[2:], pta_upper[2:], 
                         facecolor="blue", alpha=0.5)
    
    plt.ylabel("Mean error, sm")
    plt.xlabel("Training set size, forecast")
    plt.legend(handles=[mean_line])
    plt.show()
    plt.close()
    
    plt.figure(figsize=[14,12])
    plt.suptitle("Error prediction error by training size for each ensemble ")
    [pta_mean, pta_std, pta_max, pta_min] = zip(*ptas_by_ens)
    for mn, st, mi, mx, i in zip(zip(*pta_mean), zip(*pta_std),
                         zip(*pta_min), zip(*pta_max), range(8)):
      
        pta_lower =[m-s for m,s in zip(mn, st)]
        pta_upper =[m+s for m,s in zip(mn, st)]             
        plt.subplot(3,3,i+1)
        
        plt.title("Ensemble {}".format(i+1))
        plt.plot(ts_sizes[2:], mn[2:])
        plt.fill_between(ts_sizes[2:], pta_lower[2:], pta_upper[2:], 
                         facecolor="blue", alpha=0.5)
        plt.fill_between(ts_sizes[2:], mi[2:], mx[2:], facecolor="red", alpha=0.5)
        
        plt.ylim(0, 6)
      
    plt.show()
    plt.close()
      
    ###Fraction of predictions
    plt.figure(figsize=[10,10])
    plt.title("Fraction of predictions")
    plt.stackplot(ts_sizes, *(list(zip(*ens_by_place))), alpha=0.5)
    plt.ylabel("Count of predictions")
    plt.xlabel("Training set size, forecast")
    plt.show()
    plt.close()
    
        
    ###Mean error by training size
#    plt.figure(figsize=[10,10])
#    plt.title("Best to selected ensemble error distance")
#    pline, = plt.plot(ts_sizes, best2selected_pred_dist, label="Predicted")
#    aline, = plt.plot(ts_sizes, best2selected_act_dist, label="Actual")
#    plt.legend(handles=[pline, aline])
#    plt.ylabel("Mean error")
#    plt.xlabel("Training set size, forecast")
#    plt.show()
#    plt.close()  
    
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
    
