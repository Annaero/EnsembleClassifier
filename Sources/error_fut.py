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
from scipy.stats import norm
import numpy as np

        
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
    
    total = len(hiromb)
    learn_count = 60
    validate_count = total - learn_count
    variants = 20  
    
    coefs = list(read_ens_coeffs(coeffsFile))
    classifier = EnsembleClassifier([hiromb, swan, noswan], coefs, measurements)
    
#    ml_errors = []
#    ens_errors = []
#    best_errors = []
    errors = list()
    for shift in range(0, 4):
        classifier.prepare(3+shift, shift)
        
#        ml_avg_list = []
#        ens_avg_list = []
#        best_avg_list = []
        
        ml_errors = []
        ens_errors = []
        best_errors = []
        
        shuffle = ShuffleSplit(total, variants, validate_count, learn_count)
        for (train_set, validate_set) in shuffle:
            ts = list(train_set)
            vs = list(validate_set)
            
            classifier.train(ts)
            
            ml = [classifier.predict_best_ensemble(i)[1] for i in vs]
            ml_errors+=ml
#            ml_avg = sum(ml) / len(vs)
#            ml_avg_list.append(ml_avg)
            
            ens = [classifier.get_biggest_ensemble(i)[1] for i in vs]
            ens_errors+=ens
#            ens_avg = sum(ens) / len(vs)
#            ens_avg_list.append(ens_avg)
            
            best = [classifier.get_best_ensemble(i)[1] for i in vs]
            best_errors+=best
            
        errors.append(ml_errors)
        binsc = 20
        
        plt.figure(figsize=(15,10))
    
        bins = np.histogram(ens_errors+best_errors+ml_errors, bins = binsc)[1]
            
        plt.hist(best_errors, bins, color='green' ,normed=1, alpha=0.4, histtype='bar', label="best")
        plt.hist(ens_errors, bins, color='blue', normed=1, alpha=1, histtype='step', label="ens") 
        plt.hist(ml_errors, bins, color='red', normed=1, alpha=1, histtype='step', label="ml")

        plt.show()
        plt.close()        
        
        plt.figure(figsize=(15,10))
        plt.title("Error distribution with shift {}".format(shift))
        x = np.linspace(0, 10, binsc)
        
        params = norm.fit(ens_errors)
        pdf_f = norm.pdf(x, loc=params[0],scale=params[1])
        ensL, = plt.plot(x, pdf_f, color='blue', label="ensemble")
            
        params = norm.fit(best_errors)
        pdf_f = norm.pdf(x, loc=params[0],scale=params[1])
        bestL, = plt.plot(x, pdf_f, color='green', label="best")
        
        params = norm.fit(ml_errors)
        pdf_f = norm.pdf(x, loc=params[0],scale=params[1])
        mlL, =  plt.plot(x, pdf_f, color='red', label="predicted")
        
        plt.legend(handles=[ensL, bestL, mlL])

        plt.show()
        plt.close()
        
    plt.figure(figsize=(30,30))
    plt.title("Error distributions")
    x = np.linspace(0, 10, binsc)
        
    labels = list()
    for error in errors:
        params = norm.fit(error)
        pdf_f = norm.pdf(x, loc=params[0],scale=params[1])
        mlL, =  plt.plot(x, pdf_f, label=str(errors.index(error)))
        labels.append(mlL)
        
    plt.legend(handles=labels)
   
    plt.show()
    plt.close()
 

  