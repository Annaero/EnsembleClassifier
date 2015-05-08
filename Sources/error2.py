# -*- coding: utf-8 -*-

import sys
import os
import os.path
import matplotlib.pyplot as plt
from enscalibration import MODELS
from regres import read_ens_coeffs
from EnsembleClassifier import EnsembleClassifier 

from sklearn.cross_validation import ShuffleSplit

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
    
    classifier = EnsembleClassifier(predictions, coefs, measurements)
    classifier.prepare(3) 

    total = len(predictions[0])
    max_learn_count = 100
    validate_count = total - max_learn_count
    variants = 20

    ml_errors = []
    ens_errors = []
    best_errors = []
    for learn_count in range(1, max_learn_count):
#        print("Learn count {}".format(learn_count))
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
        
        
        plt.plot(range(1, max_learn_count), err)
    plt.show()
    plt.close()