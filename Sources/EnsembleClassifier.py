# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:19:26 2015

@author: Krikunov Alexey
@email: sabacasam@yandex.ru

"""

from sklearn.svm import SVR
from dtw import dtw

def get_elements(a, b):
    return [a[i] for i in b]

def dist_mesurement(predicted, actual):
    ln = min(len(predicted), len(actual))
    dist, cost, path = dtw(predicted[:ln], actual[:ln])
    return dist

def ensemble_predictions(models, coeffs):
    predictions = list()
    for models_prediction in zip(*models):
#        print(models_prediction)
        predictions_by_time = [list(ens) for ens in zip(*models_prediction)]
        ensemble_predection = [sum([ p*c for p,c in zip(ens, coeffs)]) \
                                    for ens in predictions_by_time]
        predictions.append(ensemble_predection)
    return predictions


class EnsembleClassifier(object):
    _errors_by_ens = list()
    _best_ensemble_by_time = list()
    _ensembles = list()
    
    def __init__(self, models, coefs, measurements):
        self._prediction_models = None  
        self._measurements = measurements
        self._ens_count = len(coefs)
        self._p_len = len(models[0][0])
        self._p_period = len(models[0])

#        #find ensemble having most count of non-zero elements        
#        [sum(map(lambda x: 1 if x!=0 else 0 )) for coef in coefs]
#        self._biggest_ens = 

        self._make_ensembles_predictions(models, coefs)
            
    def prepare(self, p_count = 1, distance_measure = dist_mesurement):
        """Make some precalculations"""
        
        def get_x(i):
            if i > 0:
                fst = i * 6 - p_count - 1
                end = i * 6
                msm = self._measurements[fst : end + 1]
            else:
                msm = [0]*(p_count-1) + [self._measurements[0]]
            return msm
        self._get_x = get_x
        
        self._x_by_time = [get_x(i) for i in range(self._p_period)]
        
        for ensemble in self._ensembles:
            errors = list()
            for i in range(self._p_period):
                predicted = ensemble[i]
                actual = self._measurements[i*6:]
                errors.append(dist_mesurement(predicted, actual))
            self._errors_by_ens.append(errors)
        rmse_by_time = zip(*self._errors_by_ens)
        
        for predictions in rmse_by_time:
            best = min(predictions)
            currentLevel = predictions.index(best)
            self._best_ensemble_by_time.append(currentLevel)
        
    def train(self, train_set):      
        self._prediction_models = \
            [SVR(kernel='rbf', C=1e3, gamma=0.3) for n in range( self._ens_count)]
           
        predictors = get_elements(self._x_by_time, train_set)
        for pm, errors in zip(self._prediction_models, self._errors_by_ens):
            predicate = get_elements(errors, train_set)
            pm.fit(predictors, predicate)
        
    def predict_best_ensemble(self, t):
        X = self._get_x(t)
        predicted_errors = [pm.predict(X) for pm in self._prediction_models]
        predicted_ensemble = predicted_errors.index(min(predicted_errors))
        
        actual_error = self._errors_by_ens[predicted_ensemble][t]
        return predicted_ensemble, actual_error
        
    def get_best_ensemble(self, t):
        best_ensemble = self._best_ensemble_by_time[t]
        return best_ensemble, self._errors_by_ens[best_ensemble][t]
        
    def get_biggest_ensemble(self, t):
        return self._ens_count - 1, self._errors_by_ens[-1][t]
        
    def _make_ensembles_predictions(self, models, coefs):
        models.append([[1] * self._p_len for n in range(self._p_period)])
        for coef in coefs:
            ensemble = ensemble_predictions(models, coef)
            self._ensembles.append(ensemble)
        