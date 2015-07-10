# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:19:26 2015

@author: Krikunov Alexey
@email: sabacasam@yandex.ru

"""
from sklearn.linear_model import LinearRegression

from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet


from dtw import dtw
from copy import deepcopy

def get_elements(a, b):
    return [a[i] for i in b]

def dtw_measurement(predicted, actual):
    ln = min(len(predicted), len(actual))
    dist, cost, path = dtw(predicted[:ln], actual[:ln])
    return dist

def ensemble_predictions(models, coeffs):
    predictions = list()
    for models_prediction in zip(*models):
        predictions_by_time = [list(ens) for ens in zip(*models_prediction)]
        ensemble_predection = [sum([ p*c for p,c in zip(ens, coeffs)]) \
                                    for ens in predictions_by_time]
        predictions.append(ensemble_predection)
    return predictions

class EnsembleClassifierBase(object):
    def __init__(self, models, coefs, measurements, error_measurement = dtw_measurement):
        self._prediction_models = None  
        self._measurements = measurements
        self._ens_count = len(coefs)
        self._p_len = len(models[0][0])
        self._p_period = len(models[0])

        self._make_ensembles_predictions(models, coefs)
        
        self._errors_by_ens = list()
        for ensemble in self._ensembles:
            errors = list()
            for i in range(self._p_period):
                predicted = ensemble[i]
                actual = self._measurements[i*6:]
                errors.append(error_measurement(predicted, actual))
            self._errors_by_ens.append(errors)
        self.__error_by_time = list(zip(*self._errors_by_ens))
        
        self._best_ensemble_by_time = list()
        for predictions in self.__error_by_time:
            numered_predictions = [(n, p) for p, n in zip(predictions, range(len(predictions)))]
            current_level = [p[0] for p in sorted(numered_predictions, key=lambda x: x[1])]
            self._best_ensemble_by_time.append(current_level[0])  
            
    def prepare(self, p_count = 1, 
                    zero_point_shift = 0):
        """Make some precalculations""" 
        def get_x(i):
            if i > 0:
                fst = i * 6 - p_count + 1 + zero_point_shift
                fst = fst if fst>=0 else 0
                end = i * 6 + zero_point_shift
                msm = self._measurements[fst : end + 1]
            else:
                msm = [0]*(p_count-1) + [self._measurements[0]]
                
            msm = [0]*(p_count-len(msm)) + msm
            return msm
            
        self._get_x = get_x
        self._x_by_time = [get_x(i) for i in range(self._p_period)]
        
    def get_predict_to_actual_error(self, t):
        X = self._get_x(t)
        predicted_errors = [float(pm.predict(X)) for pm in self._prediction_models]
        actual_errors = self.__error_by_time[t]
        return list(zip(predicted_errors, actual_errors))
        
    def _make_ensembles_predictions(self, models, coefs):
        self._ensembles = list()
        models.append([[1] * self._p_len for n in range(self._p_period)])
        for coef in coefs:
            ensemble = ensemble_predictions(models, coef)
            self._ensembles.append(ensemble)
    
    def get_best_ensemble(self, t):
        best_ensemble = self._best_ensemble_by_time[t]
        return best_ensemble, self._errors_by_ens[best_ensemble][t]
        
    def get_biggest_ensemble(self, t):
        return self._ens_count - 1, self._errors_by_ens[-1][t]
        
    def get_ensemble(self, t, ensemble):
        return ensemble, self._errors_by_ens[ensemble][t]     
    
    def copy(self):
        return deepcopy(self)

class EnsembleClassifier(EnsembleClassifierBase):
    def __init__(self, models, coefs, measurements, error_measurement = dtw_measurement):
        super(EnsembleClassifier, self).__init__(models, coefs, measurements, error_measurement)
    
    def train(self, training_set, regression_model = LinearRegression):  
        self._prediction_models = \
            [regression_model() for n in range(self._ens_count)]
#            [SVR(kernel='rbf', C=1e6, gamma=0.3) for n in range( self._ens_count)]
                
        predictor_points = get_elements(self._x_by_time, training_set)
        for pm, errors in zip(self._prediction_models, self._errors_by_ens):
            predicate = get_elements(errors, training_set)           
            pm.fit(predictor_points, predicate)

    def predict_best_ensemble(self, t):
        X = self._get_x(t)
        predicted_errors = [pm.predict(X) for pm in self._prediction_models]
        predicted_ensemble = predicted_errors.index(min(predicted_errors))
        
        actual_error = self._errors_by_ens[predicted_ensemble][t]
        return predicted_ensemble, actual_error
        
       
      
class AssimilationEnsembleClassifier(EnsembleClassifierBase):
    def __init__(self, models, coefs, measurements, error_measurement = dtw_measurement):
        super(AssimilationEnsembleClassifier, self).__init__(models, coefs, measurements, error_measurement)    
    
    def train(self, training_set, regression_model = LinearRegression):
        self._prediction_models = \
            [regression_model() for n in range(self._ens_count)]
#            [SVR(kernel='rbf', C=1e6, gamma=0.3) for n in range( self._ens_count)]
                
        predictor_points = get_elements(self._x_by_time, training_set)
        for pm, errors, known_dist in zip(self._prediction_models,
                                          self._errors_by_ens, self._known_dist_by_ens):
            predicate = get_elements(errors, training_set)
            known_dist_predictor = get_elements(known_dist, training_set)
            predictors = [p + [d] for p, d in zip(predictor_points, known_dist_predictor)]          
            pm.fit(predictors, predicate)
            
    def predict_best_ensemble(self, t):    
        X = self._get_x(t)
        distanses = list(self.__known_dist_by_time[t])
        predicted_errors = [pm.predict(X+[d]) for pm, d in zip(self._prediction_models, distanses)]
        predicted_ensemble = predicted_errors.index(min(predicted_errors))
        
        actual_error = self._errors_by_ens[predicted_ensemble][t]
        return predicted_ensemble, actual_error
                   
    def find_nearest(self, knowledge_len, error_measurement=dtw_measurement):
        self._known_dist_by_ens = list()
        self.__known_dist_by_time = list()
        for ensemble in self._ensembles:
            distanses = list()
            for i in range(self._p_period):
                predicted = ensemble[i]
                actual = self._measurements[i * 6 : i * 6 + knowledge_len ]
                distanses.append(error_measurement(predicted, actual))
            self._known_dist_by_ens.append(distanses)
        self.__known_dist_by_time = list(zip(*self._known_dist_by_ens))  
        
    def get_nearest_ensemble(self, t):
        distanses = self.__known_dist_by_time[t]
        nearestes = [i for i,val in enumerate(distanses) if val==min(distanses)]  
        if len(nearestes) == 1:
            nearest = nearestes[0]
        else:
            nearest = max(nearestes)
        neares_errors = self._errors_by_ens[nearest]        
        
        return nearest, neares_errors[t]
        
        
class ANNEnsembleClassifier(EnsembleClassifierBase): 
    def __init__(self, models, coefs, measurements, error_measurement = dtw_measurement):
        super(ANNEnsembleClassifier, self).\
            __init__(models, coefs, measurements, error_measurement)
        self._ens_combinatinons = [[1 if x!=0 else 0 for x in ens[:-1]] for ens in coefs]
        
    def prepare(self, p_count = 1, zero_point_shift = 0):
        super(ANNEnsembleClassifier, self).prepare(p_count, zero_point_shift)
        self._build_ann(p_count) 
        self.p_count = p_count
        
    def _build_ann(self, inputNeurons = 2):
        ann = FeedForwardNetwork()        
        
        #Make layers
        inLayer = LinearLayer(inputNeurons)
        hiddenLayer = SigmoidLayer(3)        
        outLayer = SigmoidLayer(3)
        
        ann.addInputModule(inLayer)
        ann.addModule(hiddenLayer)
        ann.addOutputModule(outLayer)
        
        #make connections
        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outLayer)
        
        ann.addConnection(in_to_hidden)
        ann.addConnection(hidden_to_out)    
        
        ann.sortModules()
        
        self._ann = ann        
        
    def train(self, training_set):
        predictor_points = get_elements(self._x_by_time, training_set)
        ensembles = get_elements(self._best_ensemble_by_time, training_set)
        ens_combinations = [self._ens_combinatinons[ens] for ens in ensembles]
        
        ds = SupervisedDataSet(self.p_count, 3)
        for input_data, target in zip(predictor_points, ens_combinations):
            ds.appendLinked(input_data, target)
            
        trainer = BackpropTrainer(self._ann, ds)
        trainer.trainEpochs(100)
        
        
    def predict_best_ensemble(self, t):
        X = self._get_x(t)
        ann_output = self._ann.activate(X)
        predicted_combination = [1 if x>=0.4 else 0 for x in ann_output]
        try:
            predicted_ensemble = self._ens_combinatinons.index(predicted_combination)
        except:
            predicted_ensemble = len(self._ens_combinatinons)-1 
        actual_error = self._errors_by_ens[predicted_ensemble][t]
        return predicted_ensemble, actual_error  