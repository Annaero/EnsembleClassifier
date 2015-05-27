# -*- coding: utf-8 -*-
from abc import ABCMeta


#class SelectionStrategy(__metaclass__ = ABCMeta):
#    def __init__(self, classifier, threshold):
#        self.__classifier = classifier
#        self.__threshold = threshold
#        self.__history = list() #(selected, best, error_delta)
#
##    @abstractmethod
#    def get_next_ensemble(self, t):
#        return NotImplemented

class NoneStrategy():
    def __init__(self, classifier, threshold=None):
        self.__classifier = classifier
        self.__history = list() #(selected, best, error_delta )   
    
    def retrain_classifier(self, new_trainning_set, with_knowledge=False):
        if not with_knowledge:
            self.__classifier.train(new_trainning_set)
        else:
            self.__classifier.train_with_data(new_trainning_set)
    
    def get_next_ensemble(self, t, with_knowledge=False):
        if not with_knowledge:
            ml, mlErr = self.__classifier.predict_best_ensemble(t)
        else:
            ml, mlErr = self.__classifier.predict_best_with_data
        best, bestErr = self.__classifier.get_best_ensemble(t)
        ens, ensErr = self.__classifier.get_biggest_ensemble(t)
    
        return bestErr, ensErr, mlErr, None    
            
class SimpleSelectionStrategy():
    def __init__(self, classifier, threshold):
        self.__classifier = classifier
        self.__threshold = threshold
        self.__history = list() #(selected, best, error_delta    
    
    def retrain_classifier(self, new_trainning_set, with_knowledge=False):
        if not with_knowledge:
            self.__classifier.train(new_trainning_set)
        else:
            self.__classifier.train_with_data(new_trainning_set)   
    
    def get_next_ensemble(self, t, with_knowledge=False):
        if not with_knowledge:
            ml, mlErr = self.__classifier.predict_best_ensemble(t)
        else:
            ml, mlErr = self.__classifier.predict_best_with_data
        best, bestErr = self.__classifier.get_best_ensemble(t)
        ens, ensErr = self.__classifier.get_biggest_ensemble(t)
        
        if self.__history and self.__history[-1][2] < self.__threshold:
            selected = self.__history[-1][0]
        else:
            selected = ml
        
        _, resultErr = self.__classifier.get_ensemble(t, selected)
            
        self.__history.append((selected, best, resultErr - bestErr))
        return bestErr, ensErr, mlErr, resultErr
           