# -*- coding: utf-8 -*-

from functools import reduce
from collections import defaultdict

class NBaes:
    def __init__(self):
        self._counts = defaultdict(int)
        self._classes = set() 
        
    def fit(self, X, Y):
        assert len(X)==len(Y)
        self.samples = len(X)
        for i in range(self.samples):
            self._fit_vector(X[i], Y[i])
            
    def _fit_vector(self, X, y):
        self._classes.add(y)        
        
        pairs = zip(["~"] + X[:-1], X)
        for pair in pairs:
            self._counts[(pair, y)] += 1
        self._counts[("d", y)] += 1
        self._counts[("class", y)] +=1
        self._counts["total"] += 1
        
    @property   
    def  _classProbabilities(self):
        classProbabilities = dict()
        for sclass in self._classes:
            classProbabilities[sclass] =  self._counts[("class", sclass)] /\
                                                self._counts["total"]
        return classProbabilities
        
    def predict(self, X, classProbabilities=None):
        if not classProbabilities:
            classProbabilities = self._classProbabilities
        
        probabilities = dict()
        for sclass in self._classes:
             probabilities[sclass] = self._get_seq_prob(X, sclass) * classProbabilities[sclass]    
        #return probabilities
        return max(probabilities, key=probabilities.get)
        
    def _get_class_prob(self, sclass):
        return self._counts[("class", sclass)] / self._counts["total"]

    def _get_pair_prob(self, X, sclass):
        stype = "d" if isinstance(X, tuple) else "s" 
        if (X, sclass) in self._counts:
            return self._counts[(X, sclass)] / self._counts[(stype, sclass)]
        else:
            return 1 / self._counts[(stype, sclass)]

    def _get_seq_prob(self, X, sclass):
        pairs = zip(["~"] + X[:-1], X)
        return reduce(lambda p, x: p * self._get_pair_prob(x, sclass), pairs, 1.0)        