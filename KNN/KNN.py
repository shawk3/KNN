from scipy import spatial
import numpy as np
from collections import defaultdict
from operator import itemgetter



class KNN(object):
    """K nearest neighbor machine learning
    Right now it does nothing"""

    def __init__(self, data, target):
        self.k = 2
        self.data = data
        self.data = self.data.astype(np.float)
        self.target = target
        self.trained = False
        self.tree = None
        


    def train(self):
        print("Training")
        self.trained = True
        self.kdTree()

    

    def test(self, data, target):
        data = data.astype(np.float)
        if(not self.trained):
            self.train()
        correct = 0
        false = 0
        for i,d in enumerate(data) :
            x = self.predict(d)
            if self.predict(d) == target[i] :
                correct += 1
            else :
                false += 1
        return (correct / (correct + false))


    def predict(self, datum):
        if(not self.trained):
            self.train()
        map = defaultdict(int)
        nearestNeighbors = self.tree.query(datum, self.k)
        if self.k > 1:
            for i in range(0,self.k):
                map[self.target[nearestNeighbors[1][i]]] += 1
            return self.getMostCommonOccurence(map)
        return self.target[nearestNeighbors[1]]
        

    def setK(self, neighborCount):
        self.k = neighborCount

    def kdTree(self):
        self.tree = spatial.KDTree(self.data)

    
    def getMostCommonOccurence(self, map):
        value = 0
        occurence = 0
        for k in map.keys():
            if map[k] > occurence:
                value = k
                occurence = map[k]
        return value

        
        






        






    


