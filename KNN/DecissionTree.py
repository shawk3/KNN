import numpy as np
from collections import defaultdict


class DecissionTree(object):
    """description of class"""

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.classes = self.getClasses(self.target)
        self.features = self.getFeatures(self.data)
        self.tree = self.createTree(data, target, self.classes, self.features)
        #print(self.tree)

    def getClasses(self, target):
        classes = []
        for i in target:
            if i not in classes:
                classes.append(i)
        return classes

    def getFeatures(self, data):
        features = []
        for i in range(len(data[0])):
            features.append(i)
        return features

    def predict(self, x):
        #print(self.traversePath(self.tree, x))
        return self.traversePath(self.tree, x)

    def traversePath(self, tree, x):
        frequency = self.getFrequencies(self.target, self.classes)
        default = max(frequency, key=frequency.get)
        if type(tree) != type(self.tree):
            return tree
        for feature in tree:
            for value in tree[feature]:
                if x[feature] == value:
                   return self.traversePath(tree[feature][value], x)
        return default

    def test(self, testData, TestTarget):
        correctCount = 0
        if len(testData) == 0:
            return 0
        for i,x in enumerate(testData):
            prediction = self.traversePath(self.tree, x)
            if prediction == TestTarget[i]:
                correctCount += 1
        return correctCount / len(testData)


    def getEntropy(self, p):
        if p != 0:
            return -p * np.log2(p)
        return 0

    def getFrequencies(self, target, classes):
        map = defaultdict(int)
        for c in target:
            map[c] += 1
        return map

    def calculateNewEntropy(self, data, target, classes, feature):
        entropy = 0
        nData = len(data)

        values = self.getClasses(data[:,feature])

        
        subEntropy = 0
        for value in values:
            subEntropy = 0
            featureCounts = 0
            dataIndex = 0
            subTarget = []
            for datapoint in data:
                if datapoint[feature] == value:
                    featureCounts += 1
                    subTarget.append(target[dataIndex])
                dataIndex += 1
            classValues = self.getClasses(subTarget)
            
            classCounts = np.zeros(len(classValues))
            classIndex = 0
            for classValue in classValues:
                classCounts[classIndex] = subTarget.count(classValue)
                classIndex += 1
            for i in range(len(classValues)):
                subEntropy += self.getEntropy(float(classCounts[i])/sum(classCounts))
            entropy += float(featureCounts)/nData*subEntropy
        return entropy

    def createTree(self, data, target, classes, featureNames):
        nData = len(data)
        nFeatures = len(featureNames)
        frequency = self.getFrequencies(target, classes)
        default = max(frequency, key=frequency.get)
        if nData == 0 or nFeatures == 0 or frequency[default] == nData:
            #print("returned default:", default)
            return default
        else:
            newEntropy = defaultdict(int)
            for feature in featureNames:
                newEntropy[feature] = self.calculateNewEntropy(data, target, classes, feature)
            bestFeature = min(newEntropy, key=newEntropy.get)
            #print("Best Feature", bestFeature)
            tree = {bestFeature:{}}
            newFeatureNames = list(featureNames)
            newFeatureNames.remove(bestFeature)
            values = self.getClasses(data[:,bestFeature])
            #print("Values:: ", values)
            for value in values:
                #print("Value:", value)
                newData = []
                newTarget = []
                for i, datapoint in enumerate(data):
                    if datapoint[bestFeature] == value:
                        newData.append(datapoint)
                        newTarget.append(target[i])
                subtree = self.createTree(np.array(newData), newTarget, classes, newFeatureNames)
                tree[bestFeature][value] = subtree
            return tree

    def printTree(self):
        tree = self.printLevel(self.tree)
        self.level(tree)
        return 0

    def level(self, tree):
        levels = []
        for i in tree:
            if type(i) != type(tree):
                levels.append(i)
            else:
                levels.append(i[0])
        print(levels)
        for i in tree:
            if type(i) == type(tree):
                self.level(i)
        return 0

    def printLevel(self, tree):
        values = []
        if type(tree) != type(self.tree):
            return tree
        for feature in tree:
            values.append(feature)
            for value in tree[feature]:
                values.append(value)
                values.append(self.printLevel(tree[feature][value]))
        return values
