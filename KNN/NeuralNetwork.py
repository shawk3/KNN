import Neuron as Neuron
import numpy as np

class NeuralNetwork(object):
    """description of class"""
    def __init__(self, numofNeurons, xlen):
        self.n = xlen
        self.neurons = {}
        self.currentLayer = 1
        self.neurons[self.currentLayer] = self.createLayer(numofNeurons, xlen)

    def createLayer(self, numNeurons, numInputs):
        neurons = []
        for i in range(numNeurons):
            weights = []
            for j in range(numInputs + 1):
                weights.append(np.random.rand())
            neuron = Neuron.Neuron(weights)
            neurons.append(neuron)
        return neurons

    def addNewLayer(self, numNeurons):
        n = len(self.neurons[self.currentLayer])
        self.currentLayer = self.currentLayer + 1
        self.neurons[self.currentLayer] =  self.createLayer(numNeurons, n)

    def run(self, x):
        inputs = list(x)
        for i in range(1,self.currentLayer+1):
            inputs = self.runLayer(i,inputs)
        return inputs

    def runLayer(self, layer, x):
        outputs = []
        for n in (self.neurons[layer]):
            outputs.append(n.evaluate(x))
        return outputs

    def train(self, data):
        return 0

    def predict(self, data):
        outputs = self.run(data)
        maxindex = outputs.index(max(outputs))
        return maxindex

    def unique(self, data):
        map = {}
        i = 0
        for d in data:
            if d not in map:
                map[i] = d
                i = i + 1
        return map

    def test(self, data, targets):
        targetMap = self.unique(targets)
        count = 0
        for i,d in enumerate(data):
            prediction = self.predict(d)
            if targets[i] == targetMap[prediction]:
                count = count + 1
        return count / len(data)

