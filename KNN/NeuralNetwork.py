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
            print(i)
            inputs = self.runLayer(i,inputs)
        return inputs

    def runLayer(self, layer, x):
        outputs = []
        for n in (self.neurons[layer]):
            outputs.append(n.evaluate(x))
        return outputs

    def train(self, data):
        return 0

    def test(self, data):
        run(data[0])
        return 0

