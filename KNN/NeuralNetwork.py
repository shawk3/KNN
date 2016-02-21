import Neuron as Neuron
import numpy as np


class NeuralNetwork(object):
    """description of class"""
    def __init__(self, numofNeurons, xlen, learnRate):
        self.n = xlen
        self.neurons = {}
        self.currentLayer = 1
        self.learningRate = learnRate
        self.neurons[self.currentLayer] = self.createLayer(numofNeurons, xlen)
        

    def createLayer(self, numNeurons, numInputs):
        neurons = []
        for i in range(numNeurons):
            weights = []
            for j in range(numInputs + 1):
                weights.append(np.random.rand()*2-1)
            neuron = Neuron.Neuron(weights, self.learningRate)
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

    def updateWeights(self):
        for layer in self.neurons.values():
            for neuron in layer:
                neuron.update()

    def saveWeights(self):
        for layer in self.neurons.values():
            for neuron in layer:
                neuron.saveWeight()

    def revert(self):
        for layer in self.neurons.values():
            for neuron in layer:
                neuron.revertToBest()

    def train(self, data, targets, validata, valitarget):
        oldscore = 0
        newscore = self.test(validata, valitarget)
        bestscore = abs(newscore)
        targetMap = self.unique(targets)
        trainingscores = []
        validationscores = []
        validationscores.append(newscore)
        k = 0
        while not newscore == 1 and k < 50:
            count = 0
            for i,d in enumerate(data):
                outputs = self.run(d)
                for j,o in enumerate(outputs):
                    target = 0
                    if targets[i] == targetMap[j]:
                        target = 1
                    self.neurons[self.currentLayer][j].defineOutError(target, len(targets))
                self.assignerror(self.currentLayer - 1, len(targets))
            trainingscore = self.test(data, targets)
            trainingscores.append(trainingscore)
            #print('training: ', totalerror)
            oldscore = np.abs(newscore)
            self.updateWeights()
            newscore = self.test(validata, valitarget)
            validationscores.append(newscore)
            #print('Validation: ', newscore)
            if newscore > bestscore:
                bestscore = np.abs(newscore)
                self.saveWeights()
                k = 0
            elif oldscore >= newscore:
                k = k+1
            else:
                k = 0
        self.revert()
        return [validationscores, trainingscores]

    def assignerror(self, layer, size):
        if layer == 0:
            return
        for i, neuron in enumerate(self.neurons[layer]):
            sum = 0
            for k in self.neurons[layer + 1]:
                sum = sum + (k.inputWeights[i+1])* k.error
            neuron.defineHiddenError(sum, size)
        self.assignerror(layer-1, size)


    def predict(self, data):
        outputs = self.run(data)
        maxindex = outputs.index(max(outputs))
        return maxindex

    def unique(self, data):
        map = {}
        i = 0
        for d in data:
            if d not in map.values():
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