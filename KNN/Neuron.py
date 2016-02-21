import numpy as np

class Neuron(object):
    """description of class"""

    def __init__(self, iw, learningRate):
        self.inputWeights = iw
        self.bestWeights = list(iw)
        self.lastoutput = 0
        self.changeWeights = 0
        self.error = 0
        self.learningRate = learningRate
        self.lastinputs = [0]*len(iw)

    def __str__(self):
        return 'Weights:' + str(self.inputWeights)

    def update(self):
        for i,w in enumerate(self.inputWeights):
            self.inputWeights[i] = w- self.learningRate*self.error*self.lastinputs[i]
        self.error = 0
        if self.learningRate >=.05:
            self.learningRate = self.learningRate*.97

    def saveWeight(self):
        self.bestWeights = list(self.inputWeights)

    def revertToBest(self):
        self.inputWeights = list(self.bestWeights)

    def evaluate(self, x):
        #self.update()
        inputs = [-1]
        inputs.extend(x)
        self.lastinputs = inputs
        if len(inputs) == len(self.inputWeights):
            value = np.inner(inputs, self.inputWeights)
            self.lastoutput = (1/(1 + np.exp(-value)))
            return  self.lastoutput
        return None

    def getLastOutput(self):
        return self.lastoutput

    def defineOutError(self, t, size):
        a = self.lastoutput
        self.error = self.error + a*(1-a)*(a-t) / size

    def defineHiddenError(self, sum, size):
        a = self.lastoutput
        self.error = self.error + a*(1-a)*sum / size


