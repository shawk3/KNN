import numpy as np

class Neuron(object):
    """description of class"""

    def __init__(self, iw):
        self.inputWeights = iw
        self.lastoutput = 0

    def __str__(self):
        return 'Weights:' + str(self.inputWeights)

    def learn(self):
        return 0

    def evaluate(self, x):
        inputs = [1]
        inputs.extend(x)
        if len(inputs) == len(self.inputWeights):
            value = np.inner(inputs, self.inputWeights)
            self.lastoutput = (1/(1 + np.exp(-value)))
            return  self.lastoutput
        return None
        


