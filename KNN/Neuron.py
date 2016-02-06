import numpy as np

class Neuron(object):
    """description of class"""

    def __init__(self, iw):
        self.inputWeights = iw

    def __str__(self):
        return 'Weights:' + str(self.inputWeights)

    def learn(self):
        return 0

    def evaluate(self, x):
        inputs = [1]
        inputs.extend(x)
        if len(inputs) == len(self.inputWeights):
            value = np.dot(inputs, self.inputWeights)
            if value > 1:
                return 1
            else:
                return 0
        return None
        


