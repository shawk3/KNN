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
        if len(x) == len(self.inputWeights):
            return None
        


