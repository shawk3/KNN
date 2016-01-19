import numpy as np

class DataOpener(object):
    """description of class"""

    def read_file(self, name):
        data = [[]]
        target = []
        with open(name, 'r') as f:
            data[0] = f.readline().split(',')
            for i,line in enumerate(f):
                datum = line.split(',')
                data.append(datum)

        return data

    def normalize(self,x):
        x = x.astype(np.float)
        mean = np.average(x)
        std = np.std(x)
        z = (x - mean)/std
        return z

    def setValues(self, x, n, text):
        for i,t in enumerate( x):
            t = t.rstrip()          
            if t == text:
                x[i] = n
