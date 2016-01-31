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

    def set3Categories(self, x, l, h, v1, v2, v3):
        #not yet implemented
        y = list(x)
        for i, xi in enumerate(x):
            if xi <= l:
                y[i] = v1
                x[i] = 0
            elif xi >= h:
                y[i] = v3
                x[i] = 2
            else:
                y[i] = v2
                x[i] = 1
        return y

    def setnCategories(self, x, n):
        #x = sorted(x)
        size = int(np.floor(len(x)/n))
        remainder = len(x) % n
        extra = 0
        start = 0
        for i in range(n):
            if i >= n - remainder:
                extra += 1
                start = extra - 1
            for j in range(start, size + extra):
                x[j+size*i] = i

