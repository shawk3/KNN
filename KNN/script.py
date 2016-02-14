import NeuralNetwork as NN
from sklearn import datasets
import DataOpener as DO
import numpy as np


do = DO.DataOpener()
dsetindex = int(input('To analyze diabetes enter 1. To analyze iris, enter 2: '))
count = int(input('How many times would you like to run the test: '))

if dsetindex == 2:
    iris = datasets.load_iris()
    iris.data[: , 0] = do.normalize(iris.data[:,0])
    iris.data[: , 1] = do.normalize(iris.data[:,1])
    iris.data[: , 2] = do.normalize(iris.data[:,2])
    iris.data[: , 3] = do.normalize(iris.data[:,3])

    for i in range(count):
        nn = NN.NeuralNetwork(5,4)
        nn.addNewLayer(3)
        print(nn.test(iris.data, iris.target))

if dsetindex == 1:
    data = np.array(do.read_file("indianDiabetes.txt")).astype(np.float16)
    data[: , 0] = do.normalize(data[:,0])
    data[: , 1] = do.normalize(data[:,1])
    data[: , 2] = do.normalize(data[:,2])
    data[: , 3] = do.normalize(data[:,3])
    data[: , 4] = do.normalize(data[:,4])
    data[: , 5] = do.normalize(data[:,5])
    data[: , 6] = do.normalize(data[:,6])
    data[: , 7] = do.normalize(data[:,7])

    for i in range(count):
        nn = NN.NeuralNetwork(2,8)
        #nn.addNewLayer(2)
        #print(data[0,0:8])
        print(nn.test(data[:,0:8], data[:, 8]))

    #print(data)
