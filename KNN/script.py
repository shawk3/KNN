import NeuralNetwork as NN
from sklearn import datasets
import DataOpener as DO
import numpy as np
from sklearn.cross_validation import train_test_split as tts
import matplotlib.pyplot as plt




do = DO.DataOpener()
dsetindex = int(input('To analyze diabetes enter 1. To analyze iris, enter 2: '))
count = int(input('How many times would you like to run the test: '))
ts = float (input('What test size percentage(eg. 0.25): '))
r = float(input('What learning rate? '))


if dsetindex == 2:
    iris = datasets.load_iris()
    iris.data[: , 0] = do.normalize(iris.data[:,0])
    iris.data[: , 1] = do.normalize(iris.data[:,1])
    iris.data[: , 2] = do.normalize(iris.data[:,2])
    iris.data[: , 3] = do.normalize(iris.data[:,3])

    

    for i in range(count):
        xtrain, xtest, ytrain, ytest = tts(iris.data, iris.target, test_size= ts)
        xtrain, xvalidate, ytrain, yvalidate = tts(xtrain, ytrain, test_size= ts)
        nn = NN.NeuralNetwork(3,4,r)
        nn.addNewLayer(3)
        scores = nn.train(xtrain, ytrain, xvalidate, yvalidate)
        print('Test: ', nn.test(xtest, ytest))

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
        xtrain, xtest, ytrain, ytest = tts(data[:,0:8], data[:, 8], test_size= ts)
        xtrain, xvalidate, ytrain, yvalidate = tts(xtrain, ytrain, test_size= ts)
        nn = NN.NeuralNetwork(10,8,r)
        nn.addNewLayer(5)
        nn.addNewLayer(2)
        #print(data[0,0:8])
        scores = nn.train(xtrain, ytrain, xvalidate, yvalidate)
        print(nn.test(xtest, ytest))
    #print(data)

if dsetindex == 3:
    nn = NN.NeuralNetwork(2,2,1)
    nn.addNewLayer(2)
    scores = nn.train([[.2,-.1],[.3, .2],[.1,-.2],[-.1,.2]], ['A','B','A','B'],[[.3,-.1],[.3, 0],[.1,-.3]], ['A','B','A'] )
    print(nn.test([[-.2,.1],[.3, -.2],[.1,-.2]], ['B','A','A']))


x = plt.plot(scores[0], 'b', scores[1], 'g')
plt.ylabel('some numbers')
plt.legend(x, ['Validation', 'Training'])
plt.show()