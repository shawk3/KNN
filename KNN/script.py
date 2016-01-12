import numpy as np
from sklearn.cross_validation import train_test_split as tts
from sklearn import datasets
iris = datasets.load_iris()
from KNN import KNN
import math

ts = int (input('What test size percentage(eg. 25): '))
count = int(input('How many times would you like to run the test: '))

sum = 0
for i in range(0,count):
    xtrain, xtest, ytrain, ytest = tts(iris.data, iris.target, test_size= ts)
    knn = KNN()

    knn.train(xtrain, ytrain)
    test_result = knn.test(xtest, ytest)
    sum += test_result
    print('Result ', i, ': ' , test_result)

print('\nAverage: ', math.floor(sum / count * 100), '%')