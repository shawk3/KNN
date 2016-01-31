import numpy as np
from sklearn.cross_validation import train_test_split as tts
from sklearn import datasets
import math
from KNN import KNN
from DataOpener import DataOpener
from DecissionTree import DecissionTree
from sklearn.neighbors import KNeighborsClassifier


do = DataOpener()



dsetindex = int(input('To analyze test enter 1. To analyze iris, enter 2. voting enter 3, Lenses enter 4: '))
ts = float (input('What test size percentage(eg. 0.25): '))
count = int(input('How many times would you like to run the test: '))

if dsetindex == 1:
    testList = [["Tall","Hair", "Teeth"],["Tall","Receding", "No Teeth"],["short","No Hair", "No Teeth"],["short", "Hair", "Teeth"]]

    target = ["Man","old","baby", "child"]
    dts = DecissionTree(np.array(testList), target)
    
    print(dts.predict(["short", "Hair", "Teeth"]))

    dts.printTree()

if dsetindex == 2:

    iris = datasets.load_iris()

    do.set3Categories(iris.data[:,0],5.4, 6.3, "S", "M", "L")
    do.set3Categories(iris.data[:,1],2.9, 3.4, "S", "M", "L")
    do.set3Categories(iris.data[:,2],2.5, 4.9, "S", "M", "L")
    do.set3Categories(iris.data[:,3],.7, 1.7, "S", "M", "L")

    target = np.array(do.set3Categories(iris.target, 0,2,"Flower1", "Flower2", "Flower3"))


    sum = 0;
    for i in range(count):
        xtrain, xtest, ytrain, ytest = tts(iris.data, target, test_size= ts)
        #xtrain, xtest, ytrain, ytest = tts(iris.data, iris.target, test_size= .1)


        #print(xtrain[:1], ytrain)
        #print(iris.data[:,3])
        dt = DecissionTree(xtrain, ytrain)
        score = dt.test(xtest, ytest)
        sum += score
        print("Trial ", i, ": ", score)
    print("\nAverage: ", sum / count)

    #dt.printTree()
    """
Iris DATA based off of pre-analysis with excel
Column 0:  Low <= 5.4;   Med;    High >= 6.3
Column 1:  Low <= 2.9;   Med ;   High >= 3.4
Column 2:  Low <= 2.5;   Med ;   High >= 4.9
Column 3:  Low <= .7;   Med ;   High >= 1.7

"""

if dsetindex == 3:
    set = do.read_file("voting.txt")
    features = np.array(set)[:,0:15]
    classes = np.array(set)[:,16]

    sum = 0;
    for i in range(count):
        xtrain, xtest, ytrain, ytest = tts(features, classes, test_size= ts)

        dt = DecissionTree(xtrain, ytrain)
        score = dt.test(xtest, ytest)
        sum += score
        print("Trial ", i, ": ", score)
    print("\nAverage: ", sum / count)

if dsetindex == 4:
    set = do.read_file("lenses.txt")
    features = np.array(set)[:,0:3]
    classes = np.array(set)[:,4]

    sum = 0;
    for i in range(count):
        xtrain, xtest, ytrain, ytest = tts(features, classes, test_size= ts)

        dt = DecissionTree(xtrain, ytrain)
        score = dt.test(xtest, ytest)
        sum += score
        print("Trial ", i, ": ", score)
    print("\nAverage: ", sum / count)

    #dt.printTree()



