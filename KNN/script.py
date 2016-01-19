import numpy as np
from sklearn.cross_validation import train_test_split as tts
from sklearn import datasets
import math
from KNN import KNN
from DataOpener import DataOpener
from sklearn.neighbors import KNeighborsClassifier




db = DataOpener()





#print(iris.data)

dsetindex = int(input('To analyze cars enter 1. To analyze iris, enter 2: '))
ts = float (input('What test size percentage(eg. 0.25): '))
count = int(input('How many times would you like to run the test: '))
k = int(input('What would you like k to be equal to: '))


if dsetindex == 2:
    iris = datasets.load_iris()
    iris.data[: , 0] = db.normalize(iris.data[:,0])
    iris.data[: , 1] = db.normalize(iris.data[:,1])
    iris.data[: , 2] = db.normalize(iris.data[:,2])
    iris.data[: , 3] = db.normalize(iris.data[:,3])

    sum = 0
    for i in range(0,count):
        xtrain, xtest, ytrain, ytest = tts(iris.data, iris.target, test_size= ts)
        knn = KNN(xtrain, ytrain)
    
        knn.setK(k)
    
        knn.train()
        test_result = knn.test(xtest, ytest)
        sum += test_result
        print('Result ', i+1, ': ' , test_result)

    print('\nAverage: ', math.floor(sum / count * 100), '%')
    
if dsetindex == 1:
    data = db.read_file("car.txt")
    np_data = np.array(data)
    db.setValues(np_data[:,0], 0, "low")
    db.setValues(np_data[:,0], 1, "med")
    db.setValues(np_data[:,0], 2, "high")
    db.setValues(np_data[:,0], 3, "vhigh")
    
    db.setValues(np_data[:,1], 0, "low")
    db.setValues(np_data[:,1], 1, "med")
    db.setValues(np_data[:,1], 2, "high")
    db.setValues(np_data[:,1], 3, "vhigh")
    
    db.setValues(np_data[:,2], 5, "5more")
    
    db.setValues(np_data[:,3], 6, "more")
    
    db.setValues(np_data[:,4], 0, "small")
    db.setValues(np_data[:,4], 1, "med")
    db.setValues(np_data[:,4], 2, "big")
        
    db.setValues(np_data[:,5], 0, "low")
    db.setValues(np_data[:,5], 1, "med")
    db.setValues(np_data[:,5], 2, "high")
    
    db.setValues(np_data[:,6], 0, "unacc")
    db.setValues(np_data[:,6], 1, "acc")
    db.setValues(np_data[:,6], 2, "good")
    db.setValues(np_data[:,6], 3, "vgood")
    
    np_data[:,0] = db.normalize(np_data[:,0])
    np_data[:,1] = db.normalize(np_data[:,1])
    np_data[:,2] = db.normalize(np_data[:,2])
    np_data[:,3] = db.normalize(np_data[:,3])
    np_data[:,4] = db.normalize(np_data[:,4])
    np_data[:,5] = db.normalize(np_data[:,5])

    sum = 0
    for i in range(0,count):
        xtrain, xtest, ytrain, ytest = tts(np_data[:,0:5], np_data[:,6], test_size= ts,)
        knn = KNN(xtrain, ytrain)

        knn.setK(k)
    
        knn.train()
        test_result = knn.test(xtest, ytest)
        sum += test_result
        print('Result ', i+1, ': ' , test_result)

    print('\nAverage: ', math.floor(sum / count * 100), '%')



if dsetindex == 3:

    iris = datasets.load_iris()
    sum = 0
    for j in range(0,count):
        xtrain, xtest, ytrain, ytest = tts(iris.data, iris.target, test_size= ts)
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(xtrain, ytrain)
        predictions = classifier.predict(xtest)
        correct = 0
        for i, p in enumerate(predictions):
            if(p == ytest[i]):
                correct += 1
        test_result = correct / ytest.size
        sum += test_result
        print('Result ', j+1, ': ' , test_result)

    print('\nAverage: ', math.floor(sum / count * 100), '%')


