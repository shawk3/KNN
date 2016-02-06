import NeuralNetwork as NN


for i in range(10):
    nn = NN.NeuralNetwork(5,5)
    nn.addNewLayer(2)
    print(nn.run([1,-5,-1,1,5]))