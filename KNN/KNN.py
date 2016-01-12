class KNN(object):
    """K nearest neighbor machine learning
    Right now it does nothing"""
    def train(self, data, target):
        print("Training")

    def test(self, data, target):
        correct = 0
        false = 0
        for i,d in enumerate(data) :
            if self.predict(d) == target[i] :
                correct += 1
            else :
                false += 1
        return (correct / (correct + false))

    def predict(self, data):
        return 1






    


