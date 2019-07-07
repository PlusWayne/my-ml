import numpy as np


class perceptron(object):
    def __init__(self,
                 eta=0.0001,
                 max_iteration=10000,
                 verbose=100):
        self.eta = eta
        self.max_iteration = max_iteration
        self.verbose = verbose
        self.w = np.array([0])
        self.b = np.array([0])

    def train(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train, dtype=np.int32)
        self.w = np.random.rand(1, X_train.shape[1])
        self.b = np.random.rand()
        n = 0

        while n <= self.max_iteration:
            index = np.random.randint(0, X_train.shape[0] - 1)
            if y_train[index] * (np.dot(self.w, X_train[index]) + self.b) <= 0:
                self.w = self.w + self.eta * y_train[index] * X_train[index]
                self.b = self.b + self.eta * y_train[index]
            n += 1

            # Test over all train data
            if n % self.verbose == 0 and self.verbose != 0:
                y_predict = np.matmul(X_train, self.w.T) + self.b
                y_predict[y_predict >= 0] = 1
                y_predict[y_predict < 0] = -1
                accuarcy = np.sum(y_train == y_predict.reshape(1, -1)) / X_train.shape[0]

                print('The accuarcy is {0} at iteration {1}'.format(accuarcy, n))

        print("Training is Finished!")

    def predict(self, X_test):
        y_predict = np.matmul(X_test, self.w.T) + self.b
        y_predict[y_predict >= 0] = 1
        y_predict[y_predict < 0] = -1

        return y_predict
