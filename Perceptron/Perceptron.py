class Perceptron(object):
    def __init__(self,
                 eta=0.0001,
                 max_iteration=10000,
                 verbose=False):
        self.eta = eta
        self.max_iteration = max_iteration
        self.verbose = False
        self.w =