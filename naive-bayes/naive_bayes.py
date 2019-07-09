import numpy as np


class naive_bayes(object):
    def __init__(self, lambda_=0.1):
        self.lambda_ = lambda_
        self.classes_ = None
        self.class_number = None
        self.class_prior = None
        self.prior = None

    def train(self, X_train, y_train):
        self.classes_ = np.unique(y_train)
        self.classes_number = [0] * len(self.classes_)