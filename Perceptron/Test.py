from Perceptron import perceptron
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

raw_data = load_digits(n_class=2)
X = np.array(raw_data.data)
y = np.array(raw_data.target)
y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = perceptron(max_iteration=2000, verbose=1000)
clf.train(np.array(X_train), np.array(y_train))
y_ = clf.predict(X_test)

count = 0
for i in range(len(y_)):
    if y_test[i] == y_[i]:
        count += 1

print(count/len(y_))