import numpy as np
import tensorflow as tf


class LogisticRegression(object):
    def __init__(self,
                 learning_rate=0.01,
                 max_iteration=2000):
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration
        self.sess = tf.InteractiveSession()
        self.w = None
        self.b = None

    def train(self, x_train, y_train, batch_size=32):
        x_train = np.array(x_train)
        y_train = np.array(y_train).reshape(-1, 1)
        print(x_train.shape)
        print(y_train.shape)
        self.x = tf.placeholder(shape=[None, x_train.shape[1]], dtype=tf.float64, name='input_data')
        self.y_ = tf.placeholder(shape=[None, 1], dtype=tf.float64, name='label')
        self.w = tf.Variable(tf.random_normal(shape=[x_train.shape[1], 1], dtype=tf.float64))
        self.b = tf.Variable(tf.constant(value=0.01, shape=[1], dtype=tf.float64))
        self.y = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.b)
        loss = -tf.reduce_sum(self.y_ * tf.log(self.y+0.000001) + (1 - self.y_) * tf.log(1 - self.y + 0.000001))
        # loss = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y, labels=self.y_))
        train = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        self.sess.run(tf.global_variables_initializer())

        for i in range(self.max_iteration):
            rnd_indices = np.random.randint(0, len(x_train), batch_size)
            x_batch = x_train[rnd_indices]
            y_batch = y_train[rnd_indices]
            self.sess.run(train, feed_dict={self.x: x_batch, self.y_: y_batch})
            if i % 100 == 0:
                obj = self.sess.run(loss, feed_dict={self.x: x_train,
                                                     self.y_: y_train})
                print(obj)

    def predict(self, x_test):
        y_predict = self.sess.run(self.y, feed_dict={self.x: x_test})
        y_predict[y_predict >= 0.5] = 1
        y_predict[y_predict < 0.5] = 0

        return y_predict


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

raw_data = load_digits(n_class=2)
X = np.array(raw_data.data)
y = np.array(raw_data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = LogisticRegression(max_iteration=2)
clf.train(np.array(X_train), np.array(y_train))
# print(X_test.shape)
y_ = clf.predict(np.array(X_test))
count = 0
for i in range(len(y_)):
    if y_test[i] == y_[i]:
        count += 1

print(count/len(y_))