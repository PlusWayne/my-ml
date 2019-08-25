import tensorflow as tf


# two class fm
class FactorizationMachines(object):
    def __init__(self, feature_dim, factor_size, iteration=2000):
        self.feature_dim = feature_dim
        self.iteration = iteration
        self.w = tf.Variable(tf.random.normal(shape=(self.feature_dim, 1), dtype=tf.float64))
        self.v = tf.Variable(
            tf.random.normal(shape=(self.feature_dim, factor_size), mean=0.02, stddev=0.1, dtype=tf.float64))
        self.b = tf.Variable(tf.constant(shape=(1,), value=0.01, dtype=tf.float64))

    def train_step(self, train, label, optimizer):
        with tf.GradientTape() as tape:
            linear_part = tf.add(tf.matmul(train, self.w), self.b)
            sum_part = tf.square(tf.matmul(train, self.v))
            square_part = tf.matmul(tf.square(train), tf.square(self.v))
            second_part = 0.5 * tf.reduce_sum(tf.subtract(sum_part, square_part), 1)
            second_part = tf.reshape(second_part, (-1, 1))
            predict = linear_part + second_part
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict))
        gradients = tape.gradient(loss, [self.w, self.b, self.v])
        optimizer.apply_gradients(zip(gradients, [self.w, self.b, self.v]))

    def fit(self, train, label):
        optimizer = tf.optimizers.Adam()
        for i in range(self.iteration):
            self.train_step(train, label, optimizer)
            if i % 100 == 0:
                # t_label = tf.one_hot(label, 1)
                linear_part = tf.add(tf.matmul(train, self.w), self.b)
                sum_part = tf.square(tf.matmul(train, self.v))
                square_part = tf.matmul(tf.square(train), tf.square(self.v))
                second_part = 0.5 * tf.reduce_sum(tf.subtract(sum_part, square_part), 1)
                second_part = tf.reshape(second_part, (-1, 1))
                predict = linear_part + second_part
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict))
                tf.print(loss)

    def predict(self, train):
        linear_part = tf.add(tf.matmul(train, self.w), self.b)
        sum_part = tf.square(tf.matmul(train, self.v))
        square_part = tf.matmul(tf.square(train), tf.square(self.v))
        second_part = 0.5 * tf.reduce_sum(tf.subtract(sum_part, square_part), 1)
        second_part = tf.reshape(second_part, (-1, 1))
        predict = linear_part + second_part
        predict = tf.nn.sigmoid(predict)
        predict = tf.math.round(predict)
        return tf.reshape(predict, (-1, 1))

    def save_model(self):
        import pickle
        with open('parameters.txt', 'wb+') as file:
            pickle.dump([self.w, self.b], file)

    def load_model(self):
        import pickle
        with open('parameters.txt', 'rb') as file:
            variables = pickle.load(file)
            [self.w, self.b] = variables


# multi class fm
class MultiClassFactorizationMachines(object):
    def __init__(self, feature_dim, num_class, factor_size, iteration=2000):
        self.feature_dim = feature_dim
        self.iteration = iteration
        self.num_class = num_class
        self.w = tf.Variable(tf.random.normal(shape=(self.feature_dim, self.num_class), dtype=tf.float64))
        self.v = tf.Variable(
            tf.random.normal(shape=(self.feature_dim, factor_size), mean=0, stddev=0.1, dtype=tf.float64))
        self.b = tf.Variable(tf.constant(shape=(1,), value=0.01, dtype=tf.float64))
        self.loss = None

    def train_step(self, train, label, optimizer):
        with tf.GradientTape() as tape:
            linear_part = tf.add(tf.matmul(train, self.w), self.b)
            sum_part = tf.square(tf.matmul(train, self.v))
            square_part = tf.matmul(tf.square(train), tf.square(self.v))
            second_part = 0.5 * tf.reduce_sum(tf.subtract(sum_part, square_part), 1)
            second_part = tf.reshape(second_part, (-1, 1))
            predict = tf.add(linear_part, second_part)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=predict))
            self.loss = loss
        gradients = tape.gradient(loss, [self.w, self.b, self.v])
        optimizer.apply_gradients(zip(gradients, [self.w, self.b, self.v]))

    def fit(self, train, label):
        optimizer = tf.optimizers.Adam()
        for i in range(self.iteration):
            self.train_step(train, label, optimizer)
            # if i % 100 == 0:
            #     # t_label = tf.one_hot(label, 1)
            #     linear_part = tf.add(tf.matmul(train, self.w), self.b)
            #     sum_part = tf.square(tf.matmul(train, self.v))
            #     square_part = tf.matmul(tf.square(train), tf.square(self.v))
            #     second_part = 0.5 * tf.reduce_sum(tf.subtract(sum_part, square_part), 1)
            #     second_part = tf.reshape(second_part, (-1, 1))
            #     predict = tf.add(linear_part, second_part)
            #     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=predict))
            #     tf.print(loss)

    def predict(self, train):
        linear_part = tf.add(tf.matmul(train, self.w), self.b)
        sum_part = tf.square(tf.matmul(train, self.v))
        square_part = tf.matmul(tf.square(train), tf.square(self.v))
        second_part = 0.5 * tf.reduce_sum(tf.subtract(sum_part, square_part), 1)
        second_part = tf.reshape(second_part, (-1, 1))
        predict = tf.add(linear_part, second_part)
        predict = tf.nn.softmax(predict)
        predict = tf.argmax(predict, axis=1) + 1
        return tf.reshape(predict, (-1, 1))

    def save_model(self):
        import pickle
        with open('parameters.txt', 'wb+') as file:
            pickle.dump([self.w, self.b, self.v], file)

    def load_model(self):
        import pickle
        with open('parameters.txt', 'rb') as file:
            variables = pickle.load(file)
            [self.w, self.b, self.v] = variables


# todo sparse two class logistic regression
class SparseFactorizationMachines(object):
    def __init__(self, feature_dim, factor_size, iteration=2000):
        self.feature_dim = feature_dim
        self.iteration = iteration
        self.w = tf.Variable(tf.random.normal(shape=(self.feature_dim, 1), dtype=tf.float64))
        self.v = tf.Variable(tf.random.normal(shape=(self.feature_dim, factor_size), stddev=0.05, dtype=tf.float64))
        self.b = tf.Variable(tf.constant(shape=(1,), value=0.01, dtype=tf.float64))

    def train_step(self, sparse_ids, sparse_vals, label, optimizer):
        with tf.GradientTape() as tape:
            linear_part = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids, sp_weights=sparse_vals,
                                                        combiner='sum')
            embedding = tf.nn.embedding_lookup_sparse(params=self.v, sp_ids=sparse_ids, sp_weights=sparse_vals,
                                                      combiner='sum')
            embedding_square = tf.nn.embedding_lookup_sparse(params=tf.square(self.v), sp_ids=sparse_ids,
                                                             sp_weights=tf.square(sparse_vals), combiner='sum')
            sum_square = tf.square(embedding)
            second_part = 0.5 * tf.reduce_sum(tf.subtract(sum_square, embedding_square), 1)
            second_part = tf.reshape(second_part, (-1, 1))
            predict = tf.nn.bias_add(linear_part + second_part, self.b)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict))
            # tf.print(loss)
        gradients = tape.gradient(loss, [self.w, self.b, self.v])
        optimizer.apply_gradients(zip(gradients, [self.w, self.b, self.v]))

    def batch_sparse_data(self, train_id, train_val):
        # convert sparse batch matrix for embedding lookup
        # train_id with shape [batch_size, sparse_feature_number]
        # train_val with shape [batch_size, sparse_feature_value]
        indices, ids, values = [], [], []
        for i, (id, value) in enumerate(zip(train_id, train_val)):
            if len(id) == 0:
                indices.append((i, 0))
                ids.append(0)
                values.append(0.0)
                continue
            indices.extend([(i, t) for t in range(len(id))])
            ids.extend(id)
            values.extend(value)
        shape = (len(train_id), self.feature_dim)
        return indices, ids, values, shape

    def fit(self, train_id, train_val, label):
        indices, ids, values, shape = self.batch_sparse_data(train_id, train_val)
        sparse_ids = tf.SparseTensor(indices=indices, values=ids, dense_shape=shape)
        sparse_vals = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
        optimizer = tf.optimizers.Adadelta(learning_rate=2)
        for i in range(self.iteration):
            self.train_step(sparse_ids, sparse_vals, label, optimizer)
            if i % 100 == 0:
                z = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids, sp_weights=sparse_vals,
                                                  combiner='sum')
                predict = tf.nn.bias_add(z, self.b)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict))
                template = "The training loss at {} iteration is {}"
                tf.print(template.format(i, loss))

    def predict(self, test_id, test_val):
        indices, ids, values, shape = self.batch_sparse_data(test_id, test_val)
        sparse_ids = tf.SparseTensor(indices=indices, values=ids, dense_shape=shape)
        sparse_vals = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
        linear_part = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids, sp_weights=sparse_vals,
                                                    combiner='sum')
        embedding = tf.nn.embedding_lookup_sparse(params=self.v, sp_ids=sparse_ids, sp_weights=sparse_vals,
                                                  combiner='sum')
        embedding_square = tf.nn.embedding_lookup_sparse(params=tf.square(self.v), sp_ids=sparse_ids,
                                                         sp_weights=tf.square(sparse_vals), combiner='sum')
        sum_square = tf.square(embedding)
        second_part = 0.5 * tf.reduce_sum(tf.subtract(sum_square, embedding_square), 1)
        second_part = tf.reshape(second_part, (-1, 1))
        predict = tf.nn.bias_add(linear_part + second_part, self.b)
        predict = tf.sigmoid(predict)
        predict = tf.math.round(predict)
        return tf.reshape(predict, (-1, 1))

    def save_model(self):
        import pickle
        with open('parameters.txt', 'wb+') as file:
            pickle.dump([self.w, self.b, self.v], file)

    def load_model(self):
        import pickle
        with open('parameters.txt', 'rb') as file:
            variables = pickle.load(file)
            [self.w, self.b, self.v] = variables


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    import numpy as np
    import os
    import pandas as pd
    from sklearn.feature_extraction import DictVectorizer

    cols = ['user', 'item', 'rating', 'timestamp']

    train = pd.read_csv('ml-100k/ua.base', delimiter='\t', names=cols)
    test = pd.read_csv('ml-100k/ua.test', delimiter='\t', names=cols)
    train = train.drop(['timestamp'], axis=1)
    test = test.drop(['timestamp'], axis=1)

    # DictVectorizer会把数字识别为连续特征，这里把用户id和item id强制转为 catogorical identifier
    train["item"] = train["item"].apply(lambda x: "c" + str(x))
    train["user"] = train["user"].apply(lambda x: "u" + str(x))

    test["item"] = test["item"].apply(lambda x: "c" + str(x))
    test["user"] = test["user"].apply(lambda x: "u" + str(x))

    all_df = pd.concat([train, test])
    print("all_df head", all_df.head())

    vec = DictVectorizer()
    vec.fit_transform(all_df.to_dict(orient='record'))
    # 合并训练集与验证集，是为了one hot，用完可以释放掉
    del all_df

    x_train = vec.transform(train.to_dict(orient='record')).toarray()
    x_test = vec.transform(test.to_dict(orient='record')).toarray()

    print("x_train shape", x_train.shape)
    print("x_test shape", x_test.shape)

    y_train = train['rating'].values.reshape(-1, 1)
    y_test = test['rating'].values.reshape(-1, 1)
    print("y_train shape", y_train.shape)
    print("y_test shape", y_test.shape)
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder()
    enc.fit(y_train)
    y_train_one_hot = enc.transform(y_train).toarray()
    fm = MultiClassFactorizationMachines(feature_dim=x_train.shape[1], factor_size=1, num_class=5, iteration=1)
    for i in range(10000):
        idx = np.random.choice(np.arange(x_train.shape[0]), 256)
        fm.fit(x_train[idx], y_train_one_hot[idx])
        if i % 100 == 0 and i > 0:
            tf.print(fm.loss)
    predict = fm.predict(x_test)
    accuarcy = tf.reduce_mean(tf.cast(tf.equal(tf.reshape(predict, (-1, 1)), y_test), dtype=tf.float64))
    tf.print(accuarcy)
