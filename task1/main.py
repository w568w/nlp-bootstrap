import collections
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas
from numpy import ndarray

from task1.ngram import generate_ngram


def softmax(x: ndarray) -> ndarray:
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / np.sum(x, axis=-1, keepdims=True)


def feed_forward(weight: ndarray, x: ndarray):
    return softmax(x @ weight)


def predict(weight: ndarray, x: ndarray):
    return np.argmax(feed_forward(weight, x), axis=-1)


def cross_entropy(y, y_pred):
    return -y * np.log(y_pred)


def train(x: ndarray, y: ndarray, index_generator, epochs=1000, learning_rate=0.001) -> (ndarray, ndarray):
    len_x, num_feature = x.shape
    num_type = y.shape[-1]
    W = np.ones((num_feature, num_type))
    loss = []
    for epoch in range(epochs):
        id_generator = index_generator(len_x)
        grad = np.zeros((num_feature, num_type))
        batch_size = 0
        for i in id_generator:
            batch_size += 1
            type_y = np.argmax(y[i])
            # Forward feed
            result = feed_forward(W, x[i])
            # Calculate gradient
            grad += np.outer(x[i], result - y[i])
            # Estimate the loss
            loss.append(cross_entropy(y[i], result)[type_y])
        # Update weights
        W += learning_rate * (-grad) / batch_size
    return W, loss


def mini_batch(n):
    for i in range(0, 10):
        yield random.randint(0, n - 1)


def shuffle(n):
    yield random.randint(0, n - 1)


def batch(n):
    for i in range(0, n):
        yield i


def main():
    per = 800
    file = pandas.read_csv("train.tsv", sep='\t')[:1000]
    texts = file['Phrase']
    labels = file['Sentiment']

    x = generate_ngram(texts, [1, 2, 3])
    y = np.eye(5)[labels[:per]]
    W, loss = train(x[:per], y, mini_batch, epochs=10000)
    # plt.plot(range(len(loss)), loss)
    # plt.show()
    pred = predict(W, x[per:])
    print(collections.Counter(pred))
    print(collections.Counter(labels[per:]))
    error = pred - labels[per:]
    print("Error: %d/%d" % (np.count_nonzero(error), len(error)))


if __name__ == '__main__':
    main()
