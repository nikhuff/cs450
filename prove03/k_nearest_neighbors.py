import math
import statistics
import numpy as np


class Model:
    def __init__(self, data_train, target_train, k):
        self._data_train = data_train
        self._target_train = target_train
        self._k = k

    def __euclidean_distance(self, x, y):
        distance = 0.0
        for i in range(0, len(x)):
            distance += (x[i] - y[i]) ** 2
        distance = math.sqrt(distance)
        return distance

    def predict(self, data_test):
        distances = []
        predictions = []

        for i in range(0, len(data_test)):
            for j in range(0, len(self._data_train)):
                # print(data_test[i], self._data_train[j])
                distances.append([self.__euclidean_distance(data_test[i], self._data_train[j]),
                                  self._target_train[j]])
            knn = sorted(distances, key=lambda x: x[0])[0:int(self._k)]
            knn_class = np.copy(knn)[:, 1]
            try:
                predictions.append(statistics.mode(knn_class))
            except statistics.StatisticsError:
                predictions.append(knn_class[0])

            distances = []

        return predictions

    _data_train = []
    _target_train = []
    _k = int()


class Classifier:
    def fit(self, data_train, target_train):
        k = input("Choose a k: ")
        return Model(data_train, target_train, k)