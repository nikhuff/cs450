import numpy as np

class HardCodedModel:
    def predict(self, data_test):
        predictions = np.zeros(len(data_test), dtype=int)
        return predictions

class HardCodedClassifier:
    def fit(self, data_train, target_train):
        return HardCodedModel()