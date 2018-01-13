from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np

def importIris():
    return datasets.load_iris()

def importTxt():
    pass

def importCsv():
    pass

class HardCodedModel:
    def predict(self, data_test):
        predictions = np.zeros(len(data_test), dtype=int)
        return predictions

class HardCodedClassifier:
    def fit(self, data_train, target_train):
        return HardCodedModel()

def main():
    iris = datasets.load_iris()

    data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=.3, shuffle=True)

    print("target test:", target_test)

    #classifier = GaussianNB()
    classifier = HardCodedClassifier()

    model = classifier.fit(data_train, target_train)

    target_predicted = model.predict(data_test)
    print("target predicted:", target_predicted)

    num_correct = 0
    for i in range(0, len(target_predicted)):
        if target_predicted[i] == target_test[i]:
            num_correct += 1

    percent_accurate = round((num_correct / len(target_predicted)) * 100, 2)

    print("Classifier is", percent_accurate, "% accurate.")

main()