from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import hard_coded_classifier as hc
import k_nearest_neighbors as knn
import read_data as rd


def main():
    """Prompts user for data set and classifier
       Displays percent correct"""

    print("Which data would you like to use?")
    print("1 - Iris")
    print("2 - From file")
    option = int(input())

    # default to Iris with improper input
    if option == 2:
        data = rd.readFile()
    else:
        data = rd.readIris()

    data_train, data_test, target_train, target_test = \
        train_test_split(data.data, data.target, test_size=.3)

    print("Which classifier would you like to use?")
    print("1 - GaussianNB")
    print("2 - Hard Coded")
    print("3 - KNN")

    option = int(input())

    # default to hard_coded_classifier with improper input
    if option == 1:
        classifier = GaussianNB()
    elif option == 2:
        classifier = hc.HardCodedClassifier()
    else:
        classifier = knn.Classifier()

    model = classifier.fit(data_train, target_train)
    target_predicted = model.predict(data_test)


    # loop through test target and predicted target
    # if they are equal, increment number correct
    num_correct = 0
    for i in range(0, len(target_predicted)):
        if target_predicted[i] == target_test[i]:
            num_correct += 1

    # get percent correct
    percent_accurate = round((num_correct / len(target_predicted)) * 100, 2)

    print("Classifier is", percent_accurate, "% accurate.")


main()
