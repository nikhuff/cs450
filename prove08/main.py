from sklearn.model_selection import train_test_split
import k_nearest_neighbors as knn
import read_data as rd
import decision_tree as dt
import neural_net as nn


def run_test(data, algorithm, test_size=0.3):

    print("Running Experiment...")
    print("Dataset shape: {}".format(data.data.shape))

    # Randomizes the order, then breaks the data into training and testing sets
    data_train, data_test, targets_train, targets_test = train_test_split(data.data, data.target, test_size=test_size)

    # Build a model using the provided algorithm
    model = algorithm.fit(data_train, targets_train)
    # model.display_tree(model.tree)

    # Use the model to make a prediction
    targets_predicted = model.predict(data_test)

    # Compute the amount we got correct
    correct = (targets_test == targets_predicted).sum()
    total = len(targets_test)
    percent = correct / total * 100

    # Compute difference if numeric computation
    total_sum = 0
    for i in range(0, len(targets_test)):
        total_sum += abs(targets_predicted[i] - targets_test[i])

    average_diff = round(total_sum / len(targets_test), 2)

    # Display result
    print("Correct: {}/{} or {:.2f}%".format(correct, total, percent))
    print("Classifier has an average difference of {} from target data.".format(average_diff))

    # Send the model back
    return model

def get_data():

    # data = rd.read_iris()
    # data = rd.read_car()
    # data = rd.read_indian()
    data = rd.read_mpg()
    # data = rd.read_lenses()
    # data = rd.read_votes()

    return data

def get_algorithm():

    # classifier = GaussianNB()
    # classifier = knn.Classifier()
    # classifier = dt.Classifier()
    classifier = nn.Classifier()

    return classifier

def main():
    """
    Prompts user for data set and classifier
    Displays percent correct
    """

    algorithm = get_algorithm()
    data = get_data()

    model = run_test(data, algorithm)


if __name__ == '__main__':
    main()
