import numpy as np


class Model:
    tree = {}

    def __init__(self, tree):
        self.tree = tree

    def predict(self, data_test):
        print(data_test)

    def display_tree(self, tree, indent=0):
        for key, value in tree.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                Model.display_tree(self, value, indent + 1)
            else:
                print('\t' * (indent + 1) + str(value))


class Classifier:
    """
    """
    possible_values = {}
    data_values = {}

    def set_possible_values(self, data, features):

        for i in range(0, len(features)):
            values = []
            c_data = data[:, i]

            # make list of possible values for feature
            for datapoint in c_data:
                if datapoint not in values:
                    values.append(datapoint)

            self.possible_values[features[i]] = values

    def calc_entropy(self, classes):
        total = len(classes)

        counts = np.bincount(classes)
        probs = counts[np.nonzero(counts)] / total

        return - np.sum(probs * np.log2(probs))

    def find_best_feature(self, data, classes, features):
        best_feature = features[0]
        smallest_entropy = 100

        # loop through every feature to find best info gain
        for i in range(0, len(features)):
            values = []
            feature = features[i]
            c_data = data[:, i]

            # make list of possible values for feature
            for datapoint in c_data:
                if datapoint not in values:
                    values.append(datapoint)

            class_list = []
            entropy = 0

            # loop through each value and calculate weighted entropy for choosing that feature
            for value in values:
                index = np.where(c_data == value)
                c_class_list = classes[index]
                class_list.extend(c_class_list)
                entropy += (Classifier.calc_entropy(self, c_class_list) * len(c_class_list))
            entropy = entropy / len(class_list)

            # replace best feature if entropy is smaller
            if entropy < smallest_entropy:
                best_feature = feature
                smallest_entropy = entropy

        return best_feature

    def make_tree(self, data, classes, features):
        frequency = np.bincount(classes)
        if len(frequency) != 0:
            default = np.argmax(frequency)
        else:
            default = 0

        if (classes == classes[0]).sum() == len(classes):
            return classes[0]
        elif len(data) == 0 or len(features) == 0:
            return default
        else:
            best_feature = Classifier.find_best_feature(self, data, classes, features)
            tree = {best_feature: {}}
            for value in self.possible_values[best_feature]:
                index = 0
                new_data = np.empty((0, len(features) - 1))
                new_classes = []
                bfi = features.index(best_feature)
                for datapoint in data:
                    if datapoint[bfi] == value:
                        if bfi == 0:
                            datapoint = datapoint[1:]
                            new_features = features[1:]
                        elif bfi == len(features):
                            datapoint = datapoint[:-1]
                            new_features = features[:-1]
                        else:
                            datapoint = datapoint[:bfi]
                            np.append(datapoint, datapoint[bfi + 1:])
                            new_features = features[:bfi]
                            np.append(new_features, features[bfi + 1:])
                        new_data = np.append(new_data, [datapoint], axis=0).astype(int)
                        new_classes = np.append(new_classes, classes[index]).astype(int)
                    index += 1
                subtree = Classifier.make_tree(self, new_data, new_classes, new_features)
                tree[best_feature][value] = subtree

        return tree

    def fit(self, data_train, target_train, features):
        Classifier.set_possible_values(self, data_train, features)
        tree = Classifier.make_tree(self, data_train, target_train, features)
        return Model(tree)
