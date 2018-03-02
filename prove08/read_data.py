from sklearn import datasets
from sklearn import preprocessing
import pandas as pd
import numpy as np


class DataContainer:
    def __init__(self, data, target, features):
        scaler = preprocessing.MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
        normalized_target = scaler.fit_transform([target])
        self.data = normalized_data
        self.target = target
        # self.target = normalized_target.flatten()
        self.features = features


def read_iris():
    dataset = datasets.load_iris()
    data = dataset.data
    target = dataset.target
    return DataContainer(data, target, None)


def read_car():
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    cleanup_data = {"buying":   {"low": 0, "med": 1, "high": 2, "vhigh": 3},
                    "maint":    {"low": 0, "med": 1, "high": 2, "vhigh": 3},
                    "doors":    {"2": 2, "3": 3, "4": 4, "5more": 5},
                    "persons":  {"2": 2, "4": 4, "more": 6},
                    "lug_boot": {"small": 0, "med": 1, "big": 2},
                    "safety":   {"low": 0, "med": 1, "high": 2},
                    "class":    {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}}

    car_data = pd.read_csv("data/car.data", header=None, names=headers, index_col=False)
    car_data.replace(cleanup_data, inplace=True)
    data = car_data.values
    target = data[:, 6]
    data = np.delete(data, 6, 1)

    return DataContainer(data, target, headers)


def read_indian():
    headers = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    dataset = pd.read_csv("data/indians.data", header=None, names=headers, index_col=False)
    dataset[[1, 2, 3, 4, 5]] = dataset[[1, 2, 3, 4, 5]].replace(0, np.NaN)
    dataset.dropna(inplace=True)

    data = dataset.values
    target = data[:, 8]
    data = np.delete(data, 8, 1)

    return DataContainer(data, target, None)


def read_mpg():
    headers = [0, 1, 2, 3, 4, 5, 6, 7]

    dataset = pd.read_csv("data/auto-mpg.data", header=None, names=headers, dtype=float,
                          index_col=False, usecols=range(8), delim_whitespace=True, na_values='?')

    dataset.dropna(inplace=True)

    data = dataset.values
    target = data[:, 0]
    data = np.delete(data, 0, 1)

    return DataContainer(data, target, headers)


def read_lenses():
    headers = ["age", "prescription", "astigmatic", "tear production rate", "lense type"]

    dataset = pd.read_csv("data/lenses.data", header=None, names=headers, usecols=range(1, 6), delim_whitespace=True)

    data = dataset.values
    target = data[:, 4]
    data = np.delete(data, 4, 1)
    headers.remove("lense type")

    return DataContainer(data, target, headers)


def read_votes():
    headers = ["handicapped-infants", "water-project", "budget-resolution", "physician-fee-freeze", "el-salvador-aid",
               "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras",
               "aid-to-nicaraguan-contras", "immigration", "synfuels-corporation-cutback", "education-spending",
               "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa", "party"]

    dataset = pd.read_csv("data/house-votes-84.data", header=None, names=headers, index_col=False)

    cleanup = {"y": 2,
               "n": 0,
               "?": 1,
               "democrat": 0,
               "republican": 1}
    dataset.replace(cleanup, inplace=True)

    data = dataset.values
    target = data[:, 0]
    data = np.delete(data, 0, 1)
    headers.remove("party")

    return DataContainer(data, target, headers)
