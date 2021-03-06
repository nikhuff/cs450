from sklearn import datasets
import pandas as pd
import numpy as np

class data_container:
    data = np.zeros(1)
    target = np.zeros(1)

    def __init__(self, data, target):
        self.data = data
        self.target = target

def readIris():
    return datasets.load_iris()

def readFile():
    path = input("File path:")
    return datasets.load_files(path)

def read_car_data():
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

    return data_container(data, target)

def read_indian_data():
    headers = [0,1,2,3,4,5,6,7,8]

    dataset = pd.read_csv("data/indians.data", header=None, names=headers, index_col=False)
    dataset[[1, 2, 3, 4, 5]] = dataset[[1, 2, 3, 4, 5]].replace(0, np.NaN)
    dataset.dropna(inplace=True)

    data = dataset.values
    target = data[:, 8]
    data = np.delete(data, 8, 1)

    return data_container(data, target)

def read_mpg_data():
    headers = [0, 1, 2, 3, 4, 5, 6, 7]

    dataset = pd.read_csv("data/auto-mpg.data", header=None, names=headers, dtype={3: np.float},
                          index_col=False, usecols=range(8), delim_whitespace=True, na_values='?')

    data = dataset.values
    target = data[:, 0]
    data = np.delete(data, 0, 1)

    return data_container(data, target)