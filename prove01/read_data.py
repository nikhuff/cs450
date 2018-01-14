from sklearn import datasets

def readIris():
    return datasets.load_iris()

def readFile():
    path = input("File path:")
    return datasets.load_files(path)
