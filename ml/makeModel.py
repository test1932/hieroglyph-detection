from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import sys
import numpy as np
import pickle

def main(filename1, filename2):
    data = open(filename1, "r").read().split("\n")
    targets = np.array([line[0] for line in data])
    rest = [line[3:-1] for line in data]
    lines = np.array([list(map(int, line.split(","))) for line in rest])
    model = MLPClassifier(hidden_layer_sizes=[200,200,200])
    
    X_train, X_test, y_train, y_test = train_test_split(lines, targets)
    model.fit(lines, targets)
    file = open(filename2, "wb")
    pickle.dump(model, file)
    file.close()
    print(cross_val_score(model, lines, targets))

if __name__ == '__main__':
    filename = sys.argv[1]
    writeFile = sys.argv[2]
    main(filename, writeFile)