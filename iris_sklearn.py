#!/usr/local/bin/python3
import sys

from sklearn import datasets
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def getIrisData():
    print("Preparing data ...")
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100
    )
    print("Done\n")
    return X_train, y_train, X_test, y_test


def run(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Accuracy of using {type(clf).__name__}: {acc}\n")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = getIrisData()
    run(LogisticRegression(), X_train, y_train, X_test, y_test)
    run(SVC(), X_train, y_train, X_test, y_test)
    run(DecisionTreeClassifier(), X_train, y_train, X_test, y_test)
    run(AdaBoostClassifier(), X_train, y_train, X_test, y_test)
    run(ExtraTreesClassifier(), X_train, y_train, X_test, y_test)
    run(RandomForestClassifier(), X_train, y_train, X_test, y_test)
