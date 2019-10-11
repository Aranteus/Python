# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 00:22:56 2019

@author: User
"""
#creating dataset
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

#choose train/test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .50)

#creating classifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(x_train, y_train)

tree_predictions = my_classifier.predict(x_test)

#different classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
my_classifier.fit(x_train, y_train)

knear_predictions = my_classifier.predict(x_test)

#metrics
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, tree_predictions))
print(accuracy_score(y_test, knear_predictions))