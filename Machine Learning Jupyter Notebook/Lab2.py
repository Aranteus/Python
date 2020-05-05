import pandas as pd
from random import randrange
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def trainTree(X_train, y_train, max_depth=10, min_size=2):
    X_train['y'] = y_train
    root = find_root(X_train)
    split(root, max_depth, min_size, 1)
    return root


def find_root(X_train):
    best_score = 1000
    for index in range(len(X_train.columns) - 1):
        for i, row in X_train.iterrows():
            groups = group_split(index, row[index], X_train)
            information_gain = eval_entropy(groups, X_train["y"])
            if information_gain < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], information_gain, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}


def group_split(index, value, X_train):
    left, right = pd.DataFrame(), pd.DataFrame()
    for i, row in X_train.iterrows():       # Неоптимизировано
        if row[index] < value:
            left = left.append(row)
        else:
            right = right.append(row)
    return left, right


def eval_entropy(groups, y):
    entropy = list()
    length = list()
    for group in groups:
        size = len(group)
        if size == 0:
            length.append(0)
            entropy.append(0)
            continue
        p0 = [row["y"] for i, row in group.iterrows()].count(0) / float(size)
        p1 = [row["y"] for i, row in group.iterrows()].count(1) / float(size)
        entropy.append(-p0*np.log2(p0)-p1*np.log2(p1))
        length.append(size)

    information_gain = 1 - length[0]/(length[0]+length[1])*entropy[0] - length[1]/(length[0]+length[1])*entropy[1]
    return information_gain


def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])

    if not left or not right:
        node['left'] = node['right'] = end_node(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = end_node(left), end_node(right)
        return

    if len(left) <= min_size:
        node['left'] = end_node(left)
    else:
        node['left'] = find_root(left)
        split(node['left'], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node['right'] = end_node(right)
    else:
        node['right'] = find_root(right)
        split(node['right'], max_depth, min_size, depth + 1)


def end_node(X_train):
    y_train = X_train["y"]
    return max(y_train, key=y_train.count)


def predictTree(model, X_test):
    prediction = [predictNode(model, row) for i, row in X_test.iterrows()]
    return prediction


def predictNode(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predictNode(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predictNode(node['right'], row)
        else:
            return node['right']

def trainForest(X_train, y_train, max_depth=10, min_size=2, n_trees=5):
    forest = pd.DataFrame()
    for i in range(n_trees):
        X_train = choose_features(X_train)
        model = trainTree(X_train, y_train, max_depth, min_size)
        forest.append(model)
    return forest


def choose_features(X_train):
    features = pd.DaraFrame()
    while len(features) < 5:
        index = randrange(len(X_train.columns))
        if index not in features:
            features.append(index)
    X_train = X_train.iloc[:, features]
    return X_train


def predictForest(model, X_test):
    for i, row in X_test.iterrows():
        predictions = [predictNode(tree, row) for tree in model]
    return max(set(predictions), key=predictions.count)


def trainGradientBoosting(X_train, y_train, n_trees=20):
    trees = []
    model = DecisionTreeRegressor.fit(X_train, y_train)
    trees.append(model)
    for i in range(n_trees):
        prediction = trees[0].predict(X_train)
        boosting = np.sum([0.1 * tree.predict(X_train) for tree in trees[1:]])
        prediction += boosting
        logistic_loss = 1 / (1 + np.exp(-prediction))
        residuals = y_train - logistic_loss
        new_model = DecisionTreeRegressor.fit(X_train, residuals)
        trees.append(new_model)
    return trees


def predictGradientBoosting(trees, X_test):
    final_prediction = pd.DataFrame()
    prediction = trees[0].predict(X_test)
    boosting = np.sum([0.1 * tree.predict(X_test) for tree in trees[1:]])
    prediction += boosting
    logistic_loss = 1 / (1 + np.exp(-prediction))
    for loss in logistic_loss:
        final_prediction.append(int(loss > 0.5))
    return final_prediction


def trainRND(X_train, y_train):
    return list(y_train)


def predictRND(model, X_test):
    length = len(X_test)
    prediction = model
    np.random.shuffle(prediction)
    prediction = np.random.choice(prediction, size=length, replace=False)
    return prediction


def makeROCCurve(y_test, preds, positive_label=1):
    tp = fp = tn = fn = 0
    for truth, score in y_test, preds:
        if score > 0.5:
            if truth:
                tp += 1
            else:
                fp += 1
        else:
            if not truth:
                tn += 1
            else:
                fn += 1
    accuracy = (tp + tn) / (fp + tn + tp + fn)
    false_positive_rate = fp / (fp + tn) if (fp + tn) != 0 else 0
    true_positive_rate = tp / (tp + fn) if (tp + fn) != 0 else 0
    plt.plot(false_positive_rate, true_positive_rate)
    plt.show()


def main():
    data = pd.read_csv("train.csv")
    data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Embarked"])
    data.Sex = data.Sex.replace({"male": 0, "female": 1})
    data = data.dropna()
    print(data.describe())

    index = int(len(data)/100*80)
    train = data[0:index]
    test = data[index:]
    y_train = train.pop("Survived")
    X_train = train
    y_test = test.pop("Survived")
    X_test = test

    random_model = trainRND(X_train, y_train)
    predRandom = predictRND(random_model, X_test)
    print(predRandom.shape)
    makeROCCurve(y_test, predRandom)

    tree_model = trainTree(X_train, y_train)
    predTree = predictTree(tree_model, X_test)
    print(predTree.shape)
    makeROCCurve(y_test, predTree)

    forest_model = trainForest(X_train, y_train)
    predForest = predictForest(forest_model, X_test)
    print(predForest.shape)
    makeROCCurve(y_test, predForest)

    grad_model = trainGradientBoosting(X_train, y_train)
    predGradientBoosting = predictGradientBoosting(grad_model, X_test)
    print(predGradientBoosting.shape)
    makeROCCurve(y_test, predGradientBoosting)


main()
