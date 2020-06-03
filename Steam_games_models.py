import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    dtr = DecisionTreeRegressor(max_depth=5)
    model = BaggingRegressor(base_estimator=dtr, n_estimators=100, random_state=42).fit(X_train, y_train).predict(X_test)
    print(model.score)

def main():
    data = pd.read_csv("final.csv")
    print(data)
    d1 = pd.get_dummies(data["genres__001"])
    d2 = pd.get_dummies(data["genres__002"])
    d3 = pd.get_dummies(data["genres__003"])
    d4 = pd.get_dummies(data["genres__004"])
    d5 = pd.get_dummies(data["genres__005"])
    d6 = pd.get_dummies(data["genres__006"])

    d12 = d1.add(d2, fill_value=0)
    d123 = d12.add(d3, fill_value=0)
    d1234 = d123.add(d4, fill_value=0)
    d12345 = d1234.add(d5, fill_value=0)
    full_dummy = d12345.add(d6, fill_value=0)

    data = data.drop(columns=["genres__001", "genres__002", "genres__003", "genres__004", "genres__005", "genres__006"])
    full_data = pd.concat([data, full_dummy], axis=1)


    while True:
        pass


main()