import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import warnings


def train_model(X, y):
    dtr = DecisionTreeRegressor(max_depth=5)
    models = [dtr, BaggingRegressor(base_estimator=dtr, random_state=42), Ridge(), LinearRegression()]
    for model in models:
        print(model)
        pred = cross_val_predict(model, X, y, cv=10)
        print("Mean predict:")
        print(pred.mean())
        for score in ["r2", "neg_mean_squared_error", "neg_root_mean_squared_error"]:
            print(score + ":")
            scores = cross_val_score(model, X, y, scoring=score, cv=10)
            print(scores.mean())
            print()
        #boxplot по pred
        #boxplot по scores



def main():
    warnings.filterwarnings("ignore")
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

    X = full_data.copy()
    X = X.drop(columns= ['recommendations', 'name'])
    y = pd.DataFrame(full_data[['recommendations']])

    train_model(X, y)

    while True:
        pass


main()