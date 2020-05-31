import pandas as pd
import numpy as np
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import seaborn as sns

""" Ссылка на гитхаб: https://github.com/Aranteus/Python/tree/master/Machine%20Learning%20Jupyter%20Notebook """
""" Работа выполнена Якубовым Артёмом """

def load_data():
    json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource']

    raw_data = pd.read_csv("train.csv",
                           converters={column: json.loads for column in json_columns},
                           dtype={'fullVisitorId': 'str'},
                           nrows=None)

    for column in json_columns:
        data = pd.json_normalize(raw_data[column])
        data.columns = [f"{column}.{subcolumn}" for subcolumn in data.columns]
        clean_data = raw_data.drop(column, axis=1).merge(data, right_index=True, left_index=True)
    return clean_data


def transform(data):
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    data = data.replace({'device.isMobile': {True: 1, False: 0}})
    data = data.replace({'socialEngagementType': {'Engaged': 1, 'Not Socially Engaged': 0}})
    data = data.replace({'device.isMobile': {True: 1, False: 0}})
    data = data.replace({'device.browser': {'not available in demo dataset': np.nan}})
    data = data.replace({'geoNetwork.subContinent': {'not available in demo dataset': np.nan}})
    data = data.replace({'geoNetwork.region': {'not available in demo dataset':np.nan}})
    data = data.replace({'geoNetwork.networkDomain': {'not available in demo dataset': np.nan}})
    data = data.replace({'trafficSource.medium': {'not available in demo dataset': np.nan}})

    data['totals.pageviews'].fillna(0, inplace=True)
    data['totals.pageviews'] = data['totals.pageviews'].astype(float)
    data['totals.bounces'].fillna(0, inplace=True)
    data['totals.bounces'] = data['totals.bounces'].astype(float)
    data['totals.transactionRevenue'].fillna(0, inplace=True)
    data['device.browser'] = data['device.browser'].astype(float)
    data['totals.transactionRevenue'] = data['totals.transactionRevenue'].astype(float)
    data['geoNetwork.subContinent'] = data['geoNetwork.subContinent'].astype(float)
    data['geoNetwork.region'] = data['geoNetwork.region'].astype(float)
    data['geoNetwork.networkDomain'] = data['geoNetwork.networkDomain'].astype(float)
    data['trafficSource.medium'] = data['trafficSource.medium'].astype(float)

    for column in data.columns:
        if type(data[column]) == 'object' or type(data[column]) == 'str':
            data = data.drop([column], axis=1)

    data['totals.transactionRevenue'] = np.log(data['totals.transactionRevenue'].values)
    return data


def make_models(data):
    score = pd.DataFrame()
    scores = pd.DataFrame()
    y = data['totals.transactionRevenue']
    X = data.drop(columns=['totals.transactionRevenue'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_train)
    score.append(cross_val_score(gbr, X_test, y_test, cv=5))
    
    rfr = RandomForestRegressor(random_state=42)
    rfr.fit(X_train, y_train)
    score.append(cross_val_score(rfr, X_test, y_test, cv=5))
    
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    score.append(cross_val_score(svm, X_test, y_test, cv=5))
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    score.append(cross_val_score(lr, X_test, y_test, cv=5))

    models = pd.DataFrame(["GradientBoosting", "Random Forest", "SVM", "LinReg"])
    scores['Оценка'] = score
    scores["Модель"] = models

    sns.boxplot(y='Оценка', x='Модель', data=scores)


def main():
    data = load_data()
    small_data = data.sample(frac=0.1)  # 10% split
    transformed_data = transform(small_data)
    make_models(transformed_data)


main()
