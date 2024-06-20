import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

dataset = pd.read_csv('/home/naturalcapsule/python/Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


class Classification:
    def __init__(self) -> None:
        pass
    def Label_Encode(self, y):
        le = LabelEncoder()
        y = le.fit_transform(y)
        return y

    def auto_OneHotEncode(self, x, columns_to_apply_onehotencoding: list):
        ct = ColumnTransformer([('encoder', OneHotEncoder(), columns_to_apply_onehotencoding)], remainder = 'passthrough')
        x = ct.fit_transform(x)
        return x
    
    def auto_SimpleImputer(self, x, strategy: str, missing_values: None):
        si = SimpleImputer(strategy = strategy, missing_values = missing_values)
        x = si.fit_transform(x)
        return x
    
    def fit_LogisticRegression(self, x, y):
        classifier = LogisticRegression(n_jobs = -1, max_iter = 100000)
        classifier.fit(x, y)

        params = [{'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 'random_state': [42, 1, 0], 'penalty': ['l1', 'l2']}]
        grid_search = GridSearchCV(estimator = classifier, param_grid = params, scoring = 'accuracy', n_jobs = -1)
        grid_search.fit(x, y)
        print(grid_search.best_score_  * 100)
        return grid_search.best_params_


c = Classification()
y = c.Label_Encode(y)

x = c.auto_OneHotEncode(x, [0])

x = c.auto_SimpleImputer(x, 'mean', np.nan)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print(c.fit_LogisticRegression(x_train, y_train))