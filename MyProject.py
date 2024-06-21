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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

class Classification:
    def __init__(self) -> None:
        print('Be patient this will take some time!')
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
    
    def fit_LogisticRegression(self, x_train, x_test, y_train, y_test):
        classifier = LogisticRegression(n_jobs = -1, max_iter = 100000)
        classifier.fit(x_train, y_train)

        params = [{'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 'random_state': [42, 1, 0]}]
        grid_search = GridSearchCV(estimator = classifier, param_grid = params, scoring = 'accuracy', n_jobs = -1)
        grid_search.fit(x_train, y_train)

        kfold = cross_val_score(estimator = classifier, cv = 10, X = x_train, y = y_train)

        for key, value in grid_search.best_params_.items():
            print(f'The best value paramter for {key} is {value}')
        print(f'Grid Search Score: {grid_search.best_score_  * 100}%')
        y_pred = grid_search.predict(x_test)
        print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
        print(f'Accuracy Score on the test set: {accuracy_score(y_test, y_pred) * 100}%')
        print(f'mean score of KFold: {kfold.mean() * 100}%')
        print("Standard Deviation: {:.2f}%".format(kfold.std()*100))
        return ''
    
    def fit_KNeighborsClassifier(self, x_train, x_test, y_train, y_test):
        sc = StandardScaler()
        x_train, x_test = sc.fit_transform(x_train), sc.transform(x_test)
        classifier = KNeighborsClassifier(n_jobs = -1)
        classifier.fit(x_train, y_train)

        params = [{'n_neighbors': range(5, 31), 'weights': ['uniform', 'distance'], 'leaf_size': range(30, 60),
                    'p': range(1, 11)}]
        grid_search = GridSearchCV(estimator = classifier, param_grid = params, scoring = 'accuracy', n_jobs = -1)
        grid_search.fit(x_train, y_train)
        kfold = cross_val_score(estimator = classifier, cv = 10, X = x_train, y = y_train)

        for key, value in grid_search.best_params_.items():
            print(f'The best value paramter for {key} is {value}')
        print(f'Grid Search Score: {grid_search.best_score_  * 100}%')
        y_pred = grid_search.predict(x_test)
        print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
        print(f'Accuracy Score on the test set: {accuracy_score(y_test, y_pred) * 100}%')
        print(f'mean score of KFold: {kfold.mean() * 100}%')
        print("Standard Deviation: {:.2f}%".format(kfold.std()*100))
        return ''

    def fit_SVC(self, x_train, x_test, y_train, y_test):
        sc = StandardScaler()
        x_train, x_test = sc.fit_transform(x_train), sc.transform(x_test)
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(x_train, y_train)
        params = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
        grid_search = GridSearchCV(estimator = classifier, param_grid = params, scoring = 'accuracy', n_jobs = -1)
        grid_search.fit(x_train, y_train)
        kfold = cross_val_score(estimator = classifier, cv = 10, X = x_train, y = y_train)

        for key, value in grid_search.best_params_.items():
            print(f'The best value paramter for {key} is {value}')
        print(f'Grid Search Score: {grid_search.best_score_  * 100}%')
        y_pred = grid_search.predict(x_test)
        print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
        print(f'Accuracy Score on the test set: {accuracy_score(y_test, y_pred) * 100}%')
        print(f'mean score of KFold: {kfold.mean() * 100}%')
        print("Standard Deviation: {:.2f}%".format(kfold.std()*100))
        return ''
    
    def fit_DecisionTreeClassifier(self, x_train, x_test, y_train, y_test):
        classifier = DecisionTreeClassifier()
        classifier.fit(x_train, y_train)
        params = {'criterion': ['gini', 'entropy', 'log_loss'], 'random_state': [0, 1, 42]}
        grid_search = GridSearchCV(estimator = classifier, param_grid = params, scoring = 'accuracy', n_jobs = -1)
        grid_search.fit(x_train, y_train)
        kfold = cross_val_score(estimator = classifier, cv = 10, X = x_train, y = y_train)

        for key, value in grid_search.best_params_.items():
            print(f'The best value paramter for {key} is {value}')
        print(f'Grid Search Score: {grid_search.best_score_  * 100}%')
        y_pred = grid_search.predict(x_test)
        print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
        print(f'Accuracy Score on the test set: {accuracy_score(y_test, y_pred) * 100}%')
        print(f'mean score of KFold: {kfold.mean() * 100}%')
        print("Standard Deviation: {:.2f}%".format(kfold.std()*100))
        return ''

    def fit_RandomForestClassifier(self, x_train, x_test, y_train, y_test):
        classifier = RandomForestClassifier(n_jobs = -1)
        classifier.fit(x_train, y_train)
        params = {'criterion': ['gini', 'entropy', 'log_loss'], 'random_state': [0, 1, 42], 'n_estimators': range(1, 100)}
        grid_search = GridSearchCV(estimator = classifier, param_grid = params, scoring = 'accuracy', n_jobs = -1)
        grid_search.fit(x_train, y_train)
        kfold = cross_val_score(estimator = classifier, cv = 10, X = x_train, y = y_train)

        for key, value in grid_search.best_params_.items():
            print(f'The best value paramter for {key} is {value}')
        print(f'Grid Search Score: {grid_search.best_score_  * 100}%')
        y_pred = grid_search.predict(x_test)
        print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
        print(f'Accuracy Score on the test set: {accuracy_score(y_test, y_pred) * 100}%')
        print(f'mean score of KFold: {kfold.mean() * 100}%')
        print("Standard Deviation: {:.2f}%".format(kfold.std()*100))
        return ''
    
    def fit_XGBClassifier(self, x_train, x_test, y_train, y_test):
        classifier = XGBClassifier()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        kfold = cross_val_score(estimator = classifier, cv = 10, X = x_train, y = y_train)
        print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
        print(f'Accuracy Score on the test set: {accuracy_score(y_test, y_pred) * 100}%')
        print(f'mean score of KFold: {kfold.mean() * 100}%')
        print("Standard Deviation: {:.2f}%".format(kfold.std()*100))
        return ''
    
    def fit_GaussianNB(self, x_train, x_test, y_train, y_test):
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        kfold = cross_val_score(estimator = classifier, cv = 10, X = x_train, y = y_train)
        print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
        print(f'Accuracy Score on the test set: {accuracy_score(y_test, y_pred) * 100}%')
        print(f'mean score of KFold: {kfold.mean() * 100}%')
        print("Standard Deviation: {:.2f}%".format(kfold.std()*100))
        return ''

    def fit_All(self, x_train, x_test, y_train, y_test):
        print("──────────────────────────#LogisticRegression#────────────────────────────")
        f'|{self.fit_LogisticRegression(x_train, x_test, y_train, y_test)}|'
        print("──────────────────────────#KNeighborsClassifier#────────────────────────────")
        self.fit_KNeighborsClassifier(x_train, x_test, y_train, y_test)
        print("──────────────────────────#DecisionTreeClassifier#────────────────────────────")
        self.fit_DecisionTreeClassifier(x_train, x_test, y_train, y_test)
        print("──────────────────────────#GaussianNB#────────────────────────────")
        self.fit_GaussianNB(x_train, x_test, y_train, y_test)
        print("──────────────────────────#RandomForestClassifier#────────────────────────────")
        self.fit_RandomForestClassifier(x_train, x_test, y_train, y_test)
        print("──────────────────────────#SVC#────────────────────────────")
        self.fit_SVC(x_train, x_test, y_train, y_test)
        print("──────────────────────────#XGBClassifier#────────────────────────────")
        {self.fit_XGBClassifier(x_train, x_test, y_train, y_test)}
        print("─────────────────────────────────────────────────────────────────────")
        return ''
