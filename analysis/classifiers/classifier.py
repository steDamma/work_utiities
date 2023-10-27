import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

from classifiers.random_forest import RandomForest
from classifiers.svm import SVM
from classifiers.logistic_regression import LR
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


class Classifier:
    def __init__(self, classifier_name, verbose):
        self.classifier_name = classifier_name
        self.clf, self.params = self.get_clf()
        self.verbose = verbose

    def get_clf(self):
        if self.classifier_name == 'rf':
            classifier = RandomForest()
        elif self.classifier_name == 'svm':
            classifier = SVM()
        elif self.classifier_name == 'lr':
            classifier = LR()
        else:
            print(f'{self.classifier_name} not supported!')
            return None

        named_steps = {
            'imputer': SimpleImputer(missing_values=np.nan, strategy='mean'),
            'scaler': StandardScaler(),
            'variance_threshold': VarianceThreshold(),
            'model': classifier.clf
        }

        clf = Pipeline(list(named_steps.items()))

        return clf, classifier.params

    @staticmethod
    def balance_data_undersample(data, target_column):
        min_class_samples = data[target_column].value_counts().min()
        grouped = data.groupby(target_column)
        balanced_data = pd.DataFrame()

        for group, df in grouped:
            sampled_data = df.sample(n=min_class_samples)
            balanced_data = pd.concat([balanced_data, sampled_data])

        return balanced_data


    def train(self, train_data, test_data, n_validate, leave_one_out):
        #train_data.reset_index(drop=True, inplace=True)
        train_data = self.balance_data_undersample(train_data, 'label')

        ts = test_data.drop(['month', 'day', 'hour', 'ref_timestamp', 'idx_val_below_thr', 'metric', 'label'], axis=1)
        y_ts = test_data['label']
        #loo_val = list(data.monte_carlo_leave_1_out(train_data, n_samples=n_validate))

        directory = f'../results/{self.classifier_name}/'
        filename = f'{directory}weights_{leave_one_out}.pkl'

        '''
        if os.path.exists(filename):
            clf = joblib.load(filename)
        else:
            os.makedirs(directory, exist_ok=True)
            clf = self.model_selection(train_data)
            joblib.dump(clf, filename)
        '''
        clf = self.model_selection(train_data, n_validate)

        y_pred = clf.predict(ts)
        current_result = {'pred_labels': y_pred, 'true_labels': y_ts.values}

        return current_result

    def model_selection(self, train_data, n_validate):
        clf = GridSearchCV(
            self.clf,
            self.params,
            scoring='f1',
            cv=StratifiedKFold(n_splits=n_validate, shuffle=True, random_state=None),
            refit=True,
            verbose=self.verbose
        )
        tr = train_data.drop(['month', 'day', 'hour', 'ref_timestamp', 'idx_val_below_thr', 'metric', 'label'], axis=1)
        y_tr = train_data['label']
        clf.fit(tr, y_tr)
        return clf

    @staticmethod
    def save_model(clf, filename):
        joblib.dump(clf, f'{filename}.pkl')

    @staticmethod
    def load_model(filename):
        if os.path.exists(f'{filename}.pkl'):
            return joblib.load('best_rf_model.pkl')

