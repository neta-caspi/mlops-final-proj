import pandas
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
import logging
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import plot_importance
import numpy as np
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt

logging.basicConfig(
    format=('[%(levelname)s] %(message)s'),
    level=logging.INFO
)

pd.set_option('display.max_columns', None)

def plot_feature_importance(clf, rfecv_support, X_train, y_train):
    X_train_drop_rfecv_features = X_train.loc[:, np.array(rfecv_support)]
    clf.fit(X_train_drop_rfecv_features, y_train.values.ravel())
    plt.rcParams['figure.figsize'] = [20, 20]
    plot_importance(clf)


class Pipeline:

    def __init__(self, name: str, data: pandas.DataFrame, client: str, seed=0):
        self.name = name
        self.data = data
        self.client = client
        self.seed = seed
        self.X, self.y = self.split_features_targets()
        self.categorical_feature_names = self.get_categorical_features()
        self.continuous_features_names = self.get_continuous_features()

    def get_categorical_features(self):
        return self.X.select_dtypes(include=['object']).columns.tolist()

    def get_continuous_features(self):
        return self.X.select_dtypes(exclude=['object']).columns.tolist()

    def split_features_targets(self):
        if self.client == 'bank':
            return self.split_features_targets_by_label('y')
        if self.client == 'german':
            return self.split_features_targets_by_label('classification')

    def split_features_targets_by_label(self, label_name):
        X = self.data.drop([label_name], axis=1)
        y = pd.DataFrame(self.data[label_name])
        return X, y

    def encode_target_values(self, y):
        if self.client == 'bank':
            y.replace(('yes', 'no'), (1, 0), inplace=True)
        if self.client == 'german':
            y.replace([1, 2], [1, 0], inplace=True)
        return y

    def encode_catagorical_features(self, X):
        return pd.get_dummies(X, columns=self.categorical_feature_names)

    def encode_data(self, X, y):
        logging.info(f'Encoding target values')
        y = self.encode_target_values(y)
        logging.info(f'Encoding categorical features')
        X = self.encode_catagorical_features(X)
        return X, y

    def print_classification_report(self, predictions, y_test):
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        logging.debug(f'tn: {tn}, fp {fp}, fn{fn}, tp{tp}')
        print(metrics.classification_report(y_test, predictions))

    def perform_standatization(self, X):
        X[self.continuous_features_names] = StandardScaler().fit_transform(X[self.continuous_features_names])
        return X

    def drop_correlated_features(self, X_train, X_test):
        logging.info(f'Drop correlated features 95%')

        features_count = X_train.shape[1]
        # Create correlation matrix
        corr_matrix = X_train.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        logging.info(f'Dropping features: {to_drop}')

        # Drop features
        X_train.drop(to_drop, axis=1, inplace=True)
        X_test.drop(to_drop, axis=1, inplace=True)
        features_removed = features_count - X_train.shape[1]
        logging.info(f'Dropped {features_removed} features')
        return X_train, X_test


class BankVanillaPipeline(Pipeline):

    def __init__(self, name: str, data: pandas.DataFrame):
        super().__init__(name, data, 'bank')

    def run(self):
        logging.info(f'Running pipeline: {self.name}')
        logging.info(f'Encoding data...')
        X, y = self.encode_data(self.X, self.y)
        logging.info(f'Split to train test with ratio 70%/30%')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)
        logging.info(f'Training model...')
        clf = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
        clf.fit(X_train, y_train.values.ravel())
        logging.info(f'Model done training')
        logging.info(f'Evaluating trained model...')
        predictions = clf.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        logging.debug(f'tn: {tn}, fp {fp}, fn{fn}, tp{tp}')
        print(metrics.classification_report(y_test, predictions))


class BankImprovedPipeline(Pipeline):

    def __init__(self, name: str, data: pandas.DataFrame, seed=0, balance_ratio=1):
        super().__init__(name, data, 'bank', seed)
        self.balance_ratio = balance_ratio

    def get_continuous_features(self):
        return self.X.select_dtypes(exclude=['object']).columns.tolist()

    def is_balanced(self, y):
        positive_count = y.sum().values[0]
        positive_percentage = positive_count / y.shape[0]
        return (positive_percentage < self.balance_ratio and positive_percentage > 1 - self.balance_ratio), positive_percentage

    def handle_unbalanced_data(self, X, y):
        is_data_balanced, positive_percentage = self.is_balanced(y)
        if is_data_balanced:
            logging.info(
                f'Data is balanced - no need to perform SMOTE. Positive value percentage: {positive_percentage * 100:.1f}%')
        else:
            logging.info(
                f'Data is unbalanced - running SMOTE. positive value percentage: {positive_percentage * 100:.1f}%')
            X, y = SMOTE(random_state=self.seed).fit_resample(X, y)
            _, new_positive_percentage = self.is_balanced(y)
            logging.info(
                f'SMOTE balanced data - current positive value percentage: {new_positive_percentage * 100:.1f}%')
        return X, y

    def run(self):
        logging.info(f'Running pipeline: {self.name}')
        logging.info(f'Encoding data...')
        X, y = self.encode_data(self.X, self.y)
        logging.info(f'Split to train test with ratio 70%/30%')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)
        X_train, X_test = self.drop_correlated_features(X_train, X_test)
        logging.info(f'Performing feature standardization')
        X_train = self.perform_standatization(X_train)
        logging.info(f'Handling unbalanced training data')
        X_train, y_train = self.handle_unbalanced_data(X_train, y_train)
        logging.info(f'Training model...')
        clf = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
        rfecv = RFECV(clf, step=1, cv=5, scoring='recall')
        logging.info(f'Running RFECV...')
        rfecv.fit(X_train, y_train.values.ravel())
        logging.info(f'RFECV reduced {rfecv.support_.shape[0]} to {rfecv.support_.sum()}')
        logging.info(f'Model done training')
        logging.info(f'Evaluating trained model...')
        X_test = self.perform_standatization(X_test)
        predictions = rfecv.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        logging.debug(f'tn: {tn}, fp {fp}, fn{fn}, tp{tp}')
        print(metrics.classification_report(y_test, predictions))
        # plot_feature_importance(clf, rfecv.support_, X_train, y_train)

class GermanPipeline(Pipeline):

    def __init__(self, name: str, data: pandas.DataFrame):
        super().__init__(name, data, 'german')

    def run(self):
        logging.info(f'Running pipeline: {self.name}')
        logging.info(f'Encoding data...')

        d = defaultdict(LabelEncoder)
        self.X[self.categorical_feature_names].apply(lambda x: d[x.name].fit_transform(x))

        # Encoding the variable

        X, y = self.encode_data(self.X, self.y)
        y = y.squeeze()

        logging.info(f'Split to train test with ratio 70%/30%')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        logging.info(f'Training model...')

        params = {
            'n_estimators': 3000,
            'objective': 'binary:logistic',
            'learning_rate': 0.005,
            'subsample': 0.555,
            'colsample_bytree': 0.7,
            'min_child_weight': 3,
            'max_depth': 8,
            'n_jobs': -1
        }
        eval_set = [(X_train, y_train), (X_test, y_test)]
        clf = XGBClassifier(**params).fit(X_train, y_train, eval_set=eval_set, eval_metric='auc',
                                          early_stopping_rounds=100, verbose=1000)
        clf.set_params(**{'n_estimators': clf.best_ntree_limit})
        clf.fit(X_train, y_train)

        logging.info(f'Model done training')
        logging.info(f'Evaluating trained model...')
        predictions = clf.predict(X_test, ntree_limit=clf.best_ntree_limit)
        self.print_classification_report(predictions, y_test)


class ImprovedGermanPipeline(Pipeline):

    def __init__(self, name: str, data: pandas.DataFrame, seed=0, balance_ratio=1):
        super().__init__(name, data, 'german', seed)
        self.balance_ratio = balance_ratio

    def is_balanced(self, y):
        positive_count = y.sum() #THIS IS DIFFERENT!!!!!
        positive_percentage = positive_count / y.shape[0]
        return (positive_percentage < self.balance_ratio and positive_percentage > 1 - self.balance_ratio), positive_percentage

    def handle_unbalanced_data(self, X, y):
        is_data_balanced, positive_percentage = self.is_balanced(y)
        if is_data_balanced:
            logging.info(
                f'Data is balanced - no need to perform SMOTE. Positive value percentage: {positive_percentage * 100:.1f}%')
        else:
            logging.info(
                f'Data is unbalanced - running SMOTE. positive value percentage: {positive_percentage * 100:.1f}%')
            X, y = SMOTE(random_state=self.seed).fit_resample(X, y)
            _, new_positive_percentage = self.is_balanced(y)
            logging.info(
                f'SMOTE balanced data - current positive value percentage: {new_positive_percentage * 100:.1f}%')
        return X, y

    def run(self):
        logging.info(f'Running pipeline: {self.name}')
        logging.info(f'Encoding data...')

        d = defaultdict(LabelEncoder)
        self.X[self.categorical_feature_names].apply(lambda x: d[x.name].fit_transform(x))

        # Encoding the variable

        X, y = self.encode_data(self.X, self.y)
        y = y.squeeze()

        logging.info(f'Split to train test with ratio 70%/30%')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        logging.info(f'Drop correlated features')
        X_train, X_test = self.drop_correlated_features(X_train, X_test)
        logging.info(f'Handling unbalanced training data')
        X_train, y_train = self.handle_unbalanced_data(X_train, y_train)

        logging.info(f'Training model...')

        params = {
            'n_estimators': 3000,
            'objective': 'binary:logistic',
            'learning_rate': 0.005,
            'subsample': 0.555,
            'colsample_bytree': 0.7,
            'min_child_weight': 3,
            'max_depth': 8,
            'n_jobs': -1
        }
        eval_set = [(X_train, y_train), (X_test, y_test)]
        clf = XGBClassifier(**params).fit(X_train, y_train, eval_set=eval_set, eval_metric='auc',
                                          early_stopping_rounds=100, verbose=100)

        clf.set_params(**{'n_estimators': clf.best_ntree_limit})

        rfecv = RFECV(clf, step=0.2, cv=5)

        rfecv.fit(X_train, y_train)

        logging.info(f'Model done training')
        logging.info(f'Evaluating trained model...')
        predictions = rfecv.predict(X_test)
        self.print_classification_report(predictions, y_test)
