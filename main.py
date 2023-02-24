"""
CS6140 Project 3
@filename: main.py
@author: Haotian Shen, Qiaozhi Liu
"""
import random
from time import time
from typing import Tuple, Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from IPython.display import display
from _decimal import Decimal, getcontext
from keras.layers import Dense
from keras.models import Sequential
from pandas import DataFrame
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def read_csv(filename: str) -> DataFrame:
    """Reads the csv file into a pandas DataFrame

    :param filename: str.
    :return: a pandas DataFrame.
    """
    df = pd.read_csv(filename)
    return df


def flip_bits(num: int, length: int) -> List:
    """Flips each and every bit of a binary representation of the provided length to generate one-hot encoding.

    For example, if the input is 0000, the output will be a list containing 1000, 0100, 0010, 0001 in integer format.
    :param num: the binary representation of 0.
    :param length: indicates the length of a binary representation of 0.
    :return: a list of integers derived from the binary vector (one-hot encoding).
    """
    result = []
    for i in range(length):
        flipped = int(str(num), 2) ^ (1 << i)
        result.append(flipped)
    return result


def one_hot_encoding(df: DataFrame) -> Dict:
    """Transforms the categorical data using one-hot encoding.

    The categorical value will be transformed into an integer representation of its binary encoding.
    :param df: the raw un-transformed pandas DataFrame.
    :return: a dictionary that records the transformation rules.
    """
    category_dict = {}
    for i in range(len(df.columns)):
        try:
            float(df.iloc[0, i])
        except ValueError:
            cat_set = set(df.iloc[:, i])
            num_cat = len(cat_set)
            print(f"The number of categories in this feature is {num_cat}")
            print(f"Features are {cat_set}")
            cur_dict = {}
            index = 0
            binary_rep = flip_bits(0, num_cat)
            for cat in cat_set:
                cur_dict[cat] = binary_rep[index]
                index += 1
            category_dict[i] = cur_dict
    return category_dict


def clean_data(df: DataFrame) -> DataFrame:
    """Cleans the raw data and turn all columns into numerical values.

    The categorical value will be transformed using one-hot encoding.
    :param df: the raw input pandas DataFrame.
    :return: a cleaned version of pandas DataFrame.
    """
    features = df.columns
    category_dict = one_hot_encoding(df)
    num_rows = len(df)
    num_cols = len(df.columns)
    cleaned_data = []
    for i in range(num_rows):
        cur_row = []
        for j in range(num_cols):
            if j in category_dict:
                cur_row.append(category_dict[j][df.iloc[i, j]])
            else:
                cur_row.append(float(df.iloc[i, j]))
        cleaned_data.append(cur_row)
    return pd.DataFrame(cleaned_data, columns=features)


def normalize_data(df: DataFrame) -> DataFrame:
    """Normalizes the dataset on the scale of standard deviation.

    :param df: the cleaned pandas DataFrame
    :return: normalized data
    """
    A = df.values
    m = np.mean(A, axis=0)
    D = A - m
    std = np.std(D, axis=0)
    D = D / std
    D = pd.DataFrame(D, columns=df.columns)
    return D


def pca(df: DataFrame, kept_variance: float, cutoff_index: int = -1) -> Tuple[int, DataFrame]:
    """Perform PCA analysis on the input DataFrame and preserve important variance.

    :param df: the cleaned pandas DataFrame
    :param kept_variance: the amount of variance to keep, a float number.
    :param cutoff_index: the cutoff index of the eigenvectors used for data projection.
    :return: a tuple of cutoff index and projected data using reduced dimensionality.
    """
    # keep 128 digits of precision to ensure that no rounding is performed and data information is lost.
    getcontext().prec = 128

    n = len(df)
    D = np.array(df)

    # perform SVD computation to get eigenvalues, eigenvectors
    U, S, V = np.linalg.svd(D, full_matrices=False)

    # eigenvalue
    e_val = S ** 2 / (n - 1)
    print('eigenvalues are:')
    print(e_val)
    print('\n')

    # turn it into high-precision decimal for kept-variance computation to ensure accuracy.
    e_val = [Decimal(x) for x in e_val]

    # eigenvector
    print('eigenvectors are:')
    print(V)
    print('\n')

    # compute the cutoff index based on the kept percentage variance of eigen sum.
    index = 0
    if cutoff_index != -1:
        index = cutoff_index
    else:
        e_sum = sum(e_val)
        e_cutoff = Decimal(kept_variance) * e_sum
        cur_sum = Decimal(0.0)
        for e_v in e_val:
            cur_sum += e_v
            index += 1
            if cur_sum >= e_cutoff:
                break

    # project the raw data into reduced dimensionality.
    reduced_e_vec = V[:index].T
    projected_data = np.dot(D, reduced_e_vec)

    # return the result, a tuple of cutoff index and projected data.
    return index, pd.DataFrame(projected_data)


def observe_clustering(X: DataFrame) -> None:
    """Plots the data projected onto 2 dimension to observe if there is any natural clustering.

    :param X: the projected 2-D data.
    :return: None
    """
    _, projected_data = pca(X, 1, 2)
    x = projected_data.iloc[:, 0].tolist()
    y = projected_data.iloc[:, 1].tolist()
    plt.scatter(x, y)
    plt.title('Scatter Plot to Observe Natural Clustering')
    plt.xlabel('X Data')
    plt.ylabel('Y Data')
    plt.subplots_adjust(bottom=0.2, left=0.18)
    plt.savefig(f'images/task1_{plt.gca().get_title()}.png')
    plt.show()


def predict(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: DataFrame, model: Any) -> Tuple[float, float, Any, Any, Any]:
    """Predicts the probabilities of each category in y.

    If the model passed-in does not have the predict_proba attribute, we will use CalibratedClassifierCV to transform
    the model and make an estimate.
    :param X_train: training X
    :param y_train: training y
    :param X_test: testing X
    :param y_test: testing y
    :param model: the trained model. Note that this model is already trained and retrain is not required.
    :return:
    """
    t0 = time()
    y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
    t = time() - t0
    if hasattr(model, 'predict_proba'):
        y_test_prob = model.predict_proba(X_test)[:, 1]
    else:
        calibrator = CalibratedClassifierCV(model, cv='prefit')
        model = calibrator.fit(X_train, y_train)
        y_test_prob = model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_test_pred)
    print(f'The accuracy computed using {model.__class__.__name__} is {test_acc * 100}%')
    return t, test_acc, y_test_pred, y_train_pred, y_test_prob


def analyze_models(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: DataFrame) -> Any:
    """Analyzes all the models based on the passed in training and test data.

    :param X_train: training X
    :param y_train: training y
    :param X_test: testing X
    :param y_test: testing y
    :return:
    """
    # uses 10 models to make prediction, set the random_state to 42 for all so that we have reproducible results.
    CLF = {'Logistic Regression': LogisticRegression(random_state=42),
           'SGDCClassifier L2': SGDClassifier(penalty='l2', random_state=42),
           'SGDCClassifier L1': SGDClassifier(penalty='l1', random_state=42),
           'Ridge Classifier': RidgeClassifier(random_state=42),
           'Perceptron': Perceptron(random_state=42),
           'BernoulliNB': BernoulliNB(),
           'K-Nearest Neighbor': KNeighborsClassifier(),
           'Support Vector Machine': SVC(random_state=42, probability=True),
           'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
           'Random Forest Classifier': RandomForestClassifier(random_state=42)}

    # print out the available classifier for view.
    print('\nAvailable classifiers:')
    for c in CLF:
        print('- ', c)

    def train_eval_model(model_name: str) -> None:
        """Trains and evaluates the model using metrics, including accruacy, f1 score, confusion matrix, bias, variance and plot the confusion matrix.

        :param model_name: a string value that indicates the name of the classifier
        :return: None
        """
        def eval_model():
            acc_train = accuracy_score(y_train, y_train_pred)
            print(f'Training Accuracy = {acc_train}')
            print(classification_report(y_train, y_train_pred))

            acc_test = accuracy_score(y_test, y_test_pred)
            print(f'Test Accuracy = {acc_test}')
            print(classification_report(y_test, y_test_pred))

            bias = 1 - acc_train
            variance = (1 - acc_test) - bias
            print(f'Bias: {bias}')
            print(f'Variance: {variance}')

        print(model_name)
        print(X_train.shape)
        print(y_train.shape)
        model = CLF[model_name]

        t0 = time()
        model.fit(X_train, y_train)
        training_t = time() - t0
        print(f'Training time = {training_t}')

        # save the trained models
        CLF[model_name] = model
        t, test_acc, y_test_pred, y_train_pred, y_test_score = predict(X_train, y_train, X_test, y_test, model)
        print(f'Validation and Test time = {t}')
        eval_model()
        print('Generating Confusion Matrix')
        cnf_matrix = confusion_matrix(y_test, y_test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
        disp.plot()
        plt.title(f'Confusion Matrix using {model_name}')
        plt.savefig(f'images/task3_{plt.gca().get_title()}.png')

    # evaluate all models and print results to console
    for c in CLF:
        train_eval_model(c)
        print('_' * 80)
        print('_' * 80)
    return CLF


def train_neural_network(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: DataFrame, dim: int) -> None:
    """Trains a feedforward neural network.

    :param X_train: training X
    :param y_train: training y
    :param X_test: testing X
    :param y_test: training y
    :param dim: the dimensionality of data, aka, the number of features
    :return: None
    """
    model = Sequential()
    model.add(Dense(16, input_dim=dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, verbose=1, batch_size=32)

    _, acc = model.evaluate(X_test, y_test)
    print(f'ANN Accuracy: {acc * 100}%')

    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    idx_fpr_01 = np.argmin(np.abs(fpr - 0.1))
    tpr_at_fpr_01 = tpr[idx_fpr_01]
    print(f'Feedforward neural network tpr at fpr=0.1 is: {tpr_at_fpr_01}')


def plot_roc(y_test: DataFrame, y_score: Any, model_name: str) -> None:
    """Plots the receiver operating characteristics curve and also displays a mark at FPR=0.1.

    :param y_test: testing y.
    :param y_score: the probabilities of each category in the y.
    :param model_name: a string value indicate the model name in the dictionary.
    :return: None
    """
    fpr, tpr, idx = compute_roc(y_test, y_score, 0.1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc})')
    plt.plot(fpr[idx], tpr[idx], 'x', label='FPR = 0.1', color='r')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, label='Using a random classifier', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic Curve using {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'images/task3_{plt.gca().get_title()}.png')


def compute_roc(y_test: DataFrame, y_score: Any, fpr_threshold: float) -> Tuple[Any, Any, int]:
    """Computes the TPR at a specified FPR.

    :param y_test: testing y.
    :param y_score: the probabilities of each category in the y.
    :param fpr_threshold: the specified FPR.
    :return: a tuple of false positive rate, true positive rate, and the index on the x-axis of the desired TPR.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    idx = (np.abs(fpr - fpr_threshold)).argmin()
    print(f'The true positive rate at fpr={fpr_threshold} is: {tpr[idx]}')
    return fpr, tpr, idx


def main():
    # import training and test data into DataFrame
    train_df = read_csv('data/heart_train_718.csv')
    test_df = read_csv('data/heart_test_200.csv')
    print(train_df.shape, test_df.shape)

    # Task 1: Pre-processing, Data Mining, and Visualization
    # combine the training and test set to prepare for data normalization
    combined_df = pd.concat([train_df, test_df], axis=0)
    combined_df = clean_data(combined_df)
    display(combined_df.head())

    # separate the cleaned data into X and y
    X, y = combined_df.drop('HeartDisease', axis=1), combined_df['HeartDisease']
    # normalize the data
    X_normalized = normalize_data(X)
    X_train = X_normalized.iloc[:len(train_df), :]
    X_test = X_normalized.iloc[len(train_df):, :]
    y_train = y.iloc[:len(train_df)]
    y_test = y.iloc[len(train_df):]
    column_names = X_train.columns

    # feature selection
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X, y)
    importance = rfc.feature_importances_

    feature_importance = {}
    for i in range(len(importance)):
        feature_importance[X.columns[i]] = {'importance': importance[i], 'index': i}

    sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1]['importance'], reverse=True)
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(len(sorted_feature_importance)):
        print(
            f"{f + 1}. {sorted_feature_importance[f][0]} (index: {sorted_feature_importance[f][1]['index']}, "
            f"importance: {sorted_feature_importance[f][1]['importance']})")

    # visualize the normalized training data using boxplot.
    # significant signals in the independent variables.
    plt.figure()
    sns.boxplot(data=X_train, width=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.2, left=0.18)
    plt.title('Boxplot Normalized Training X Data of All Features')
    plt.savefig(f'images/task1_{plt.gca().get_title()}.png')
    plt.show()

    # explore training data independent variable relationships using heatmap
    plt.figure()
    plt.title('Heatmap of All Independent Features')
    sns.heatmap(X_train.corr())
    plt.subplots_adjust(bottom=0.2, left=0.18)
    plt.savefig(f'images/task1_{plt.gca().get_title()}.png')
    plt.show()

    # identify strongly correlated variables
    # output a table of true of false indicating relationship among pairs of variables.
    corr_matrix = X_train.corr(numeric_only=True)
    strong_correlations = np.abs(corr_matrix) > 0.5
    print('Strong correlations among independent variables.')
    print(strong_correlations)

    # observe natural clustering within the training data by projecting it to 2-D
    observe_clustering(X_train)

    # use the selected salient features to train different models
    X_train = X_train.drop(['Sex', 'FastingBS', 'RestingECG'], axis=1)
    X_test = X_test.drop(['Sex', 'FastingBS', 'RestingECG'], axis=1)
    CLF = analyze_models(X_train, y_train, X_test, y_test)

    # feature engineering for quadratic and cubic terms.
    # these are commented out because they do not improve the results.
    # print('feature engineering quadratic and cubed terms')
    # X_train['Cholesterol_squared'] = X_train['Cholesterol'].apply(lambda x: x ** 2)
    # X_test['Cholesterol_squared'] = X_test['Cholesterol'].apply(lambda x: x ** 2)
    # X_train['Cholesterol_cubic'] = X_train['Cholesterol'].apply(lambda x: x ** 3)
    # X_test['Cholesterol_cubic'] = X_test['Cholesterol'].apply(lambda x: x ** 3)

    # use feedforward neural network
    train_neural_network(X_train, y_train, X_test, y_test, 8)

    # plot ROC curve for 4 models
    selected_models = ['Support Vector Machine',
                       'Random Forest Classifier',
                       'K-Nearest Neighbor',
                       'Logistic Regression']
    print(f'\nThe selected models for ROC curve are {selected_models}')
    for c in CLF:
        if c in selected_models:
            _, test_acc, y_test_pred, y_train_pred, y_test_prob = predict(X_train, y_train, X_test, y_test, CLF[c])
            plot_roc(y_test, y_test_prob, c)

    # after ROC, decide to use RandomForestClassifier and K-Nearest Neighbor models for 3 iterations
    # K-Nearest Neighbor Classifier
    # First iteration
    print('\nFirst iteration using K-Nearest Neighbor')
    knn_first_iter = KNeighborsClassifier(n_neighbors=7, leaf_size=15)
    knn_first_iter.fit(X_train, y_train)
    _, test_acc, y_test_pred, y_train_pred, y_test_prob = predict(X_train, y_train, X_test, y_test, knn_first_iter)
    compute_roc(y_test, y_test_prob, 0.1)
    cnf_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
    disp.plot()
    plt.title('Confusion Matrix using K-Nearest Neighbor 1st iteration')
    plt.savefig(f'images/task4_{plt.gca().get_title()}.png')
    plt.show()

    # Second iteration
    print('\nSecond iteration using K-Nearest Neighbor')
    cov = LedoitWolf().fit(X_train).covariance_
    knn_second_iter = KNeighborsClassifier(n_neighbors=7, leaf_size=15, metric='mahalanobis', metric_params={'V': cov})
    knn_second_iter.fit(X_train, y_train)
    _, test_acc, y_test_pred, y_train_pred, y_test_prob = predict(X_train, y_train, X_test, y_test, knn_second_iter)
    compute_roc(y_test, y_test_prob, 0.1)
    cnf_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
    disp.plot()
    plt.title('Confusion Matrix using K-Nearest Neighbor 2nd iteration')
    plt.savefig(f'images/task4_{plt.gca().get_title()}.png')
    plt.show()

    # Third iteration
    print('\nThird iteration using K-Nearest Neighbor')
    knn_models = [
        KNeighborsClassifier(n_neighbors=5, leaf_size=15, weights='uniform'),
        KNeighborsClassifier(n_neighbors=5, leaf_size=20, weights='uniform'),
        KNeighborsClassifier(n_neighbors=7, leaf_size=15, weights='uniform')
    ]
    knn_ensemble = VotingClassifier(estimators=[('model%d' % i, model) for i, model in enumerate(knn_models)],
                                    voting='soft')
    knn_ensemble.fit(X_train, y_train)
    _, test_acc, y_test_pred, y_train_pred, y_test_prob = predict(X_train, y_train, X_test, y_test, knn_ensemble)
    compute_roc(y_test, y_test_prob, 0.1)
    cnf_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
    disp.plot()
    plt.title('Confusion Matrix using K-Nearest Neighbor 3rd iteration')
    plt.savefig(f'images/task4_{plt.gca().get_title()}.png')
    plt.show()

    # RandomForestClassifier
    # First iteration
    print('\nFirst iteration using Random Forest Classifier')
    rfc_first_iter = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
    rfc_first_iter.fit(X_train, y_train)
    _, test_acc, y_test_pred, y_train_pred, y_test_prob = predict(X_train, y_train, X_test, y_test, rfc_first_iter)
    compute_roc(y_test, y_test_prob, 0.1)
    cnf_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
    disp.plot()
    plt.title('Confusion Matrix using Random Forest 1st iteration')
    plt.savefig(f'images/task4_{plt.gca().get_title()}.png')
    plt.show()

    # Second iteration
    print('\nSecond iteration using Random Forest Classifier')
    rfc_second_iter = RandomForestClassifier(max_depth=10, n_estimators=100, min_samples_split=4, random_state=42)
    rfc_second_iter.fit(X_train, y_train)
    _, test_acc, y_test_pred, y_train_pred, y_test_prob = predict(X_train, y_train, X_test, y_test, rfc_second_iter)
    compute_roc(y_test, y_test_prob, 0.1)
    cnf_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
    disp.plot()
    plt.title('Confusion Matrix using Random Forest 2nd iteration')
    plt.savefig(f'images/task4_{plt.gca().get_title()}.png')
    plt.show()

    # Third iteration
    print('\nThird iteration using Random Forest Classifier')
    rfc_third_iter = RandomForestClassifier(max_depth=10, n_estimators=50, min_samples_split=4, max_features=None,
                                            random_state=42)
    rfc_third_iter.fit(X_train, y_train)
    _, test_acc, y_test_pred, y_train_pred, y_test_prob = predict(X_train, y_train, X_test, y_test, rfc_third_iter)
    compute_roc(y_test, y_test_prob, 0.1)
    cnf_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
    disp.plot()
    plt.title('Confusion Matrix using Random Forest 3rd iteration')
    plt.savefig(f'images/task4_{plt.gca().get_title()}.png')
    plt.show()


if __name__ == '__main__':
    # set the random seed
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
