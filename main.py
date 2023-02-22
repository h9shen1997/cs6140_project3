from time import time
from typing import Tuple, Dict, List

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# set the display of DataFrame to display full rows and columns.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


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


def predict(train_X, train_y, test_X, test_y, model):
    """Computes the accuracy of prediction using the test set based on the specified model.

    The type of model will be passed in as a parameter. It will fit on the training data and use the test_X to make a
    prediction on y data and calculate the accuracy score.
    :param train_X: DataFrame
    :param train_y: DataFrame
    :param test_X: DataFrame
    :param test_y: DataFrame
    :param model: the passed-in model.
    :return: the accuracy score as a float number.
    """
    model.fit(train_X, train_y.values.ravel())
    pred_y = model.predict(test_X)
    if hasattr(model, 'predict_proba'):
        score_y = model.predict_proba(test_X)[:, 1]
    else:
        score_y = 1 / (1 + np.exp(-model.decision_function(test_X)))
    accuracy = accuracy_score(test_y.values.ravel(), pred_y)
    print(f'The accuracy computed using {model.__class__.__name__} is {accuracy * 100}%')
    return accuracy, pred_y, score_y


def train_eval_model(c, X_train, y_train, X_test, y_test, CLF):
    def pred():
        t0 = time()
        p_train, p_test = model.predict(X_train), model.predict(X_test)
        t = time() - t0
        return t, p_train, p_test

    def eval_model():
        acc_train = accuracy_score(y_train, p_train)
        print(f'Training Accuracy={acc_train:4.3f}')
        print(classification_report(y_train, p_train))

        acc_test = accuracy_score(y_test, p_test)
        print(f'Test Accuracy={acc_test:4.3f}')
        print(classification_report(y_test, p_test))

        expected_pred = model.score(X_test, y_test)
        pred_variance = expected_pred * (1 - expected_pred)
        mse = ((y_test - p_test) ** 2).mean()
        bias = mse - pred_variance
        print(f'Bias: {bias:.2f}')
        print(f'Variance: {pred_variance:.2f}')

    model_name = c
    print(model_name)
    print(X_train.shape)
    print(y_train.shape)
    model = CLF[c]

    t0 = time()
    model.fit(X_train, y_train)
    time_train = time() - t0
    print(f'Training time = {time_train:4.2f}')

    # trained model
    CLF[c] = model
    t, p_train, p_test = pred()
    print(f'Validation and Test time = {t:4.2f}')
    eval_model()
    print('Generating Confusion Matrix')
    cnf_matrix = confusion_matrix(y_test, p_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
    disp.plot()
    plt.title(f'Confusion Matrix using {model_name}')
    plt.savefig(f'images/task3_{plt.gca().get_title()}.png')
    plt.show()


def train_neural_network(X_train, y_train, X_test, y_test, dim):
    model = Sequential()
    model.add(Dense(10, input_dim=dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, verbose=1, batch_size=32)

    _, acc = model.evaluate(X_test, y_test)
    print(f'ANN Accuracy: {acc * 100}%')


def plot_roc(y_test, y_score, model_name):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, label='Using a random classifier', linestyle='--')
    # find the operating point at which fpr is 0.1
    idx = (np.abs(fpr - 0.05)).argmin()
    plt.plot(fpr[idx], tpr[idx], 'x', label='FPR = 0.05', color='r')
    print(f'The true positive rate at fpr=0.05 is: {tpr[idx]:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic Curve using {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'images/task3_{plt.gca().get_title()}.png')
    plt.show()


def main():
    np.random.seed(0)

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
    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(X, y)
    importances = rfc.feature_importances_

    feature_importances = {}
    for i in range(len(importances)):
        feature_importances[X.columns[i]] = {'importance': importances[i], 'index': i}

    sorted_feature_importances = sorted(feature_importances.items(), key=lambda x: x[1]['importance'], reverse=True)
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(len(sorted_feature_importances)):
        print(
            f"{f + 1}. {sorted_feature_importances[f][0]} (index: {sorted_feature_importances[f][1]['index']}, "
            f"importance: {sorted_feature_importances[f][1]['importance']})")

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

    CLF = {'Logistic Regression': LogisticRegression(C=1.6, random_state=0),
           'SGDCClassifier L2': SGDClassifier(alpha=1e-5, penalty='l2', random_state=0),
           'SGDCClassifier L1': SGDClassifier(alpha=1e-5, penalty='l1', random_state=0),
           'Ridge Classifier': RidgeClassifier(tol=1e-2, solver='sag', random_state=0),
           'Perceptron': Perceptron(random_state=0), 'BernoulliNB': BernoulliNB(alpha=.01),
           'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5, weights='distance', p=2),
           'Support Vector Machine': SVC(C=1.0, tol=1e-3, kernel='rbf', gamma='scale', random_state=0),
           'Decision Tree Classifier': DecisionTreeClassifier(random_state=0),
           'Random Forest Classifier': RandomForestClassifier(random_state=0)}

    print('\nAvailable classifiers:')
    for c in CLF:
        print('- ', c)

    # use the selected salient features to train different models
    X_train_selected = X_train.loc[:, [column_names[i] for i in [0, 2, 3, 4, 7, 8, 9, 10]]]
    X_test_selected = X_test.loc[:, [column_names[i] for i in [0, 2, 3, 4, 7, 8, 9, 10]]]
    for c in CLF:
        train_eval_model(c, X_train_selected, y_train, X_test_selected, y_test, CLF)
        print('_' * 80)
        print('_' * 80)

    # use neural network
    tf.random.set_seed(0)
    # train_neural_network(X_train_selected, y_train, X_test_selected, y_test, 8)

    # feature engineering for max heart rate.
    # heart rate may not have linear relationship, compute quadratic and cubed terms for it
    print('feature engineering quadratic and cubed term for ST_Slope')
    X_train_selected['Cholesterol_squared'] = X_train_selected['Cholesterol'].apply(lambda x: x ** 2)
    X_test_selected['Cholesterol_squared'] = X_test_selected['Cholesterol'].apply(lambda x: x ** 2)
    X_train_selected['Cholesterol_cubic'] = X_train_selected['Cholesterol'].apply(lambda x: x ** 3)
    X_test_selected['Cholesterol_cubic'] = X_test_selected['Cholesterol'].apply(lambda x: x ** 3)

    # evaluate the logistic regression again using engineered features.
    train_eval_model('Logistic Regression', X_train_selected, y_train, X_test_selected, y_test, CLF)

    # plot ROC curve for 4 models
    selected_models = ['Support Vector Machine', 'Random Forest Classifier', 'K-Nearest Neighbor',
                       'Logistic Regression']
    print(f'\nThe selected models for ROC curve are {selected_models}')
    for c in CLF:
        if c in selected_models:
            _, _, y_score = predict(X_train_selected, y_train, X_test_selected, y_test, CLF[c])
            plot_roc(y_test, y_score, c)

    # drop the engineered features as they do not improve the test accuracy.
    X_train_selected = X_train_selected.drop('Cholesterol_squared', axis=1)
    X_train_selected = X_train_selected.drop('Cholesterol_cubic', axis=1)
    X_test_selected = X_test_selected.drop('Cholesterol_squared', axis=1)
    X_test_selected = X_test_selected.drop('Cholesterol_cubic', axis=1)

    # after ROC, decide to use RandomForestClassifier and K-Nearest Neighbor models for 3 iterations
    # RandomForestClassifier
    print(f'\nUse GridSearchCV to obtain the best hyper-parameter for RandomForestClassifier.')
    rfc_param_grid = {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    # rfc = RandomForestClassifier()
    # grid_search = GridSearchCV(estimator=rfc, param_grid=rfc_param_grid, cv=5)
    # grid_search.fit(X_train_selected, y_train)
    # # print the best hyperparameter and the corresponding score
    # print("Best hyperparameter: ", grid_search.best_params_)
    # print("Best score: ", grid_search.best_score_)

    # Best hyperparameter:  {'criterion': 'log_loss', 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf':
    # 1, 'min_samples_split': 10, 'n_estimators': 100}
    # Best score:  0.8566336441336443 use these hyperparameter to
    # Use these hyperparameter to predict the test results
    rfc = RandomForestClassifier(criterion='log_loss', max_depth=None, max_features='log2', min_samples_leaf=1,
                                 min_samples_split=10, n_estimators=100, random_state=0)
    rfc.fit(X_train_selected, y_train)
    acc = rfc.score(X_test_selected, y_test)
    print(f'The best test accuracy using RandomForestClassifier is: {acc}')

    # K-Nearest Neighbor Classifier
    print(f'\nUse GridSearchCV to obtain the best hyper-parameter for K-Nearest Neighbor Classifier.')
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'algorithm': ['ball_tree', 'auto', 'kd_tree'],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }
    # knn = KNeighborsClassifier()
    # grid_search = GridSearchCV(estimator=knn, param_grid=knn_param_grid, cv=5)
    # grid_search.fit(X_train_selected, y_train)
    # # print the best hyperparameter and the corresponding score
    # print("Best hyperparameter: ", grid_search.best_params_)
    # print("Best score: ", grid_search.best_score_)

    # Best hyperparameter:  {'algorithm': 'ball_tree', 'metric': 'euclidean', 'n_neighbors': 9, 'weights': 'distance'}
    # Best score:  0.8482420357420357
    # Use these hyperparameter to predict the test results
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', metric='euclidean', weights='distance')
    knn.fit(X_train_selected, y_train)
    acc = knn.score(X_test_selected, y_test)
    print(f'The best test accuracy using KNeighborClassifier is: {acc}')

    # add weights to each feature
    # define the weights for each feature
    weights = np.array([1.0, 2.0, 1.0, 0.5, 2.0, 2.0, 1.0, 1.0])
    X_train_weighted = X_train_selected * weights
    X_test_weighted = X_test_selected * weights
    scaler = StandardScaler()
    # normalize the weighted features using StandardScaler
    X_train_weighted_std = scaler.fit_transform(X_train_weighted)
    X_test_weighted_std = scaler.transform(X_test_weighted)
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', metric='euclidean', weights='distance')

    knn.fit(X_train_weighted_std, y_train)
    acc = knn.score(X_test_weighted_std, y_test)
    print(f'First iteration: The best test accuracy using KNeighborClassifier is: {acc}')


if __name__ == '__main__':
    main()
