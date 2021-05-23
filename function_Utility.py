"""
<Red Wine Quality>

Copyright (c) 2021
Licensed
Written by <Jiajing Cai/ Zhenghao Deng/ Xinhang Li>
"""

import codecs
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

def read_csv(path):
    """

    :param path: Data set path
    :return: wine_df Dataframe
    """
    #import dataset
    file = codecs.open(path, 'r', 'utf-8')
    
    #Separate data and make it readable
    data = file.readlines()

    for i in range(len(data)):
        if i == 0:
            data[0] = [item.strip('"') for item in data[i].split(';')]
            data[0][len(data[0]) - 1] = data[0][len(data[0]) - 1].strip('"\n')
            continue
        data[i] = [float(item) for item in data[i].split(';')]

    wine_df = pd.DataFrame(columns=data[0], data=data[1:])

    return wine_df


def data_cleansing(dataframe):

    print("Check null and missing values in the dataframe and the result is:")
    print(dataframe.isnull().sum())
    dataframe = dataframe.drop_duplicates(inplace=True)

    return


def split_set(dataframe):

    Y = dataframe.quality
    X = dataframe.drop('quality', axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    return X_train, X_test, Y_train, Y_test


def modeling(X_train, X_test, Y_train, Y_test):


    clf_lr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    clf_lr.fit(X_train, Y_train)
    y_multi_log_pred = clf_lr.predict(X_test)

    clf_tree = DecisionTreeClassifier()
    clf_tree.fit(X_train, Y_train)
    y_pred_tree = clf_tree.predict(X_test)

    print('Accuracy of LogisticRegression Model: ' + str(metrics.accuracy_score(Y_test, y_multi_log_pred)))
    print('Accuracy of DecisionTree Model: ' + str(metrics.accuracy_score(Y_test, y_pred_tree)))

    importances = clf_tree.feature_importances_

    d = {"feature": X_train.columns, "importances": importances}
    feature_im = pd.DataFrame(data=d)
    feature_im = feature_im.sort_values('importances', ascending=True)
    plt.barh('feature', 'importances', data=feature_im)
    plt.title('Feature Importance - Red wine quality (Decision Tree)')
    plt.xticks(rotation=90)
    print('Visualize Feature Importance: ')
    plt.show()

    return


def binary_classificaion(dataframe):

    bins = (2, 6.5, 8)
    quality_level = ['bad', 'good']
    dataframe['quality'] = pd.cut(dataframe['quality'], bins = bins, labels = quality_level)
    
    label = LabelEncoder()
    dataframe['quality'] = label.fit_transform(dataframe['quality'])
        
    x = dataframe.drop('quality', axis = 1)
    y = dataframe['quality']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    return x_train, x_test, y_train, y_test


def Random_Forests(x_train, x_test, y_train, y_test):

    rf_model = RandomForestClassifier(n_estimators=200)
    rf_model.fit(x_train, y_train)
    pred_rf = rf_model.predict(x_test)

    print(classification_report(y_test, pred_rf))
    
    acc_rf = accuracy_score(y_test, pred_rf)
    print('Accuracy of Random Forests Model:', acc_rf*100)
    
    importances = rf_model.feature_importances_

    d = {"feature": x_train.columns, "importances": importances}
    feature_im = pd.DataFrame(data=d)
    feature_im = feature_im.sort_values('importances', ascending=True)
    plt.barh('feature', 'importances', data=feature_im)
    plt.title('Feature Importance - Red wine quality (Random Forest)')
    plt.xticks(rotation=90)
    print('Visualize Feature Importance: ')
    plt.show()
    
    return
