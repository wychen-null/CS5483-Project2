import random

import numpy as np
from matplotlib import pyplot as plt
from numpy import *
from sklearn import *
import pandas as pd
from sklearn.utils import shuffle
from sklearn.utils.multiclass import type_of_target
random.seed(4487)


# load data from file
def load_data():
    xl = pd.ExcelFile('C:/Users/lijia/PycharmProjects/CancerDetection/data/data2.xlsx')
    # print(xl.sheet_names)
    dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}

    data1 = dfs['7']
    # data1.to_csv('data1_test.csv', index=False)
    data2 = dfs['1'].loc[:, ['Patient', 'Age at Diagnosis']].drop([554]).drop_duplicates()
    data3 = pd.read_csv('C:/Users/lijia/PycharmProjects/CancerDetection/data/data1.csv')
    # Median cfDNA Fragment Size (bp), GC Corrected Fragment Ratio Profile, age
    # PA Score,Patient Type,% of Mapped Reads Mapping to Mitochondria, 39 z score
    combined_data = data1.set_index('Patient').join(data2.set_index('Patient')).join(data3.set_index('Patient'))

    combined_data = combined_data[~combined_data['Patient Type'].isin(['Duodenal Cancer'])]  # 移除十二指肠癌
    le = preprocessing.LabelEncoder().fit(combined_data['Patient Type'])
    combined_data['label'] = le.transform(combined_data['Patient Type'])  # transform to label
    combined_data = combined_data.drop(['Patient Type'], axis=1)
    combined_data.to_csv('data.csv', index=False)
    print(
        'The number of samples and features are %d and %d, respectively' % (
            combined_data.shape[0], combined_data.shape[1]))

    x = combined_data.iloc[:, 0:44]
    x[isnan(x)] = 0
    y = combined_data.iloc[:, 44]
    label = []
    for i in range(0, 8):
        label.append(le.inverse_transform([i])[0])

    print(type_of_target(x))
    print(type_of_target(y))
    print(type(x))
    print(type(y))

    x, y = shuffle(x, y, random_state=4487)

    scaler = preprocessing.StandardScaler()  # make scaling object
    x = scaler.fit_transform(x)  # use training data to fit scaling parameters

    return x, y, label


def train_test_split(x, y):
    trainX, testX, trainY, testY = \
        model_selection.train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=4487)

    # scaler = preprocessing.StandardScaler()  # make scaling object
    # trainXn = scaler.fit_transform(trainX)  # use training data to fit scaling parameters
    # testXn = scaler.transform(testX)  # apply scaling to test data
    #
    # return trainXn, trainY, testXn, testY
    return trainX, trainY, testX, testY


def plot_roc(validY, validProb):
    # ROC AUC
    fpr, tpr, thresholds = metrics.roc_curve(validY, validProb, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    print("ROC_AUC :", roc_auc)

    plt.plot(fpr, tpr, 'k-', lw=2)
    plt.title('AUC={:.4f}'.format(roc_auc))
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    x, y, label = load_data()

