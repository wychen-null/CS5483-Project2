import random
import pickle
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
    xl = pd.ExcelFile('C:/Users/HUAWEI/Desktop/CancerDetection/preprocess/data2.xlsx')
    dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}

    data1 = dfs['7']
    data2 = dfs['1'].loc[:, ['Patient', 'Age at Diagnosis']].drop([554]).drop_duplicates()
    data3 = pd.read_csv('C:/Users/HUAWEI/Desktop/CancerDetection/preprocess/data1.csv')
    combined_data = data1.set_index('Patient').join(data2.set_index('Patient')).join(data3.set_index('Patient'))
    combined_data['label'] = np.where(combined_data['Patient Type'] == 'Healthy', 0, 1)  # transform to label
    combined_data = combined_data.drop(['Patient Type'], axis=1)
    combined_data.to_csv('data.csv', index=False)
    print(
        'The number of samples and features are %d and %d, respectively' % (
            combined_data.shape[0], combined_data.shape[1]))

    x = combined_data.iloc[:, 0:44]
    x[isnan(x)] = 0
    y = combined_data.iloc[:, 44]
    label=np.where(combined_data['label'] == 0, 'Healthy', 'Postive')

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
    return trainX, trainY, testX, testY


def plot_roc(validY, validProb):
    # ROC AUC
    fpr, tpr, thresholds = metrics.roc_curve(validY, validProb, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    print("ROC_AUC :", roc_auc)

    fig = plt.figure()
    fig.patch.set_facecolor('#e8e8f8')
    plt.plot(fpr, tpr, lw=2, color='#7777cb')
    plt.title('AUC={:.4f}'.format(roc_auc))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.gca().set_facecolor('#e8e8f8')

    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    x, y, label = load_data()
    data = {
    'x': x,
    'y': y,
    'label': label
}

# 打开一个文件以便写入
    with open('data.pkl', 'wb') as file:
    # 使用pickle的dump方法将数据序列化至文件
        pickle.dump(data, file)

