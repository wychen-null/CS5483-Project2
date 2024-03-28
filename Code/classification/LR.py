from numpy import logspace
from sklearn import linear_model, metrics

from preprocess.utils import *


def LR(trainX, trainY, testX):
    lr = linear_model.LogisticRegressionCV(Cs=logspace(-4, 4, 20), cv=5, max_iter=10000, multi_class='multinomial', scoring='accuracy')
    lr.fit(trainX, trainY)

    prob = lr.predict_proba(testX)[:, 1]
    pred = lr.predict(testX)

    return pred, prob


if __name__ == '__main__':
    x, y, label = load_data()
    trainX, trainY, testX, testY = train_test_split(x, y)

    # 0.7590
    lr_pred, lr_prob = LR(trainX, trainY, testX)
    acc = metrics.accuracy_score(testY, lr_pred)  # 超低
    print(acc)
    plot_roc(testY, lr_prob)
