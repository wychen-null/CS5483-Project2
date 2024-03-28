from sklearn import naive_bayes
from sklearn.model_selection import GridSearchCV

from preprocess.utils import *


# 伯努利贝叶斯
def Bnl_NB(trainX, trainY, testX):
    model = naive_bayes.BernoulliNB()
    # cross-validation: choose best model
    alpha = [1.0e-10, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1]
    param_grid = {'alpha': alpha}

    bnb = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    bnb.fit(trainX, trainY)
    print("Best alpha :", bnb.best_params_)

    pred = bnb.predict(testX)
    prob = bnb.predict_proba(testX)[:, 1]
    return pred, prob


# 高斯贝叶斯
def Gn_NB(trainX, trainY, testX):
    gnb = naive_bayes.GaussianNB()
    gnb.fit(trainX, trainY)

    pred = gnb.predict(testX)
    prob = gnb.predict_proba(testX)[:, 1]
    return pred, prob


if __name__ == '__main__':
    x, y, label = load_data()
    trainX, trainY, testX, testY = train_test_split(x, y)

    # 0.8388-0.8939
    bnb_pred, bnb_prob = Bnl_NB(trainX, trainY, testX)
    acc = metrics.accuracy_score(testY, bnb_pred)  # 超低
    print(acc)
    plot_roc(testY, bnb_prob)

    # 0.8430
    gnb_pred, gnb_prob = Gn_NB(trainX, trainY, testX)
    acc = metrics.accuracy_score(testY, gnb_pred)  # 超低
    print(acc)
    plot_roc(testY, gnb_prob)
