from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from preprocess.utils import *
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

random.seed(4487)


def RF(trainX, trainY, testX):
    param_grid = {
        'n_estimators': [500, 600, 800]
    }

    model = RandomForestClassifier(random_state=4487)

    rf = model_selection.GridSearchCV(model, param_grid, cv=5)
    rf.fit(trainX, trainY)

    print("RF Parameter:", rf.best_params_)

    pred = rf.predict(testX)
    prob = rf.predict_proba(testX)[:, 1]
    return pred, prob


def XGBoost(trainX, trainY, testX):
    xclf = XGBClassifier(objective="multi:softprob",
                         eval_metric='logloss',
                         random_state=4487,
                         use_label_encoder=False,
                         num_class=8)
    paramgrid = {
        'learning_rate': logspace(-3, 3, 20),
        'n_estimators': array([500, 1000])
    }
    # print(paramgrid)

    xgb = model_selection.GridSearchCV(xclf, paramgrid, cv=5, n_jobs=-1)
    xgb.fit(trainX, trainY)
    print("best params:", xgb.best_params_)

    pred = xgb.predict(testX)
    prob = xgb.predict_proba(testX)[:, 1]

    return pred, prob


def KNN(trainX, trainY, testX):
    model = neighbors.KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [i for i in range(2, 30)],
        'weights': ['distance', 'uniform'],
        'p': [1, 2]
    }
    knn = model_selection.GridSearchCV(model, param_grid=param_grid, cv=5)
    knn.fit(trainX, trainY)

    print("knn best parameters: ", knn.best_params_)
    pred = knn.predict(testX)
    prob = knn.predict_proba(testX)[:, 1]
    return pred, prob


def GPC(trainX, trainY, testX):
    model = GaussianProcessClassifier(random_state=4487)
    kernel_options = [1.0 * RBF(length_scale=10),
                      1.0 * Matern(length_scale=10)]
    param_grid = {'kernel': kernel_options, 'n_restarts_optimizer': [0, 1, 2]}
    gpc = model_selection.GridSearchCV(model, param_grid, cv=5)
    gpc.fit(trainX, trainY)

    print("GPC best parameters: ", gpc.best_params_)
    pred = gpc.predict(testX)
    prob = gpc.predict_proba(testX)[:, 1]
    return pred, prob


if __name__ == '__main__':
    x, y, label = load_data()
    trainX, trainY, testX, testY = train_test_split(x, y)

    # 0.7687-0.7672
    RF_pred, RF_prob = RF(trainX, trainY, testX)
    acc = metrics.accuracy_score(testY, RF_pred)
    print(acc)
    plot_roc(testY, RF_prob)

    # 0.7741
    xgb_pred, xgb_prob = XGBoost(trainX, trainY, testX)
    acc = metrics.accuracy_score(testY, xgb_pred)
    print(acc)
    plot_roc(testY, xgb_prob)

    # 0.7631-0.6990
    knn_pred, knn_prob = KNN(trainX, trainY, testX)
    acc = metrics.accuracy_score(testY, knn_pred)
    print(acc)
    plot_roc(testY, knn_prob)

    # 0.7879-0.8017
    gpc_pred, gpc_prob = GPC(trainX, trainY, testX)
    acc = metrics.accuracy_score(testY, gpc_pred)
    print(acc)
    plot_roc(testY, gpc_prob)
