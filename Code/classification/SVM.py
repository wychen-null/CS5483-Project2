from preprocess.utils import *
random.seed(4487)


def SVM_linear(trainX, trainY, testX):
    svc = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
    param_grid = {
        'C': [0.01, 0.1, 0.5, 1],
        # 'C': [50]
    }
    # s_svc = model_selection.GridSearchCV(svc, param_grid, cv=5, scoring='roc_auc')
    s_svc = model_selection.GridSearchCV(svc, param_grid, cv=5)
    s_svc.fit(trainX, trainY)

    best_params = s_svc.best_params_
    print("SVM(linear) Best Parameters:", best_params)

    pred = s_svc.predict(testX)
    prob = s_svc.predict_proba(testX)[:, 1]

    return pred, prob


def SVM_rbf(trainX, trainY, testX):
    svc = svm.SVC(kernel='rbf', probability=True)
    paramgrid = {'C': logspace(-3, 3, 13), 'gamma': logspace(-4,3,20)}

    svc_rbf = model_selection.GridSearchCV(svc, paramgrid, cv=5, n_jobs=-1, verbose=True)
    svc_rbf.fit(trainX, trainY)

    print("SVM(rbf) Best Parameter:", svc_rbf.best_params_)

    pred = svc_rbf.predict(testX)
    prob = svc_rbf.predict_proba(testX)[:, 1]

    return pred, prob


def SVM_poly(trainX, trainY, testX):
    svc = svm.SVC(kernel='poly', probability=True)

    parameters = {
        'C': [0.1, 1],
        'degree': [2, 3]
    }

    svc_poly = model_selection.GridSearchCV(svc, parameters, cv=5, scoring='accuracy')
    svc_poly.fit(trainX, trainY)
    best_params = svc_poly.best_params_
    # svc_poly = svm.SVC(kernel='poly', C=best_params['C'], degree=best_params['degree'], probability=True)
    # svc_poly.fit(trainX, trainY)

    pred = svc_poly.predict(testX)
    prob = svc_poly.predict_proba(testX)[:, 1]

    return pred, prob


if __name__ == '__main__':
    x, y, label = load_data()
    trainX, trainY, testX, testY = train_test_split(x, y)

    # SVM-linear kernel 0.7507-0.7603-0.6460
    SVM_pred, SVM_prob = SVM_linear(trainX, trainY, testX)
    acc = metrics.accuracy_score(testY, SVM_pred)
    print(acc)
    plot_roc(testY, SVM_prob)

    # SVM-rbf kernel 0.7865-0.8209-0.7466
    SVM_rbf_pred, SVM_rbf_prob = SVM_rbf(trainX, trainY, testX)
    acc = metrics.accuracy_score(testY, SVM_rbf_pred)
    print(acc)
    plot_roc(testY, SVM_rbf_prob)

    # SVM-poly kernel 0.6928-0.6901-0.5165
    SVM_poly_pred, SVM_poly_prob = SVM_poly(trainX, trainY, testX)
    acc = metrics.accuracy_score(testY, SVM_poly_pred)
    print(acc)
    plot_roc(testY, SVM_poly_prob)


