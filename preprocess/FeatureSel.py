import random
from numpy import vstack, argsort
from sklearn import preprocessing, ensemble, model_selection, feature_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from utils import load_data

random.seed(100)


def feature_rank_acc(x, y):
    clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    for i in range(0, 300):
        output = model_selection.cross_validate(clf, x, y, cv=10, scoring='accuracy', return_estimator=True)
        for idx, estimator in enumerate(output['estimator']):
            if idx == 0 and i == 0:
                FeatureImp = estimator.feature_importances_
            else:
                FeatureImp = vstack((FeatureImp, estimator.feature_importances_))

    ImpMean = FeatureImp.mean(axis=0)
    ImpStd = FeatureImp.std(axis=0)
    sorted_idx = ImpMean.argsort()[::-1]

    # Plot the impurity-based feature importances of the forest
    plt.figure(figsize=(10, 6))
    plt.title("Feature importances")
    plt.bar(range(x.shape[1]), ImpMean[sorted_idx],
            color='lightblue', yerr=ImpStd[sorted_idx], align="center", ecolor='gray', error_kw={'elinewidth': 1})
    plt.xticks(range(x.shape[1]), x.columns[sorted_idx], rotation=90, fontsize=10)
    plt.xlim([-1, x.shape[1]])
    plt.subplots_adjust(bottom=0.55)
    plt.savefig("feature_rank_acc.jpg")
    # plt.show()
    # fig.savefig(r'Decrese_in_Accuracy.eps', format='eps', bbox_inches='tight')


def RandomCV(x, y, nfold, nestimator):
    #  ---Classification with n-fold cross-validation---
    # --- x is feature, y is label, clf is classifier, n is number of fold
    # ---  define K-fold cross validation ---
    KF = KFold(n_splits=nfold, shuffle=True, random_state=5)
    i = 0
    for train_index, test_index in KF.split(x):
        # ---  Seperate traing set and test set ---#
        x_train, x_test = x.iloc[train_index][:], x.iloc[test_index][:]
        y_train = y.iloc[train_index][:]

        # ---  creat and train the model ---#
        clf = ensemble.RandomForestClassifier(n_estimators=nestimator, random_state=42, class_weight="balanced")
        clf.fit(x_train, y_train)
        if i == 0:
            FeatureImp = clf.feature_importances_
        else:
            FeatureImp = vstack((FeatureImp, clf.feature_importances_))
        i += 1
    return FeatureImp


def feature_rank_purity(x, y):
    for i in range(0, 300):
        if i == 0:
            FeatureImp = RandomCV(x, y, 5, 100)
        else:
            FeatureImp = vstack((FeatureImp, RandomCV(x, y, 5, 100)))

    ImpMean = FeatureImp.mean(axis=0)
    ImpStd = FeatureImp.std(axis=0)
    sorted_idx = ImpMean.argsort()[::-1]

    # Plot the impurity-based feature importances of the forest
    plt.figure(figsize=(10, 6))
    plt.title("Feature importances")
    plt.bar(range(x.shape[1]), ImpMean[sorted_idx],
            color='lightblue', yerr=ImpStd[sorted_idx], align="center", ecolor='gray', error_kw={'elinewidth': 1})
    plt.xticks(range(x.shape[1]), x.columns[sorted_idx], rotation=90, fontsize=10)
    plt.xlim([-1, x.shape[1]])
    plt.subplots_adjust(bottom=0.55)
    plt.savefig("feature_rank_purity.jpg")
    # plt.show()
    # fig.savefig(r'Decrese_in_Purity.eps', format='eps', bbox_inches='tight')


def feature_rank_recu(x, y):
    clf = ensemble.RandomForestClassifier(n_estimators=31, random_state=42, class_weight="balanced")
    rfecv = feature_selection.RFECV(estimator=clf, step=1, cv=StratifiedKFold(5), scoring='accuracy')
    rfecv.fit(x, y)
    print(rfecv.ranking_)

    Imp = rfecv.ranking_
    sorted_idx = Imp.argsort()[::-1]
    ImpStd = Imp.std(axis=0)

    plt.figure(figsize=(10, 6))
    plt.title("Feature importances")
    plt.bar(range(x.shape[1]), Imp[sorted_idx], color='lightblue')
    plt.xticks(range(x.shape[1]), x.columns[sorted_idx], rotation=90, fontsize=10)
    plt.xlim([-1, x.shape[1]])
    plt.subplots_adjust(bottom=0.55)
    plt.savefig("feature_rank_recu.jpg")
    # plt.show()
    # fig.savefig(r'Decrese_in_RECU.eps', format='eps', bbox_inches='tight')


def print_coefs(coefs, name):
    # sort coefficients from smallest to largest, then reverse it
    inds = argsort(abs(coefs))[::-1]
    # print out
    print("weight : feature description")
    for i in inds:
        print("{: .3f} : {:5s}".format(coefs[i], name[i]))


if __name__ == '__main__':
    x, y, label = load_data()
    feature_rank_acc(x, y)
    feature_rank_purity(x, y)
    feature_rank_recu(x, y)

    # alphas = numpy.logspace(-3, 6, 50)
    # rr = sklearn.linear_model.RidgeCV(alphas=alphas, cv=5)
    # rr.fit(x, y)
    # print_coefs(rr.coef_, label)
