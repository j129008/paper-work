import numpy as np
from lib.learner import Learner
from lib.feature import *
from tqdm import tqdm as bar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn_crfsuite import metrics
import lightgbm as lgb
import csv

path = './data/data_lite.txt'
result_table = csv.writer( open('./csv/other_model.csv', 'w') )
vec = UniVec(path, vec_size=50, k=2, mode='chain')
X = np.array(vec.X)
Y = np.array(vec.Y)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, shuffle=False
)

clf_list = [
    ('random forest', RandomForestClassifier(n_jobs=8)),
    ('adaboost', AdaBoostClassifier()),
    ('MLP', MLPClassifier(hidden_layer_sizes=(20,10,5), alpha=1e-10)),
    ('GaussianNB', GaussianNB()),
    ('svm', svm.SVC(C=1000, max_iter=5000)),
    ('decision tree', DecisionTreeClassifier()),
]

def report(pred, truth, csv_table, clf_name):
    label = 'E'
    pred_lab = VecContext.y2lab(pred)
    truth_lab = VecContext.y2lab(truth)
    P = metrics.flat_precision_score(truth_lab, pred_lab, pos_label=label)
    R = metrics.flat_recall_score(truth_lab, pred_lab, pos_label=label)
    f1 = metrics.flat_f1_score(truth_lab, pred_lab, pos_label=label)
    print(clf_name)
    print(metrics.flat_classification_report(
        truth_lab, pred_lab, labels=('I', 'E'), digits=4
    ))
    csv_table.writerow([clf_name, P, R, f1])

for clf_name, clf in bar(clf_list):
    clf.fit(x_train, y_train)
    pred_vec = clf.predict(x_test)
    report(pred_vec, y_test, result_table, clf_name)


# lgb
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=0.1, shuffle=True
)

train_data = lgb.Dataset(x_train, y_train)
valid_data = lgb.Dataset(x_valid, y_valid, reference=train_data)

params = {
        'task'             : 'train',
        'boosting_type'    : 'gbdt',
        'objective'        : 'binary',
        'metric'           : {'l2', 'auc'},
        'num_leaves'       : 2047,
        'learning_rate'    : 0.05,
        'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8,
        'bagging_freq'     : 5,
        'verbose'          : 0
}
num_round = 1000
bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data], early_stopping_rounds=10)

pred = bst.predict(x_test)
report(pred, y_test, result_table, 'lgb')
