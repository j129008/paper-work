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
import csv

path = './data/data_proc.txt'
result_table = csv.writer( open('./other_model.csv', 'w') )
vec = UniVec(path)
X = np.array(vec.X)
Y = np.array(vec.Y)

clf_list = [
    RandomForestClassifier(n_jobs=8),
    KNeighborsClassifier(n_neighbors=3),
    AdaBoostClassifier(),
    MLPClassifier(hidden_layer_sizes=(20,10,5), alpha=1e-10),
    GaussianNB(),
    DecisionTreeClassifier()
]

kf = KFold(n_splits=3)
for clf in bar(clf_list):
    P_list = []
    R_list = []
    f1_list = []
    for train, test in kf.split(X):
        clf.fit(X[train], Y[train])
        pred_vec = clf.predict(X[test])
        pred_lab = vec.y2lab(pred_vec)
        y_test = vec.y2lab(Y[test])
        label = 'E'
        P = metrics.flat_precision_score(y_test, pred_lab, pos_label=label)
        R = metrics.flat_recall_score(y_test, pred_lab, pos_label=label)
        f1 = metrics.flat_f1_score(y_test, pred_lab, pos_label=label)
        P_list.append(P)
        R_list.append(R)
        f1_list.append(f1)
        print(P, R, f1)
    avg = lambda l:sum(l)/len(l)
    result_table.writerow([avg(P_list), avg(R_list), avg(f1_list)])
