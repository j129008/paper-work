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
result_table = csv.writer( open('./csv/other_model.csv', 'w') )
vec = UniVec(path, vec_size=50, k=1)
X = np.array(vec.X)
Y = np.array(vec.Y)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, shuffle=False
)

clf_list = [
    RandomForestClassifier(n_jobs=8),
    #  KNeighborsClassifier(n_neighbors=8, n_jobs=8),
    #  AdaBoostClassifier(),
    #  MLPClassifier(hidden_layer_sizes=(20,10,5), alpha=1e-10),
    #  GaussianNB(),
    #  DecisionTreeClassifier(),
    #  svm.SVC(C=1000, max_iter=5000)
]

for clf in bar(clf_list):
    clf.fit(x_train, y_train)
    pred_vec = clf.predict(x_test)
    pred_lab = vec.y2lab(pred_vec)
    y_test = vec.y2lab(y_test)
    label = 'E'
    P = metrics.flat_precision_score(y_test, pred_lab, pos_label=label)
    R = metrics.flat_recall_score(y_test, pred_lab, pos_label=label)
    f1 = metrics.flat_f1_score(y_test, pred_lab, pos_label=label)
    print(P, R, f1)
    result_table.writerow(P, R, f1)
