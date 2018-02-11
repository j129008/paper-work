from gensim.models import Word2Vec
from pprint import pprint
from lib.learner import Learner
from lib.feature import *
from sklearn.ensemble import RandomForestClassifier
from sklearn_crfsuite import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle

data = UniVec('./data/data4.txt')
x_train, x_test, y_train, y_test = train_test_split(
    data.X, data.Y, test_size=0.6, random_state=1, shuffle=False
)

clf = svm.SVC()
clf.fit(x_train, y_train)

y_pred = clf.predict( x_test )
y_pred = data.y2lab(y_pred)
y_test = data.y2lab(y_test)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=('I', 'E'), digits=4
))
