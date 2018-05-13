from lib.data import Data
from lib.learner import Learner
from lib.feature import *
from lib.ensumble_learner import Bagging, Boosting
from lib.metric import *
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics

train_path = './data/train.txt'
crf_k      = 5

# CRF
crf_train  = Context(train_path, k=crf_k)
tdiff = Tdiff(train_path)
pmi = MutualInfo(train_path)
man = Learner(crf_train + tdiff + pmi)
man.X_train = man.X
man.Y_train = man.Y
man.train_CV(n_iter=4, cv=6)
man.save('./pickles/crf_model-best.pkl')
