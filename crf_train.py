from lib.data import Data
from lib.learner import Learner
from lib.feature import *
from lib.ensumble_learner import Bagging, Boosting
from lib.metric import *
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics

train_path = './data/train-valid.txt'
crf_k      = 5

# CRF
crf_train  = Context(train_path, k=crf_k)
man = Learner(crf_train)
man.X_train = man.X
man.Y_train = man.Y
man.train()
man.save('./pickles/crf_model-valid.pkl')
