from lib.learner import Learner
from lib.feature import *
from sklearn_crfsuite import metrics

train_path = '../data/train_lite.txt'
crf_k      = 5

# CRF
crf_train  = Context(train_path, k=crf_k)
man = Learner(crf_train)
man.X_train = man.X
man.Y_train = man.Y
man.train()
man.save('./pickles/crf_model-best.pkl')
