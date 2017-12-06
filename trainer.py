from lib.crf import CRF
from lib.data import Data
from lib.learner import Learner
from lib.feature import *
from lib.ensumble_learner import Bagging

bagging = Bagging('./data/data2.txt')
bagging.feature_loader(funcs=[context], params=[[]])
bagging.resize(0.1)

for voter in range(3, 11):
    for train_size in [ 0.03, 0.05, 0.1, 0.2, 0.3, 0.4 ]:
        print('voter:', voter, 'data size:', train_size)
        model = bagging.train(voter=voter, train_size=train_size)
        bagging.report(model=model)
