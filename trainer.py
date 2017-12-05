from lib.crf import CRF
from lib.data import Data
from lib.learner import Learner
from lib.feature import *
from lib.ensumble_learner import Bagging

bagging = Bagging('./data/data2.txt')
bagging.feature_loader(funcs=[context], params=[[]])
bagging.resize(0.1)
bagging.train()
bagging.report()
