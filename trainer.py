from lib.crf import CRF
from lib.data import Data
from lib.learner import Learner
from lib.feature import *
from lib.ensumble_learner import Boosting

boosting = Boosting('./data/data2.txt')
boosting.feature_loader(funcs=[context], params=[[]])
boosting.resize(0.0000001)
boosting.train()
boosting.report()
