from lib.crf import CRF
from lib.data import Data
from lib.learner import Learner
from lib.feature import *
from pprint import pprint

learner = Learner('./data/data2.txt')
learner.feature_loader(funcs=[context], params=[[]])
learner.resize(0.1)
learner.train_CV()
learner.report()
learner.predict_file('./data/data3.txt')
learner.report()
