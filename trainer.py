from lib.crf import CRF
from lib.data import Data
from lib.learner import Learner
from lib.feature import *
from pprint import pprint

learner = Learner('./data/data3.txt')
learner.feature_loader(funcs=[context], params=[[]])
learner.train_CV()
learner.report()
