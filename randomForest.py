from lib.crf import CRF, WeightCRF
from lib.data import Data
from lib.learner import *
from lib.feature import Feature
from lib.ensumble_learner import Boosting
from lib.ensumble_learner import Bagging
from pprint import pprint
import csv

path = './data/data4.txt'

man = RandomForestLearner(path, random_state=2, max_dim=200, shuffle=False)
man.load_feature(funcs=[Feature.context], params=[{'k':1, 'n_gram':2}])
man.train()
man.report()

print('baseline')
man = Learner(path, random_state=2, shuffle=False)
man.load_feature(funcs=[Feature.context], params=[{'k':1, 'n_gram':2}])
man.train()
man.report()
