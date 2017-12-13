from lib.crf import CRF, WeightCRF
from lib.data import Data
from lib.learner import Learner, WeightLearner
from lib.feature import Feature
from lib.ensumble_learner import Boosting
from lib.ensumble_learner import Bagging
from pprint import pprint
import csv

man = Boosting('./data/data4.txt')
man.load_feature(funcs=[Feature.context], params=[{'k':1, 'n_gram':2}])
man.train()
man.report()

