from lib.crf import CRF, WeightCRF
from lib.data import Data
from lib.learner import Learner, WeightLearner
from lib.feature import Feature
from lib.ensumble_learner import Boosting
from lib.ensumble_learner import Bagging
from pprint import pprint
import csv

man = Boosting('./data/data3.txt')
man.load_feature(funcs=[Feature.context, Feature.t_diff, Feature.mi_info, Feature.rhyme], params=[{'k':1, 'n_gram':2}, {}, {}, {'path':'./data/rhyme.txt', 'pkl_path':'./pickles/rhyme_list.pkl', 'rhy_type_list':['調']}])
man.train()
print(man.alpha_list)
man.report()
man.baseline()
