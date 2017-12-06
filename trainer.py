from lib.crf import CRF
from lib.data import Data
from lib.learner import Learner
from lib.feature import Feature
from lib.ensumble_learner import Boosting
from lib.ensumble_learner import Bagging
from pprint import pprint

for x in range(1, 8):
    man = Boosting('./data/data2.txt')
    man.load_feature(funcs=[Feature.context, Feature.t_diff, Feature.mi_info], params=[{'k':x, 'n_gram':2}, {}, {}])
    man.train()
    print('context: ', x)
    pprint(man.get_score())
