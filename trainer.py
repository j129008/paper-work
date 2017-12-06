from lib.crf import CRF
from lib.data import Data
from lib.learner import Learner
from lib.feature import Feature
from lib.ensumble_learner import Boosting

man = Learner('./data/data2.txt')
man.load_feature(funcs=[Feature.context], params=[{'k':2, 'n_gram':2}])
man.train()
man.report()
