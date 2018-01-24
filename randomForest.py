from lib.crf import CRF, WeightCRF
from lib.data import Data
from lib.learner import *
from lib.feature import Feature
from lib.ensumble_learner import Boosting
from lib.ensumble_learner import Bagging
from pprint import pprint
import csv
from datetime import datetime

start = datetime.now()

path = './data/data3.txt'

#  man = RandomForestLearner(path, random_state=2, max_dim=1000)
#  man.load_feature(funcs=[Feature.context], params=[{'k':1, 'n_gram':2}])
#  man.save_index()

print('baseline')
man = Learner(path, random_state=2)
man.load_feature(funcs=[Feature.context], params=[{'k':1, 'n_gram':2}])
man.train()
man.report()

end = datetime.now()
delta = end - start
print( delta.total_seconds()/60, 'mins' )
