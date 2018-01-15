from lib.feature import Feature
from lib.ensumble_learner import Boosting
from lib.learner import Learner
from datetime import datetime

start = datetime.now()

path = './data/data3.txt'
man = Boosting(path, random_state=0)
man.load_feature(funcs=[Feature.context], params=[{'k':1, 'n_gram':2}])
man.train()
print(man.alpha_list)
man.report()

print('baseline')
man = Learner(path, random_state=0)
man.load_feature(funcs=[Feature.context], params=[{'k':1, 'n_gram':2}])
man.train()
man.report()

end = datetime.now()
delta = end - start
print( delta.total_seconds() )
