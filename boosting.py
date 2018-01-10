from lib.feature import Feature
from lib.ensumble_learner import Boosting
from lib.learner import Learner

path = './data/data4.txt'
man = Boosting(path, random_state=2)
man.load_feature(funcs=[Feature.context], params=[{'k':1, 'n_gram':2}])
man.train()
print(man.alpha_list)
man.report()

print('baseline')
man = Learner(path, random_state=2)
man.load_feature(funcs=[Feature.context], params=[{'k':1, 'n_gram':2}])
man.train()
man.report()
