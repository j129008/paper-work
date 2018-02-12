from lib.feature import Context
from lib.ensumble_learner import Boosting
from lib.learner import Learner
from datetime import datetime

start = datetime.now()

path = './data/data3.txt'
data = Context(path)
man = Boosting(data)
man.train()
print(man.alpha_list)
man.report()

print('baseline')
man = Learner(data)
man.train()
man.report()

end = datetime.now()
delta = end - start
print( delta.total_seconds() )
