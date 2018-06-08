from lib.ensumble_learner import Bagging, Boosting
from lib.feature import Context, MutualInfo

path = '../data/data_lite.txt'
data = Context(path) + MutualInfo(path)
data.segment(length=1)

man = Bagging(data)
man.train()
man.report()

man = Boosting(data)
man.train()
print(man.alpha_list)
man.report()
