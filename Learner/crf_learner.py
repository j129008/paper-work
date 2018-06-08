from lib.learner import Learner
from lib.feature import Context, MutualInfo

path = '../data/data_lite.txt'
data = Context(path) + MutualInfo(path)
man = Learner(data)

man.train(c1=0, c2=1, sub_train=0.2)
man.report()

man.train_CV(cv=3, n_iter=1)
man.report()
