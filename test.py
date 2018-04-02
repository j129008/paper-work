from lib.data import Data
from lib.learner import Learner
from lib.feature import *
from lib.ensumble_learner import Bagging, Boosting
from lib.metric import *
from lib.metric import Demo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics
from pprint import pprint

path = './data/budda_proc.txt'
context = Context(path, k=5)
tdiff = Tdiff(path)
pmi = MutualInfo(path)
man = Learner(context+tdiff+pmi)
man.train()
man.report()
demo = Demo(file_name='budda_demo.txt', learner=man)
