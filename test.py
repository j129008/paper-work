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

path = './data/data_small.txt'

context = Context(path)
man = Learner(context)
man.train()
demo = Demo(learner=man)
