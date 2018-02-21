from lib.learner import Learner
from lib.ensumble_learner import Bagging, Boosting
from lib.feature import *
import csv
import sys

from datetime import datetime

start = datetime.now()

path = './data/data3.txt'
learn_method = 'Basic'
print('start', learn_method)

def experiment(data, tune=False, random_state=0):
    if learn_method == 'Bagging':
        man = Bagging(data, random_state=random_state)
        tune = False
    elif learn_method == 'Boosting':
        man = Boosting(data, random_state=random_state)
        tune = False
    else:
        man = Learner(data, random_state=random_state)
    if tune == True:
        man.train_CV()
    else:
        man.train()
    man.report()
    return man.get_score()

# context
context_data = Context(path, k=1)
score = experiment(context_data)

end = datetime.now()
delta = end - start
print( delta.total_seconds() )
