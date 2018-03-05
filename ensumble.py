from lib.learner import Learner
from lib.ensumble_learner import Bagging, Boosting
from lib.feature import *
from tqdm import tqdm as bar
from sklearn.utils import resample
import numpy as np
import csv

path = './data/data_proc.txt'
result_table = csv.writer( open('/mnt/d/progress/ensumble.csv', 'w') )

context = Context(path)
man = Bagging(context)
man.train()
man.report()
score = man.get_score()
result_table.writerow(['bagging', score['P'], score['R'], score['f1']])

context = Context(path)
context.segment()
man = Boosting(context)
man.train()
print(man.alpha_list)
man.report()
score = man.get_score()
result_table.writerow(['boosting', score['P'], score['R'], score['f1']])

context = Context(path)
man = Learner(context)
man.train()
man.report()
score = man.get_score()
result_table.writerow(['baseline', score['P'], score['R'], score['f1']])
