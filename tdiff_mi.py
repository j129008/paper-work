from lib.learner import Learner
from lib.ensumble_learner import Bagging, Boosting
from lib.feature import *
from tqdm import tqdm as bar
import csv

path = './data/data_proc.txt'
result_table = csv.writer( open('./csv/tdiff_mi.csv', 'w') )
context = Context(path, k=5)
tdiff = Tdiff(path)
mi = MutualInfo(path)

# tdiff
man = Learner(tdiff + context)
man.train()
man.report()
score = man.get_score()
result_table.writerow(['tdiff', score['P'], score['R'], score['f1']])

# mi
man = Learner(context + mi)
man.train()
man.report()
score = man.get_score()
result_table.writerow(['mi', score['P'], score['R'], score['f1']])
