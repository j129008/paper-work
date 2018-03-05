from lib.learner import Learner
from lib.ensumble_learner import Bagging, Boosting
from lib.feature import *
from tqdm import tqdm as bar
import csv

path = './data/data_proc.txt'
result_table = csv.writer( open('/mnt/d/progress/tdiff_mi.csv', 'w') )
context = Context(path)
tdiff = Tdiff(path)
mi = MutualInfo(path)

# tdiff
man = Learner(tdiff + context)
man.train(sub_train=0.5)
man.report()
score = man.get_score()
result_table.writerow(['tdiff', score['P'], score['R'], score['f1']])

# mi
man = Learner(context + mi)
man.train(sub_train=0.5)
man.report()
score = man.get_score()
result_table.writerow(['mi', score['P'], score['R'], score['f1']])
