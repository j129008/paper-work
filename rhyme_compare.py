from lib.learner import Learner
from lib.feature import *
from tqdm import tqdm as bar
import csv

path = './data/data_proc.txt'
result_table = csv.writer( open('/mnt/d/progress/rhyme.csv', 'w') )
context = Context(path)
type_list = ['反切', '聲母', '韻目', '調', '等', '呼', '韻母']
rhyme = Rhyme(path, './data/rhyme.txt', './pickles/rhyme_list.pkl', type_list)

for t in type_list:
    t_data = rhyme.get_feature(t)
    man = Learner(t_data + context)
    man.train(sub_train=0.5)
    man.report()
    score = man.get_score()
    result_table.writerow([t, score['P'], score['R'], score['f1']])
