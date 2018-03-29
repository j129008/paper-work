from lib.learner import Learner
from lib.feature import *
from tqdm import tqdm as bar
import csv

path = './data/data_proc.txt'
result_table = csv.writer( open('./csv/rhyme.csv', 'w') )
context = Context(path, k=5)
type_list = ['反切', '聲母', '韻目', '調', '等', '呼', '韻母']
rhyme = Rhyme(path, './data/rhyme.txt', './pickles/rhyme_list.pkl', type_list)

for t in type_list:
    t_data = rhyme.get_feature(t)
    man = Learner(t_data + context)
    man.train()
    man.report()
    score = man.get_score()
    result_table.writerow([t, score['P'], score['R'], score['f1']])
