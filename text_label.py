from lib.learner import Learner
from lib.ensumble_learner import Bagging, Boosting
from lib.feature import *
from tqdm import tqdm as bar
import csv

path = './data/data_proc.txt'
office = Label(path, lab_name='office', lab_file='./ref/tang_name/tangOffice.clliu.txt')
office.save('./pickles/office.pkl')

nianhao = Label(path, lab_name='nianhao', lab_file='./ref/tang_name/tangReignperiods.clliu.txt')
nianhao.save('./pickles/nianhao.pkl')

address = Label(path, lab_name='address', lab_file='./ref/tang_name/tangAddresses.clliu.txt')
address.save('./pickles/address.pkl')

context = Context(path, k=5)

compare_list = [ (office, 'office'), (nianhao, 'nianhao'), (address, 'address') ]

result_table = csv.writer( open('text_label.csv', 'w') )
for data, data_name in bar(compare_list):
    man = Learner(data + context)
    man.train()
    man.report()
    score = man.get_score()
    result_table.writerow([data_name, score['P'], score['R'], score['f1']])

# baseline
man = Learner(context)
man.train()
man.report()
score = man.get_score()
result_table.writerow(['baseline', score['P'], score['R'], score['f1']])
