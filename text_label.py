from lib.learner import Learner
from lib.ensumble_learner import Bagging, Boosting
from lib.feature import *
from tqdm import tqdm as bar
import csv

path = './data/data_proc.txt'
office = Label(path, lab_name='office', lab_file='./ref/known/office2.txt')
office.save('./pickles/office.pkl')

nianhao = Label(path, lab_name='nianhao', lab_file='./ref/known/nianhao.txt')
nianhao.save('./pickles/nianhao.pkl')

entry = Label(path, lab_name='entry', lab_file='./ref/known/entry1.txt')
entry.save('./pickles/entry.pkl')

address = Label(path, lab_name='address', lab_file='./ref/known/address2.txt')
address.save('./pickles/address.pkl')

name = Label(path, lab_name='name', lab_file='./ref/known/name5.txt')
name.save('./pickles/name.pkl')

context = Context(path)

compare_list = [ (office, 'office'), (nianhao, 'nianhao'), (entry, 'entry'), (address, 'address'), (name, 'name') ]

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
