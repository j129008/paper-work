from lib.feature import *
from pprint import pprint

path = '../data/data_lite.txt'

office = Label(path, lab_name='office', lab_file='./ref/tang_name/tangOffice.clliu.txt')
nianhao = Label(path, lab_name='nianhao', lab_file='./ref/tang_name/tangReignperiods.clliu.txt')
address = Label(path, lab_name='address', lab_file='./ref/tang_name/tangAddresses.clliu.txt')
compare_list = [ (office, 'office'), (nianhao, 'nianhao'), (address, 'address') ]

print(office.text[0][:10])
for data, data_name in compare_list:
    print('tag name:', data_name)
    pprint(data.X[0][:10])
