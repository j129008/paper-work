from glob import glob
from pprint import pprint
import re

files = glob('./*.txt')

for f_name in files:
    text = open(f_name, 'r').read()
    chap = ''.join(re.findall(r'[^\n。]+。', text))
    print(chap, file=open('./valid.out', 'a'))
