from tqdm import tqdm as bar
from copy import deepcopy
import pickle
import re

def ngram(data, num=2):
    pattern = r'(?=(.{' + str(num)  + '}))'
    return [match.group(1) for match in re.finditer(pattern, data)]

class Chapter:
    def __init__(self, text):
        label = []
        for bi in ngram( text, 2 ):
            if bi[0] == '，':
                continue
            if bi[1] == '，':
                label.append('E')
            else:
                label.append('I')
        self.text  = text.replace('，','').strip()
        self.label = label

class Data:
    def __init__(self, path):
        RAW_txt_list = open(path, 'r', encoding='utf-8').read().split('\n')[:-1]
        chapter_list = [ Chapter(RAW_txt) for RAW_txt in RAW_txt_list ]
        self.text = [ chap.text for chap in chapter_list ]
        self.Y = [ chap.label for chap in chapter_list ]
        self.X = []
    def __add__(self, other):
        self_copy = deepcopy(self)
        for chap_i in range(len(self_copy.X)):
            self_copy.X[chap_i] = [ { **self.X[chap_i][i], **other.X[chap_i][i] } for i in range(len(self.X[chap_i])) ]
        return self_copy
    def save(self, file_name):
        pickle.dump(self, open(file_name, 'wb'))
    def load(self, file_name):
        data = pickle.load(open(file_name, 'rb'))
        self.X = data.X
        self.Y = data.Y
