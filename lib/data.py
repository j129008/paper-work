import regex as re
from tqdm import tqdm as bar

def ngram(data, num=2):
    pattern = '.{' + str(num)  + '}'
    return [match.group() for match in re.finditer(pattern, data, overlapped=True)]

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

class Book:
    def __init__(self, path):
        chapter_txt_list = open(path, 'r', encoding='utf-8').read().split('\n')[:-1]
        self.chapter_list = [ Chapter(chapter_txt) for chapter_txt in chapter_txt_list ]
        self.text = self.get_text()
        self.label = self.get_label()
    def get_text(self):
        text = ''
        for chapter in self.chapter_list:
            text += chapter.text
        return text
    def get_label(self):
        label_list = []
        for chapter in self.chapter_list:
            label_list.extend(chapter.label)
        return label_list
    def __getitem__(self, i):
        return self.chapter_list[i]

class Data:
    def __init__(self, path):
        book = Book(path)
        self.chaper_list = book.chapter_list
        self.text = book.text
        self.Y = book.label
        self.X = [{} for _ in range(len(self.Y))]
    def copyer(self, func_ret):
        for i in range(len(self.text)):
            for feature_name in func_ret[i]:
                self.X[i][feature_name] = func_ret[i][feature_name]
    def load_feature(self, funcs=None, params=None):
        self.funcs = funcs
        self.params= params
        for func_i, func in enumerate(funcs):
            params[func_i]['text'] = self.text
            func_ret = func(params[func_i])
            self.copyer(func_ret)
        return self.X
    def resize(self, persent):
        resize = int( len(self.X)*persent )
        self.X = self.X[:resize]
        self.Y = self.Y[:resize]
        self.text = self.text[:resize]
    def __getitem__(self, i):
        return ( self.text[i], self.X[i], self.Y[i] )

