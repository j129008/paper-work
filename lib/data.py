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
    def feature_loader(self, funcs=None, params=None):
        self.funcs = funcs
        self.params= params
        for func_i, func in enumerate(funcs):
            func_ret = func(*params[func_i], text=self.text)
            self.copyer(func_ret)
        return self.X
    def resize(self, persent):
        resize = int( len(self.X)*persent )
        self.X = self.X[:resize]
        self.Y = self.Y[:resize]

    def crfSuite_transform(self, train_size=0.6, train_file='./train.txt', test_file='./test.txt'):
        key_list = sorted( list( self.X[0].keys() ) )
        n_train = int( len(self.X)*train_size )
        data_pool = []
        print('transform to crfSuite:')
        for i in bar( range(len(self.X)) ):
            instance = []
            for key in key_list:
                instance.append( str( self.Y[i] ) )
                instance.append( str( self.X[i][key] ) )
            data_pool.append( '\t'.join( instance ) )
        print('write to file: ', train_file, test_file)
        open(train_file, 'w').write('\n'.join(data_pool[:n_train]))
        open(test_file, 'w').write('\n'.join(data_pool[n_train:]))

    def __getitem__(self, i):
        return ( self.text[i], self.X[i], self.Y[i] )

