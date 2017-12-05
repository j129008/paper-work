import regex as re

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
        self.X = [dict()]*len(self.text)
        self.Y = book.label
    def feature_loader(self, funcs=None, params=None):
        for func in funcs:
            func_ret = func(*params, text=self.text)
            for i in range(len(self.text)):
                for feature_name in func_ret[i]:
                    self.X[i][feature_name] = func_ret[i][feature_name]
        return ( self.X, self.Y )
    def __getitem__(self, i):
        return ( self.text[i], self.X[i], self.Y[i] )

if __name__ == '__main__':
    data = Data('../data/data2.txt')
    print(data[0:10])
