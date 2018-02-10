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
    def __len__(self):
        return len(self.label)

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

class Data(Book):
    def __init__(self, path):
        super().__init__(path)
        self.Y = self.label
        self.X = [{} for _ in range(len(self.Y))]
    def __add__(self, other):
        self.X = [ { **self.X[i], **other.X[i] } for i in range(len(self.text)) ]
        return self
