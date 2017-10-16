import regex as re
from tqdm import tqdm as bar

def txt_loader(path):
    return open(path, 'r').read().split('\n')[:-1]

def ngram(data, num=2):
    pattern = '.{' + str(num)  + '}'
    return [match.group() for match in re.finditer(pattern, data, overlapped=True)]

def label( text, lab_data, lab_name ):
    lab_list = ['O']*len( text )
    print('label ' + lab_name)
    for lab in bar( lab_data ):
        try:
            p = re.compile( lab )
            m = p.search( text ).span()[0]
            # check exist
            if set( lab_list[m:m+len(lab)] ) != {'O'}:
                continue
            lab_list[m] = lab_name + 'B'
            lab_list[m+len(lab)-1] = lab_name + 'E'
            for j in range(m+1, m+len(lab)-1):
                lab_list[j] = lab_name + 'I'
        except:
            continue
    return lab_list

class book:

    def feature(self):
        feature_list = []
        text_append = '@!' + self.no_comma_text + '!@'
        for i in range( 2,len( self.no_comma_text )+2 ):
            feature_list.append({
                '@-2'    : text_append[i-2],
                '@-2~-1' : text_append[i-2   : i],
                '@-1'    : text_append[i-1],
                '@-1~0'  : text_append[i-1   : i+1],
                '@0'     : text_append[i],
                '@0~1'   : text_append[i     : i+2],
                '@1'     : text_append[i+1],
                '@1~2'   : text_append[i+1   : i+3],
                '@2'     : text_append[i+2],
                'y@-1'   : self.end_label[i-2-1],
                'y@-2'   : self.end_label[i-2-2],
                'y@-3'   : self.end_label[i-2-3],
                'y@-4'   : self.end_label[i-2-4],
                'y@-5'   : self.end_label[i-2-5]
            })
        return feature_list

    def __init__(self, text):
        label = []
        for bi in ngram( text, 2 ):
            if bi[0] == '，':
                continue
            if bi[1] == '，':
                label.append('E')
            else:
                label.append('I')
        self.no_comma_text = text.replace('，','').strip()
        self.text          = text
        self.end_label     = label
        self.feature_list  = self.feature()

    def __str__(self):
        return self.text

    def __getitem__(self, i):
        return self.feature_list[i]

    def __setitem__(self, i, value):
        self.feature_list[i][value[0]] = value[1]
