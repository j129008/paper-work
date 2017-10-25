import regex as re
from tqdm import tqdm as bar
from collections import Counter
from math import log2, sqrt
import numpy as np

def txt_loader(path):
    return open(path, 'r', encoding='utf-8').read().split('\n')[:-1]

def ngram(data, num=2):
    pattern = '.{' + str(num)  + '}'
    return [match.group() for match in re.finditer(pattern, data, overlapped=True)]

def mi_info(txt):
    w_dic = Counter(txt)
    bi_dic = Counter(ngram(txt))
    mi_score = []
    for i in range(len(txt)-1):
        Pxy = bi_dic[txt[i:i+2]]/( len(txt)-1 )
        Px = w_dic[txt[i]]/len(txt)
        Py = w_dic[txt[i+1]]/len(txt)
        mi_score.append( log2( Pxy/(Px*Py) ) )
    mi_score.append( 0.0 )
    return mi_score

def t_diff(txt):
    # w"x"yz
    w_dic = Counter(txt)
    bi_dic = Counter(ngram(txt))
    t_diff_score = []
    sigma_dic = {}
    for i in range(len(txt)-1):
        Px = w_dic[txt[i]]/len(txt)
        Py = w_dic[txt[i+1]]/len(txt)
        Pxy = bi_dic[txt[i:i+2]]/( len(txt)-1 )
        Py_x = Pxy/Px
        try:
            sigma_dic[txt[i:i+2]].append(Py_x)
        except:
            sigma_dic[txt[i:i+2]] = []
            sigma_dic[txt[i:i+2]].append(Py_x)

    for key in sigma_dic:
        sigma_dic[key] = np.var(sigma_dic[key]) + 1

    for i in range(len(txt)-3):
        Pw = w_dic[txt[i-1]]/len(txt)
        Px = w_dic[txt[i]]/len(txt)
        Py = w_dic[txt[i+1]]/len(txt)
        Pz = w_dic[txt[i+2]]/len(txt)
        Pxy = bi_dic[txt[i:i+2]]/( len(txt)-1 )
        Pwx = bi_dic[txt[i-1:i+1]]/( len(txt)-1 )
        Pyz = bi_dic[txt[i+1:i+3]]/( len(txt)-1 )
        Py_x = Pxy/Px
        Px_w = Pwx/Pw
        Pz_y = Pyz/Py
        t_x = ( Py_x - Px_w )/sqrt(sigma_dic[txt[i:i+2]]+sigma_dic[txt[i+1:i+3]])
        t_y = ( Pz_y - Py_x )/sqrt(sigma_dic[txt[i+1:i+3]]+sigma_dic[txt[i:i+2]])
        t_diff_score.append(t_x - t_y)
    t_diff_score.extend([0, 0, 0])
    return t_diff_score

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
