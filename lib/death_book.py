import regex as re
from tqdm import tqdm as bar
from collections import Counter
from math import log2, sqrt, pow
import numpy as np
import pickle

def txt_loader(path):
    return open(path, 'r', encoding='utf-8').read().split('\n')[:-1]

def ngram(data, num=2):
    pattern = '.{' + str(num)  + '}'
    return [match.group() for match in re.finditer(pattern, data, overlapped=True)]

def rhyme(txt, rhy_type, path='./data/rhyme.txt', pkl_path='./pickles/rhyme_list.pkl'):
    small_rhyme = pickle.load(open(pkl_path, 'rb'))
    data = open(path,'r' , encoding='utf-8')
    rhyme_dic = dict()
    rhyme_type = ''
    for line in data:
        id, word, exp = line.strip().split('|')
        if id.split('.')[1] == '1':
            rhyme_type = word
        rhyme_dic[word] = rhyme_type
    ret = []
    for word in txt:
        try:
            ret.append(small_rhyme[rhyme_dic[word]][rhy_type])
        except:
            ret.append(word)
    return ret


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

def t_test(f_xy, f_yz, f_x, f_y, v):
    return ( ( f_yz+0.5 )/( f_y + v/2 ) - ( f_xy+0.5 )/( f_x + v/2 ) )/sqrt( ( f_yz+0.5 )/pow(f_y + v/2, 2) + ( f_xy+0.5 )/pow(f_x + v/2, 2) )

def v_calc(bigram_list):
    v_dic = {}
    print('mapping v dic')
    for bi in bar(bigram_list):
        try:
            v_dic[bi[1]].add(bi[0])
        except:
            v_dic[bi[1]] = set()
            v_dic[bi[1]].add(bi[0])
    for key in v_dic:
        v_dic[key] = len(v_dic[key])
    return v_dic

def t_diff(txt):
    # wxyz
    w_dic = Counter(txt)
    bi_dic = Counter(ngram(txt))
    bi_list = [ ele for ele in bi_dic ]
    v_dic = v_calc(bi_list)
    t_diff_score = []

    for i in range(len(txt)-3):
        w_w = txt[i-1]
        w_x = txt[i]
        w_y = txt[i+1]
        w_z = txt[i+2]
        f_w = w_dic[w_w]
        f_x = w_dic[w_x]
        f_y = w_dic[w_y]
        f_z = w_dic[w_z]
        f_xy = bi_dic[txt[i:i+2]]
        f_wx = bi_dic[txt[i-1:i+1]]
        f_yz = bi_dic[txt[i+1:i+3]]
        t_x = t_test(f_wx, f_xy, f_w, f_y, v_dic[w_w] + v_dic[w_x] )
        t_y = t_test(f_xy, f_yz, f_x, f_y, v_dic[w_x] + v_dic[w_y] )
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
