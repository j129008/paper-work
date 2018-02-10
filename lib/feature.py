from collections import Counter
from lib.data import *
from math import log2, sqrt, pow
from tqdm import tqdm as bar
import re
import pickle

class Context(Data):
    def __init__(self, path, n_gram=2, k=1):
        super().__init__(path)
        feature_list = []
        text_append = '！'*k + self.text + '！'*k
        for i in range( k, len( self.text )+k ):
            context_feature = {}
            context_list = []
            for n in range(1, n_gram+1):
                context_list += ngram(text_append[i-k:i+k+1], num=n)
            for j, word in enumerate(context_list):
                context_feature[str(j)] = word
            feature_list.append(context_feature)
        self.X = feature_list

class MutualInfo(Data):
    def __init__(self, path):
        super().__init__(path)
        text = self.text
        w_dic = Counter(text)
        bi_dic = Counter(ngram(text))
        mi_score = []
        for i in range(len(text)-1):
            Pxy = bi_dic[text[i:i+2]]/( len(text)-1 )
            Px = w_dic[text[i]]/len(text)
            Py = w_dic[text[i+1]]/len(text)
            mi_score.append( { 'mi-info':log2( Pxy/(Px*Py) ) } )
        mi_score.append( {'mi-info': 0.0} )
        self.X = mi_score

class Tdiff(Data):
    def __init__(self, path):
        super().__init__(path)
        text = self.text
        def t_test(f_xy, f_yz, f_x, f_y, v):
            return ( ( f_yz+0.5 )/( f_y + v/2 ) - ( f_xy+0.5 )/( f_x + v/2 ) )/sqrt( ( f_yz+0.5 )/pow(f_y + v/2, 2) + ( f_xy+0.5 )/pow(f_x + v/2, 2) )

        def v_calc(bigram_list):
            v_dic = {}
            for bi in bigram_list:
                try:
                    v_dic[bi[1]].add(bi[0])
                except:
                    v_dic[bi[1]] = set()
                    v_dic[bi[1]].add(bi[0])
            for key in v_dic:
                v_dic[key] = len(v_dic[key])
            return v_dic
        # wxyz
        w_dic = Counter(text)
        bi_dic = Counter(ngram(text))
        bi_list = [ ele for ele in bi_dic ]
        v_dic = v_calc(bi_list)
        t_diff_score = []

        for i in range(len(text)-3):
            w_w = text[i-1]
            w_x = text[i]
            w_y = text[i+1]
            w_z = text[i+2]
            f_w = w_dic[w_w]
            f_x = w_dic[w_x]
            f_y = w_dic[w_y]
            f_z = w_dic[w_z]
            f_xy = bi_dic[text[i:i+2]]
            f_wx = bi_dic[text[i-1:i+1]]
            f_yz = bi_dic[text[i+1:i+3]]
            t_x = t_test(f_wx, f_xy, f_w, f_y, v_dic[w_w] + v_dic[w_x] )
            t_y = t_test(f_xy, f_yz, f_x, f_y, v_dic[w_x] + v_dic[w_y] )
            t_diff_score.append({'t-diff': t_x - t_y})
        t_diff_score.extend([{'t-diff': 0}]*3)
        self.X = t_diff_score

class Label(Data):
    def __init__(self, path, lab_name, lab_file):
        super().__init__(path)
        text = self.text
        lab_data = open(lab_file, 'r', encoding='utf-8').read().split('\n')[:-1]
        lab_list = ['O']*len( text )
        for lab in bar( lab_data ):
            try:
                p = re.compile( lab )
                m = p.search( text ).span()[0]
                # check exist
                if set( lab_list[m:m+len(lab)] ) != {'O'}:
                    continue
                lab_list[m] = lab_name + '-B'
                lab_list[m+len(lab)-1] = lab_name + '-E'
                for j in range(m+1, m+len(lab)-1):
                    lab_list[j] = lab_name + '-I'
            except:
                continue
        self.X = [ {lab_name:lab} for lab in lab_list ]

class Rhyme(Data):
    def __init__(self, path, index_file, db_file, rhy_type_list):
        super().__init__(path)
        text = self.text
        small_rhyme = pickle.load(open(db_file, 'rb'))
        data = open(index_file, 'r', encoding='utf-8')
        rhyme_dic = dict()
        rhyme_type = ''
        print('load index')
        for line in bar(data):
            id, word, exp = line.strip().split('|')
            if id.split('.')[1] == '1':
                rhyme_type = word
            rhyme_dic[word] = rhyme_type
        ret = []

        print('load db')
        for word in bar(text):
            try:
                pd_ret = small_rhyme[rhyme_dic[word]]
                exp = {}
                for types in rhy_type_list:
                    exp[types] = pd_ret[types]
                ret.append(exp)
            except:
                exp = {}
                for types in rhy_type_list:
                    exp[types] = 'O'
                ret.append(exp)
        self.X = ret
