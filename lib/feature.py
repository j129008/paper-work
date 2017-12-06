from collections import Counter
from lib.data import ngram
from math import log2, sqrt, pow
from tqdm import tqdm as bar
import re
import pickle
class Feature:
    def __init__(self):
        self.list = [self.context, self.t_diff, self.mi_info, self.label, self.rhyme]
    def context(param):
        text = param['text']
        k = param['k']
        n_gram = param['n_gram']
        feature_list = []
        text_append = '！'*k + text + '！'*k
        for i in range( k, len( text )+k ):
            context_feature = {}
            context_list = []
            for n in range(1, n_gram+1):
                context_list += ngram(text_append[i-k:i+k+1], num=n)
            for j, word in enumerate(context_list):
                context_feature[str(j)] = word
            feature_list.append(context_feature)
        return feature_list

    def mi_info(param):
        text = param['text']
        w_dic = Counter(text)
        bi_dic = Counter(ngram(text))
        mi_score = []
        for i in range(len(text)-1):
            Pxy = bi_dic[text[i:i+2]]/( len(text)-1 )
            Px = w_dic[text[i]]/len(text)
            Py = w_dic[text[i+1]]/len(text)
            mi_score.append( { 'mi-info':log2( Pxy/(Px*Py) ) } )
        mi_score.append( {'mi-info': 0.0} )
        return mi_score

    def t_diff(param):
        text = param['text']
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
        return t_diff_score

    def label(param):
        text = param['text']
        lab_name = param['lab_name']
        path = param['path']
        lab_data = open(path, 'r', encoding='utf-8').read().split('\n')[:-1]
        lab_list = ['O']*len( text )
        print('label ' + lab_name)
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
        return [ {lab_name:lab} for lab in lab_list ]

    def rhyme(param):
        text = param['text']
        path = param['path']
        pkl_path = param['pkl_path']
        rhy_type_list = param['rhy_type_list']
        small_rhyme = pickle.load(open(pkl_path, 'rb'))
        data = open(path,'r' , encoding='utf-8')
        rhyme_dic = dict()
        rhyme_type = ''
        print('step 1')
        for line in bar(data):
            id, word, exp = line.strip().split('|')
            if id.split('.')[1] == '1':
                rhyme_type = word
            rhyme_dic[word] = rhyme_type
        ret = []

        print('step 2')
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
        return ret
