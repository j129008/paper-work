from collections import Counter
from lib.data import Data
from math import log2, sqrt, pow
from gensim.models import Word2Vec
from itertools import chain
from pprint import pprint
import numpy as np
import re
import pickle
import sys
import pdb

def ngram(data, num=2):
    pattern = r'(?=(.{' + str(num)  + '}))'
    return [match.group(1) for match in re.finditer(pattern, data)]

def m_ngram(data, num=2):
    pattern = r'(?=(.{' + str(num)  + '}))'
    return [match for match in re.finditer(pattern, data)]

class Context(Data):
    def __init__(self, path, n_gram=2, k=1):
        super().__init__(path)
        for chap_text in self.text:
            text_append = '！'*k + chap_text + '！'*k
            chap_feature_list = []
            for i in range( k, len( chap_text )+k ):
                context_feature = {}
                match_list = []
                for n in range(1, n_gram+1):
                    match_list += m_ngram(text_append[i-k:i+k+1], num=n)
                for i, match in enumerate(match_list):
                    start = match.start(1) - k
                    end = match.end(1) - k-1
                    context_feature[str(start)+','+str(end)] = match.group(1)
                chap_feature_list.append(context_feature)
            self.X.append(chap_feature_list)

class VecContext(Context):
    def __init__(self, path, k=1, n_gram=1):
        super().__init__(path, n_gram=n_gram, k=k)
        self.union()
        self.x2vec()
        self.y2vec()
    def genVec(self, vec_file='./pickles/word2vec.pkl', txt_file='./data/w2v.txt'):
        try:
            w2v = pickle.load(open(vec_file, 'rb'))
        except:
            sentence = open(txt_file, 'r').read().replace('\n','').split('。')
            sentence = [ list(ele) for ele in sentence]
            sentence.append(['！'])
            w2v = Word2Vec(sentence, min_count=1, size=30, workers=8, iter=50)
            pickle.dump(w2v, open(vec_file, 'wb'))
        return w2v
    def x2vec(self):
        w2v = self.genVec()
        self.X = np.array([ [ w2v[word] for word in ins.values() ] for ins in self.X ])
    def y2vec(self):
        self.Y = np.array([ 1 if ele == 'E' else 0 for ele in self.Y ])
        return self.Y
    def y2lab(self, y):
        return [ 'E' if ele > 0.5 else 'I' for ele in y ]

class BigramVecContext(VecContext):
    def __init__(self, path, k=1, min_count=100):
        super().__init__(path, n_gram=2, k=k)
    def genBigram(self, min_count=10, txt_file='./data/w2v.txt'):
        text = open(txt_file, 'r').read()
        sentence = text.replace('\n','').replace('。','，').split('，')
        bigram = chain(*[ ngram(s) for s in sentence ])
        bigram_cnter = Counter(bigram)
        bigram_min = [ ele[0] for ele in bigram_cnter.most_common() if ele[1]>min_count ]
        return bigram_min
    def textCutter(self, bigram, text):
        proc_text = text
        for b in bigram:
            proc_text.replace(b, ','+b+',')
        proc_text = proc_text.split('。')
        proc_text = [ s.split(',') for s in proc_text ]
        return proc_text
    def genVec(self, min_count=10, vec_file='./pickles/bigram_word2vec.pkl', txt_file='./data/w2v.txt'):
        try:
            w2v = pickle.load(open(vec_file, 'rb'))
        except:
            bigram = self.genBigram(min_count=min_count)
            text = open(txt_file, 'r').read().replace('\n','')
            sentence = self.textCutter(bigram, text)
            sentence = [ word if len(word)==2 else list(word) for word in sentence]
            sentence.append(['！'])
            w2v = Word2Vec(sentence, min_count=1, size=30, workers=8, iter=50)
            pickle.dump(w2v, open(vec_file, 'wb'))
        return w2v
    def x2vec(self, min_count=10):
        w2v = self.genVec(min_count)
        X = []
        for ins in self.X:
            miss_word = 0
            w_list = []
            for word in ins.values():
                try:
                    w_list.append(w2v[word])
                except:
                    miss_word += 1
            w2v_size = 30
            w_list.extend( [ [0]*w2v_size ]*miss_word )
            X.append(w_list)
        self.X = np.array(X)

class UniVec(VecContext):
    def __init__(self, path):
        super().__init__(path, k=1)
        self.X = [ list(chain(*vec_list)) for vec_list in self.X ]

class UniformScore:
    def uniform_score(self):
        uni = lambda data: [ ins for chap in data for ins in chap ]
        score = np.array(sorted([list(ele.values())[0] for ele in uni(self.X)]))
        threshold_list = score[[ int(i*len(score)/10) for i in range(10)]+[-1]]
        for chap_i in range(len(self.X)):
            for ins_i in range(len(self.X[chap_i])):
                key = list(self.X[chap_i][ins_i].keys())[0]
                value = self.X[chap_i][ins_i][key]
                i = 0
                for threshold_i in range(len(threshold_list)-1):
                    i+=1
                    if value >= threshold_list[threshold_i] and value <= threshold_list[threshold_i+1]:
                        self.X[chap_i][ins_i][key] = str(i)
                        break

class MutualInfo(UniformScore, Data):
    def __init__(self, path):
        super().__init__(path)
        tag_name = 'mi-info'
        w_dic = Counter(''.join(self.text))
        bi_dic = Counter([ gram for chap_text in self.text for gram in ngram(chap_text)])
        for chap_text in self.text:
            mi_score = []
            for i in range(len(chap_text)-1):
                Pxy = max(bi_dic[chap_text[i:i+2]]/( len(chap_text)-1 ), sys.float_info.min)
                Px = w_dic[chap_text[i]]/len(chap_text)
                Py = w_dic[chap_text[i+1]]/len(chap_text)
                mi = log2( Pxy/(Px*Py))
                mi_score.append( { tag_name: mi } )
            mi_score.append( { tag_name: 0.0 } )
            self.X.append(mi_score)
        self.uniform_score()

class Tdiff(UniformScore, Data):
    def __init__(self, path):
        super().__init__(path)
        text = ''.join(self.text)
        def t_test(f_xy, f_yz, f_x, f_y, v):
            return ( ( f_yz+0.5 )/( f_y + v/2 ) - ( f_xy+0.5 )/( f_x + v/2 ) )/sqrt( ( f_yz+0.5 )/pow(f_y + v/2, 2) + ( f_xy+0.5 )/pow(f_x + v/2, 2) )

        def v_calc(bigram_list):
            v_dic = {}
            for bi in bigram_list:
                try:
                    v_dic[bi[1]].add(bi[0])
                except:
                    try:
                        v_dic[bi[1]] = set()
                        v_dic[bi[1]].add(bi[0])
                    except:
                        pdb.set_trace()
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
        t_diff_score.extend([{'t-diff': 0.0}, {'t-diff': 0.0}, {'t-diff': 0.0}])
        i = 0
        for y in self.Y:
            self.X.append(t_diff_score[i:i+len(y)])
            i+=len(y)
        self.uniform_score()

class Label(Data):
    def __init__(self, path, lab_name, lab_file):
        super().__init__(path)
        text = ''.join(self.text)
        lab_data = open(lab_file, 'r', encoding='utf-8').read().split('\n')[:-1]
        lab_list = ['O']*len( text )
        for lab in lab_data:
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
        lab = [ {lab_name:lab} for lab in lab_list ]
        i=0
        for y in self.Y:
            self.X.append(lab[i:i+len(y)])
            i+=len(y)

class Rhyme(Data):
    def __init__(self, path, index_file, db_file, rhy_type_list):
        super().__init__(path)
        text = ''.join(self.text)
        small_rhyme = pickle.load(open(db_file, 'rb'))
        data = open(index_file, 'r', encoding='utf-8')
        rhyme_dic = dict()
        rhyme_type = ''
        for line in data:
            id, word, exp = line.strip().split('|')
            if id.split('.')[1] == '1':
                rhyme_type = word
            rhyme_dic[word] = rhyme_type
        ret = []

        for word in text:
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
        i=0
        for y in self.Y:
            self.X.append(ret[i:i+len(y)])
            i+=len(y)
