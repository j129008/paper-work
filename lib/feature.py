from collections import Counter
from lib.data import Data
from math import log2, sqrt, pow
from gensim.models import Word2Vec
from itertools import chain
from pprint import pprint
from random import shuffle
import numpy as np
import re
import pickle
import sys
import logging
import pdb

def ngram(data, num=2):
    pattern = r'(?=(.{' + str(num)  + '}))'
    return [match.group(1) for match in re.finditer(pattern, data)]

def m_ngram(data, num=2):
    pattern = r'(?=(.{' + str(num)  + '}))'
    return [match for match in re.finditer(pattern, data)]

class Context(Data):
    def __init__(self, path, n_gram=2, k=1):
        self.k = k
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
    def __init__(self, path, k=1, n_gram=1, vec_size=50, w2v_text='./data/w2v.txt', pkl_name='./pickles/word2vec.pkl', shuffle=False):
        super().__init__(path, n_gram=n_gram, k=k)
        if shuffle == True:
            self.shuffle(seed=1)
        self.vec_size=vec_size
        self.union()
        self.x2vec(w2v_text=w2v_text, vec_file=pkl_name)
        self.y2vec()
    def genVec(self, vec_file='./pickles/word2vec.pkl', txt_file='./data/w2v.txt'):
        try:
            w2v = pickle.load(open(vec_file, 'rb'))
            if len(w2v['！']) != self.vec_size:
                print('regen w2v', len(w2v['！']), 'to', self.vec_size)
                raise 'w2v err'
        except:
            print('load:', txt_file)
            sentence = open(txt_file, 'r').read().replace('\n','').split('。')
            sentence = [ list(ele) for ele in sentence]
            sentence.append(['＊'])
            w2v = Word2Vec(sentence, min_count=1, size=self.vec_size, workers=8, iter=50)
            pickle.dump(w2v, open(vec_file, 'wb'))
        return w2v
    def x2vec(self, w2v_text='./data/w2v.txt', vec_file='./pickles/word2vec.pkl'):
        w2v = self.genVec(txt_file=w2v_text, vec_file=vec_file)
        def my_w2v(w):
            try:
                return w2v[w]
            except:
                return w2v['＊']
        self.X = np.array([ [ my_w2v(word) for word in ins.values() ] for ins in self.X ])
    def y2vec(self):
        self.Y = np.array([ 1 if ele == 'E' else 0 for ele in self.Y ])
        return self.Y
    def y2lab(y, threshold=0.5):
        return [ 'E' if ele > threshold else 'I' for ele in y ]

class BigramVecContext(VecContext):
    def __init__(self, path, k=1, min_count=100, vec_size=50, mode='tdiff'):
        self.path = path
        self.min_count = min_count
        self.mode = mode
        super().__init__(path, n_gram=2, k=k, vec_size=vec_size)
    def genBigram(self, min_count=10, txt_file='./data/w2v.txt'):
        text = open(txt_file, 'r').read()
        sentence = text.replace('\n','').replace('。','，').split('，')
        bigram = chain(*[ ngram(s) for s in sentence ])
        bigram_cnter = Counter(bigram)
        bigram_min = [ ele[0] for ele in bigram_cnter.most_common() if ele[1]>min_count ]
        return bigram_min
    def tdiffCutter(self, path):
        tdiff = Tdiff(path, uniform=False)
        context = Context(path, k=0, n_gram=1)
        data = tdiff+context
        cut_sentence = ''
        for chap in data.X:
            for ins in chap:
                if ins['t-diff']>0:
                    cut_sentence += ins['0,0']
                else:
                    cut_sentence += ins['0,0']+','
        cut_sentence = cut_sentence.replace(',,', ',')
        cut_sentence = cut_sentence.split('。')
        s_sentence = []
        for s in cut_sentence:
            s_cut = s.split(',')
            _s_cut = []
            for ele in s_cut:
                if len(ele)>2:
                    _s_cut.extend(list(ele))
                else:
                    _s_cut.append(ele)
            s_sentence.append(_s_cut)
        return s_sentence
    def textCutter(self, bigram, text):
        proc_text = text
        for b in bigram:
            proc_text = proc_text.replace(b, ','+b+',')
        proc_text = proc_text.replace(',,', ',')
        proc_text = proc_text.replace('，', '')
        proc_text = proc_text.split('。')
        proc_text = [ list(filter(None, s.split(','))) for s in proc_text ]
        ret_text = []
        for s in proc_text:
            sen = []
            for w in s:
                if len(w) == 2:
                    sen.append(w)
                else:
                    sen.extend(list(w))
            ret_text.append(sen)
        return ret_text
    def genVec(self, min_count=10, vec_file='./pickles/tdiff_word2vec.pkl', txt_file='./data/w2v.txt'):
        try:
            w2v = pickle.load(open(vec_file, 'rb'))
            return w2v
        except:
            if self.mode == 'tdiff':
                sentence = self.tdiffCutter(txt_file)
            else:
                sentence = self.textCutter(self.genBigram(), txt_file)
            sentence.append(['！'])
            w2v = Word2Vec(sentence, min_count=1, size=self.vec_size, workers=8, iter=50)
            pickle.dump(w2v, open(vec_file+'.'+str(self.min_count), 'wb'))
            return w2v
    def x2vec(self, min_count=10):
        w2v = self.genVec(min_count)
        X = []
        max_len = 0
        for ins in self.X:
            w_list = []
            f_list = sorted([ (sum([int(num) for num in pos.split(',')]), ins[pos]) for pos in ins ])
            for num, word in f_list:
                try:
                    w_list.append(w2v[word])
                except:
                    pass
            max_len = max(max_len, len(w_list))
            X.append(w_list)
        for i in range(len(X)):
            X[i].extend( [ [0]*self.vec_size ]*( max_len - len(X[i]) ) )
        self.X = X

class UniVec(VecContext):
    def __init__(self, path, k=1, vec_size=50, mode='chain', weight=None):
        super().__init__(path, k=k, vec_size=vec_size)
        if mode == 'chain':
            self.X = [ list(chain(*vec_list)) for vec_list in self.X ]
        elif mode == 'average':
            self.X = [ [ sum(u)/len(u) for u in zip(*vec_list) ] for vec_list in self.X  ]
        elif mode == 'weight':
            if weight == None:
                w = list(range(1, k+1))
                weight = w + [k] + list(reversed(w))
            vec_mul = lambda v1, v2 : [ ele[0]*ele[1] for ele in zip(v1, v2) ]
            self.X = [ [ sum(vec_mul(u, weight))/len(u) for u in zip(*vec_list) ] for vec_list in self.X  ]

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
    def __init__(self, path, uniform=True, text_file=None, noise=False):
        super().__init__(path)
        tag_name = 'mi-info'
        if text_file != None:
            _text = open(text_file, 'r').read().replace('\n', '').replace('，', '')
            w_dic = Counter(_text)
            bi_dic = Counter(ngram(_text))
        else:
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
            if noise == True:
                shuffle(mi_score)
            self.X.append(mi_score)
        if uniform == True:
            self.uniform_score()

class Tdiff(UniformScore, Data):
    def __init__(self, path, uniform=True, text_file=None, noise=False):
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
        if text_file != None:
            _text = open(text_file, 'r').read().replace('\n', '').replace('，', '')
            w_dic = Counter(_text)
            bi_dic = Counter(ngram(_text))
        else:
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
        if noise == True:
            shuffle(t_diff_score)
        i = 0
        for y in self.Y:
            self.X.append(t_diff_score[i:i+len(y)])
            i+=len(y)
        if uniform == True:
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
                lab_list[m] = 'B'
                lab_list[m+len(lab)-1] = 'E'
                for j in range(m+1, m+len(lab)-1):
                    lab_list[j] = 'I'
            except:
                continue
        lab = [ {lab_name:lab} for lab in lab_list ]
        i=0
        for y in self.Y:
            self.X.append(lab[i:i+len(y)])
            i+=len(y)

class VecLabel(Data):
    def __init__(self, path, lab_name, lab_file):
        super().__init__(path)
        text = ''.join(self.text)
        lab_data = open(lab_file, 'r', encoding='utf-8').read().split('\n')[:-1]
        lab_list = [0]*len( text )
        for lab in lab_data:
            try:
                p = re.compile( lab )
                m = p.search( text ).span()[0]
                # check exist
                if set( lab_list[m:m+len(lab)] ) != {0}:
                    continue
                lab_list[m] = 1
                lab_list[m+len(lab)-1] = 3
                for j in range(m+1, m+len(lab)-1):
                    lab_list[j] = 2
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
        logging.basicConfig(filename='rhyme.log', level=logging.DEBUG)
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
                    try:
                        exp[types] = pd_ret[types]
                    except:
                        exp[types] = pd_ret.loc[types].iloc[0]
                ret.append(exp)
            except:
                exp = {}
                logging.debug(word)
                for types in rhy_type_list:
                    exp[types] = 'O'
                ret.append(exp)
        i=0
        for y in self.Y:
            self.X.append(ret[i:i+len(y)])
            i+=len(y)
