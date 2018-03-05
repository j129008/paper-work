from sklearn.model_selection import train_test_split
from lib.learner import *
from collections import Counter
from queue import Queue
from threading import Thread
from sklearn.utils import resample
import numpy as np
from math import sqrt, log
import sys
import pdb

class EnsumbleTool:
    def get_gap(self):
        score_list = self.predict_score(self.X_train)
        score = [ score for chap_score in score_list for score in chap_score]
        max_score = max(score)
        min_score = min(score)
        f1_dic = dict()
        meature = 1000
        threshold_gap = (max_score - min_score)/meature
        threshold = min_score
        for _ in range(meature):
            threshold += threshold_gap
            train_pred = self.score2lab(threshold, score_list)
            f1 = self.get_score(train_pred, self.Y_train)['f1']
            f1_dic[f1] = threshold
        return f1_dic[max(f1_dic)]
    def score2lab(self, gap, score_list):
        res_list = []
        for chap_score in score_list:
            chap_res = []
            for score in chap_score:
                if score >= gap:
                    chap_res.append('E')
                else:
                    chap_res.append('I')
            res_list.append(chap_res)
        return res_list
    def predict(self, x):
        print('ensumble predict')
        score_list = self.predict_score(x)
        return self.score2lab(self.gap, score_list)

class Bagging(EnsumbleTool, RandomLearner):
    def __init__(self, data, random_state=None):
        super().__init__(data, random_state=random_state)
        self.model_list = []
        self.queue = Queue()
    def gen_model(self, train_size, c1, c2):
        self.queue.put( super().train(sub_train=train_size, c1=c1, c2=c2) )
    def train(self, voter=8, train_size=0.4, c1=0, c2=1):
        pool = []
        for i in range(voter):
            thread = Thread(
                target=self.gen_model,
                args=[train_size, c1, c2]
            )
            thread.start()
            pool.append(thread)
        for thread in pool:
            thread.join()
        for _ in range(voter):
            self.model_list.append(self.queue.get())
        self.gap = self.get_gap()

    def gen_predict(self, model, X):
        self.queue.put( model.predict_prob(X) )
    def predict_score(self, X):
        predict_res = []
        vote_res = []
        pool = []
        for model in self.model_list:
            thread = Thread(
                target=self.gen_predict,
                args=[model, X]
            )
            thread.start()
            pool.append(thread)
        for thread in pool:
            thread.join()
        for _ in range(len(pool)):
            predict_res.append(self.queue.get())
        predict_res = zip(*predict_res)
        for chap_pred in predict_res:
            pred_merge = zip(*chap_pred)
            chap_res = []
            for vote in pred_merge:
                score_list = [ v['E'] for v in vote ]
                prob_avg = sum(score_list)/len(score_list)
                chap_res.append(prob_avg)
            vote_res.append(chap_res)
        return vote_res

class Boosting(EnsumbleTool, WeightLearner):
    def __init__(self, data, random_state=None):
        super().__init__(data, random_state=random_state)

    def sigma_error_weight(self, Y_pred, Y_private):
        sum_of_error_weight = 0.0
        for i in range(len(Y_pred)):
            if Y_pred[i] != Y_private[i]:
                sum_of_error_weight += self.weight_list[i]
        return sum_of_error_weight

    def update_weight(self):
        Y_pred = self.crf.predict(self.X_train)
        Y_private = self.Y_train
        epsilon = max(self.sigma_error_weight(Y_pred, Y_private)/sum(self.weight_list), sys.float_info.min)
        t = sqrt((1-epsilon)/epsilon)
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y_private[i]:
                self.weight_list[i]/=t
            else:
                self.weight_list[i]*=t
        alpha = log(t)
        return alpha

    def train(self, max_model=10):
        self.model_list = []
        self.alpha_list = []
        for i in range(max_model):
            self.model = super().train()
            alpha = self.update_weight()
            if alpha > 0 and self.model != None:
                self.model_list.append( self.model )
                self.alpha_list.append( alpha )
                if alpha > 20:
                    break
            else:
                break
        if len(self.model_list) == 0:
            self.model_list = [self.model]
            self.alpha_list = [1.0]
        self.gap = self.get_gap()

    def predict_score(self, x):
        predict_res = []
        predict_list = []
        predict_score = []
        for model in self.model_list:
            predict_list.append( [ [ ins['E'] for ins in chap ] for chap in model.predict_prob(x) ] )
        predict_list = [*zip(*predict_list)]
        for merge_chap in predict_list:
            chap_score = []
            for merge_ins in zip(*merge_chap):
                chap_score.append(sum([a*b for a, b in zip(merge_ins, self.alpha_list)]))
            predict_score.append(chap_score)
        return predict_score
