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

class Bagging(RandomLearner):
    def __init__(self, data, random_state):
        super().__init__(data, random_state=random_state)
        self.model_list = []
        self.queue = Queue()
    def gen_model(self, train_size, c1, c2):
        self.queue.put( super().train(sub_train=train_size, c1=c1, c2=c2) )
    def train(self, voter=8, train_size=0.1, c1=0, c2=1):
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
            self.model_list.append(self.queue.get())
    def gen_predict(self, model, X):
        self.queue.put( model.predict_prob(X) )
    def predict(self, X):
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
            predict_res.append(self.queue.get())
        predict_res = zip(*predict_res)
        for vote in predict_res:
            prob_avg = sum(vote)/len(vote)
            if prob_avg > 0.5:
                vote_res.append('E')
            else:
                vote_res.append('I')
        return vote_res

class Boosting(WeightLearner):
    def __init__(self, data, random_state):
        super().__init__(data, random_state=random_state)

    def sigma_error_weight(self, Y_pred, Y_private):
        sum_of_error_weight = 0.0
        for i in range(len(Y_pred)):
            if Y_pred[i] != Y_private[i]:
                sum_of_error_weight += self.weight_list[i]
        return sum_of_error_weight

    def update_weight(self):
        Y_pred = super().predict(self.X_train)
        Y_private = self.Y_train
        epsilon = self.sigma_error_weight(Y_pred, Y_private)/sum(self.weight_list) + sys.float_info.min
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
            model = super().train()
            alpha = self.update_weight()
            if alpha > 0 and model != None:
                self.model_list.append( model )
                self.alpha_list.append( alpha )
            else:
                break

    def predict_score(self, x):
        predict_res = []
        predict_list = []
        predict_score = []
        for model in self.model_list:
            predict_list.append( model.predict_prob(x) )
        predict_list = [*zip(*predict_list)]
        for item in predict_list:
            score = sum( [ a*b for a, b in zip(item, self.alpha_list) ] )
            predict_score.append(score)
        return predict_score

    def score2lab(self, gap, score_list):
        res_list = []
        for score in score_list:
            if score >= gap:
                res_list.append('E')
            else:
                res_list.append('I')
        return res_list

    def get_gap(self):
        score_list = self.predict_score(self.X_train)
        end_list = []
        for i, lab in enumerate( self.Y_train ):
            if lab == 'E':
                end_list.append(score_list[i])
        gap = sum(end_list)/len(end_list)
        f1 = 0.0
        while True:
            score_dic = self.get_score(self.score2lab(gap, score_list), self.Y_train)
            print(score_dic)
            _f1 = score_dic['f1']
            if _f1 > f1:
                f1 = _f1
                gap -= 1.0
            else:
                return gap+1.0

    def predict(self, x):
        score_list = self.predict_score(x)
        gap = self.get_gap()
        return self.score2lab(gap, score_list)

