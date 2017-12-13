from sklearn.model_selection import train_test_split
from lib.learner import Learner, RandomLearner, WeightLearner
from collections import Counter
from queue import Queue
from threading import Thread
from sklearn.utils import resample
import numpy as np
from math import sqrt, log
import pdb

class Bagging(RandomLearner):
    def __init__(self, path):
        super().__init__(path)
        self.model_list = []
        self.queue = Queue()
    def gen_model(self, train_size, c1, c2):
        self.queue.put( super().train(sub_train=train_size, c1=c1, c2=c2) )
    def train(self, voter=2, train_size=0.05, c1=0, c2=1):
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
    def __init__(self, path):
        super().__init__(path)

    def sigma_error_weight(self, Y_pred, Y_private):
        sum_of_error_weight = 0.0
        for i in range(len(Y_pred)):
            if Y_pred[i] != Y_private[i]:
                sum_of_error_weight += self.weight_list[i]
        return sum_of_error_weight

    def update_weight(self):
        Y_pred = super().predict(self.X_train)
        Y_private = self.Y_train
        epsilon = self.sigma_error_weight(Y_pred, Y_private)/sum(self.weight_list)
        t = sqrt((1-epsilon)/epsilon)
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y_private[i]:
                self.weight_list[i]/=t
            else:
                self.weight_list[i]*=t
        alpha = log(t)
        return alpha

    def train(self, n_model=3):
        self.model_list = []
        self.alpha_list = []
        for i in range(n_model):
            self.model_list.append( super().train() )
            self.alpha_list.append( self.update_weight() )

    def predict(self, x):
        predict_res = []
        predict_list = []
        for model in self.model_list:
            predict_list.append( model.predict_prob(x) )
        predict_list = [*zip(*predict_list)]
        for item in predict_list:
            res = sum( [ a*b for a, b in zip(item, self.alpha_list) ] )
            if res > 0.5:
                predict_res.append('E')
            else:
                predict_res.append('I')
        return predict_res
