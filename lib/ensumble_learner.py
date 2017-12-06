from sklearn.model_selection import train_test_split
from lib.learner import Learner
from collections import Counter
from queue import Queue
from threading import Thread
from sklearn.utils import resample
import numpy as np

class Bagging(Learner):
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
        self.queue.put( model.predict(X) )
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
            vote_res.append(Counter(vote).most_common(1)[0][0])
        return vote_res

class Boosting(Learner):
    def __init__(self, path):
        super().__init__(path)
    def half_hit_idx(self, y_true=None, y_pred=None):
        err_pool_idx = []
        hit_pool_idx = []
        sample_size = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                hit_pool_idx.append(i)
            else:
                err_pool_idx.append(i)
        sample_size = min(len(hit_pool_idx), len(err_pool_idx))
        res_idx = err_pool_idx[:sample_size] + hit_pool_idx[:sample_size]
        return res_idx
    def C1_C2_disagree_idx(self, C1_pred=None, C2_pred=None):
        S3_idx = []
        for i in range(len(C1_pred)):
            if C1_pred[i] != C2_pred[i]:
                S3_idx.append(i)
        return S3_idx
    def train(self, sample_size=0.5):
        n_samples = int( len(self.Y_train)*sample_size )
        self.C1 = super().train(sub_train=sample_size)
        sample_x, sample_y = resample(self.X_train, self.Y_train, n_samples=n_samples, replace=False)
        S1_X = sample_x
        S1_Y_pred = self.C1.predict(sample_x)
        S1_Y_true = sample_y
        S2_idx = self.half_hit_idx(S1_Y_pred, S1_Y_true)
        C2_X = np.array(S1_X)[S2_idx]
        C2_Y = np.array(S1_Y_true)[S2_idx]
        self.C2 = self.get_CRF()
        self.C2.fit(C2_X, C2_Y)
        sample_x, sample_y = resample(self.X_train, self.Y_train, n_samples=n_samples, replace=False)
        C1_pred = self.C1.predict(sample_x)
        C2_pred = self.C2.predict(sample_x)
        S3_idx = self.C1_C2_disagree_idx(C1_pred, C2_pred)
        S3_X = np.array(sample_x)[S3_idx]
        S3_Y = np.array(sample_y)[S3_idx]
        self.C3 = self.get_CRF()
        self.C3.fit(S3_X, S3_Y)
    def predict(self, X):
        C1_pred = self.C1.predict(X)
        C2_pred = self.C2.predict(X)
        C3_pred = self.C3.predict(X)
        final_pred = []
        for i in range(len(C1_pred)):
            if C1_pred[i] == C2_pred[i]:
                final_pred.append(C1_pred[i])
            else:
                final_pred.append(C3_pred[i])
        return final_pred
