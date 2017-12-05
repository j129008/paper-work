from sklearn.model_selection import train_test_split
from lib.learner import Learner
from collections import Counter
from queue import Queue
from threading import Thread

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
    def gen_predict(self, model):
        self.queue.put( model.predict(self.X_private) )
    def report(self):
        predict_res = []
        vote_res = []
        pool = []
        for model in self.model_list:
            thread = Thread(
                target=self.gen_predict,
                args=[model]
            )
            thread.start()
            pool.append(thread)
        for thread in pool:
            thread.join()
            predict_res.append(self.queue.get())
        predict_res = zip(*predict_res)
        for vote in predict_res:
            vote_res.append(Counter(vote).most_common(1)[0][0])
        self.Y_pred = vote_res
        super().report()

class Boosting(Learner):
    def __init__(self, path):
        super().__init__(path)
    def train(self, C1_size=0.5):
        self.model_C1 = super().train(sub_train=C1_size)
        self.sub_y_pred = self.model_C1.predict(self.sub_x)
        err_pool = []
        hit_pool = []
        C2_pool = []
        for i in range(len(self.sub_y_pred)):
            if self.sub_y_pred[i] == self.sub_y[i]:
                hit_pool.append(i)
            else:
                err_pool.append(i)
            if len(err_pool) > len(hit_pool):

