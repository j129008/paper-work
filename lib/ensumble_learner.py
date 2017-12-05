from sklearn.model_selection import train_test_split
from lib.learner import Learner
from collections import Counter
from queue import Queue
from threading import Thread
from tqdm import tqdm as bar

class Bagging(Learner):
    def __init__(self, path):
        super().__init__(path)
        self.model_list = []
        self.queue = Queue()
    def gen_model(self, train_size, c1, c2):
        self.queue.put( super().train(sub_train=train_size) )
    def train(self, voter=2, train_size=0.05, c1=0, c2=1):
        pool = []
        for i in range(voter):
            thread = Thread(
                target=self.gen_model,
                args=(train_size, c1, c2)
            )
            thread.start()
            pool.append(thread)
        for thread in bar( pool ):
            thread.join()
            self.model_list.append(self.queue.get())

    def report(self):
        predict_res = []
        vote_res = []
        for model in self.model_list:
            predict_res.append( model.predict(self.X_private) )
        predict_res = zip(*predict_res)
        for vote in predict_res:
            vote_res.append(Counter(vote).most_common(1)[0][0])
        self.Y_pred = vote_res
        super().report()
