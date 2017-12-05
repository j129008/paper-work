from sklearn.model_selection import train_test_split
from lib.learner import Learner
from collections import Counter

class Bagging(Learner):
    def __init__(self, path):
        super().__init__(path)
        self.model_list = []
    def train(self, voter=2, train_size=0.05, c1=0, c2=1):
        for _ in range(voter):
            self.model_list.append(super().train(sub_train=train_size))

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
