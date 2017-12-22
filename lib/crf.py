from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import sklearn_crfsuite
from random import sample, randrange
import pdb

class RandomForest(RandomForestClassifier):
    def build_index(self, x):
        self.vec = DictVectorizer()
        self.vec.fit_transform(x)
    def fit(self, x, y):
        vec_x = self.vec.transform(x)
        vec_y = [ 1 if ele == 'E' else 0 for ele in y ]
        super().fit(vec_x, vec_y)
    def predict(self, x):
        vec_x = self.vec.transform(x)
        vec_y = super().predict(vec_x)
        y = [ 'E' if ele == 1 else 'I' for ele in vec_y ]
        return y

class CRF(sklearn_crfsuite.CRF):
    def fit(self, x, y):
        super().fit([ [ele] for ele in x ], [ [ele] for ele in y ])

    def predict(self, x):
        res = super().predict([ [ele] for ele in x ])
        return [ ele[0] for ele in res ]
    def predict_prob(self, x):
        res = super().predict_marginals([ [ele] for ele in x ])
        res_list = []
        for ele in res:
            try:
                res_list.append(ele[0]['E'])
            except:
                res_list.append(0.0)
        return res_list

class WeightCRF(CRF):
    def fit(self, x, y, weight_list=None):
        if weight_list == None:
            return super().fit(x, y)
        N = len(x)
        weight_list = [ int(weight*N*10) for weight in weight_list ]
        x_ = []
        y_ = []
        for i in range(len(weight_list)):
            x_.extend( weight_list[i]*[ x[i] ] )
            y_.extend( weight_list[i]*[ y[i] ] )
        super().fit(x_, y_)

class RandomCRF(CRF):
    def feature_select(self, x, feature_list):
        random_x = []
        for ins in x:
            new_ins = {}
            for key in feature_list:
                new_ins[key] = ins[key]
            random_x.append(new_ins)
        return random_x
    def fit(self, x, y):
        feature = [*x[0].keys()]
        n_feature = randrange(1, len(feature))
        self.random_feature = sample(feature, n_feature)
        random_x = self.feature_select(x, self.random_feature)
        super().fit(random_x, y)
    def predict(self, x):
        random_x = self.feature_select(x, self.random_feature)
        return super().predict(random_x)
    def predict_prob(self, x):
        random_x = self.feature_select(x, self.random_feature)
        return super().predict_prob(random_x)
