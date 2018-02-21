from random import sample, randrange
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
import sklearn_crfsuite
import pickle
import logging

logging.basicConfig(level=logging.DEBUG)

class RandomForest(RandomForestClassifier):
    def build_index(self, x, max_dim=200):
        try:
            logging.info('load index file')
            self.vec = pickle.load(open('./pickles/'+str(max_dim)+'_vec.pkl', 'rb'))
            vec_x = self.vec.transform(x)
            self.max_dim = max_dim
            self.svd = pickle.load(open('./pickles/'+str(max_dim)+'_svd.pkl', 'rb'))
            logging.info('load finish')
        except Exception as e:
            logging.info(str(e))
            logging.info('indexing')
            self.vec = DictVectorizer()
            vec_x = self.vec.fit_transform(x)
            pickle.dump(self.vec, open('./pickles/'+str(max_dim)+'_vec.pkl', 'wb'))

            logging.info('dim: ' + str(vec_x.shape[1]))
            self.max_dim = max_dim

            logging.info('running SVD to max dim: ' + str(max_dim))
            self.svd = TruncatedSVD(n_components=min(max_dim, vec_x.shape[1]))
            self.svd.fit(vec_x)
            pickle.dump(self.svd, open('./pickles/'+str(max_dim)+'_svd.pkl', 'wb'))
    def save_index(self, x, y, lab):
        logging.info('save index')
        big_vec_x = self.vec.transform(x)
        vec_x = self.svd.transform(big_vec_x)
        vec_y = [ 1 if ele == 'E' else 0 for ele in y ]
        pickle.dump(vec_x, open('./pickles/'+'_vec_x_'+lab+'.pkl', 'wb'))
        pickle.dump(vec_y, open('./pickles/'+'_vec_y_'+lab+'.pkl', 'wb'))
    def fit(self, x, y):
        logging.info('fitting')
        if len(x) <= 0:
            return False
        big_vec_x = self.vec.transform(x)
        vec_x = self.svd.transform(big_vec_x)
        vec_y = [ 1 if ele == 'E' else 0 for ele in y ]
        pickle.dump(vec_x, open('./pickles/'+'vec_x_train.pkl', 'wb'))
        pickle.dump(vec_y, open('./pickles/'+'vec_y_train.pkl', 'wb'))
        super().fit(vec_x, vec_y)
        return True
    def predict(self, x):
        big_vec_x = self.vec.transform(x)
        vec_x = self.svd.transform(big_vec_x)
        vec_y = super().predict(vec_x)
        y = [ 'E' if ele == 1 else 'I' for ele in vec_y ]
        return y
    def predict_prob(self, x):
        vec_x = self.svd.transform(self.vec.transform(x))
        vec_y = super().predict_proba(vec_x)
        y = [ 1.0 - ele[0] for ele in vec_y ]
        return y

class WeightRandomForest(RandomForest):
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
        if len(x_) > 0:
            super().fit(x_, y_)
            return True
        else:
            return False

class CRF(sklearn_crfsuite.CRF):
    def fit(self, x, y):
        if len(x) <= 0:
            return False
        super().fit([x], [y])
        return True

    def predict(self, x):
        res = super().predict([x])
        return res[0]
    def predict_prob(self, x):
        res = super().predict_marginals([x])
        res_list = [ ele['E'] for ele in res[0] ]
        return res_list

class WeightCRF(CRF):
    def fit(self, x, y, weight_list=None):
        if len(x) <= 0:
            return False
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
        return True

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
