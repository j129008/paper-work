from lib.data import Data
from lib.learner import Learner
from lib.feature import *
from lib.ensumble_learner import Bagging, Boosting
from lib.metric import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, TimeDistributed, SimpleRNN, Embedding, RNN, GRU, Bidirectional
import unittest
from pprint import pprint

path = './data/data4.txt'

class DataTest(unittest.TestCase):
    def setUp(self):
        context = Context(path)
        mi = MutualInfo(path)
        tdiff = Tdiff(path)
        rhyme = Rhyme(path, './data/rhyme.txt', './pickles/rhyme_list.pkl', ['反切', '聲母', '韻目', '調', '等', '呼', '韻母'])
        office_data = Label(path, 'office', './ref/known/office2.txt')
        self.data = context + mi + tdiff + rhyme + office_data
        self.data.shrink()
        self.data.segment(length=10)
    def test_feautre_info(self):
        pprint(self.data.X[0:2])

class CrfTest(unittest.TestCase):
    def setUp(self):
        self.data = Context(path)
        self.data.shrink()

    def test_boost(self):
        print('boost')
        learner = Boosting(self.data)
        learner.train()
        print(learner.alpha_list)
        learner.report()

    def test_bagging(self):
        print('bagging')
        learner = Bagging(self.data)
        learner.train()
        learner.report()

    def test_learner(self):
        print('crf')
        learner = Learner(self.data)
        learner.train()
        learner.report()

    def test_CV(self):
        print('crf-CV-tune')
        learner = Learner(self.data)
        learner.train_CV()
        learner.report()

    def test_analyzer(self):
        learner = Learner(self.data)
        learner.train()
        learner.state_features()
        pred = learner.predict(learner.X_private)
        ErrorAnalyze(pred=pred, feature_data=learner)
        ErrorCompare(pred_list=[pred, pred], feature_data=learner)

class KerasTest(unittest.TestCase):
    def setUp(self):
        self.k = 1
        self.data = VecContext(path, k=self.k)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data.X, self.data.Y, test_size=0.6, random_state=1, shuffle=False
        )

    def test_biLSTM(self):
        model = Sequential()
        for _ in range(4):
            model.add(Bidirectional(LSTM(50, return_sequences=True, go_backwards=True), input_shape=(self.k*2+1, 100)))
        model.add(Bidirectional(LSTM(50)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(self.x_train, self.y_train, batch_size=100, epochs=1)
        pred = model.predict(self.x_test)
        y_pred = self.data.y2lab(pred)
        y_test = self.data.y2lab(self.y_test)
        print('keras')
        print(metrics.flat_classification_report(
            y_test, y_pred, labels=('I', 'E'), digits=4
        ))

class VecModelTest(unittest.TestCase):
    def setUp(self):
        self.data = UniVec(path)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data.X, self.data.Y, test_size=0.6, random_state=1, shuffle=False
        )

    def test_RandomForest(self):
        clf = RandomForestClassifier()
        clf.fit(self.x_train, self.y_train)

        y_pred = clf.predict(self.x_test)
        y_pred = self.data.y2lab(y_pred)
        y_test = self.data.y2lab(self.y_test)
        print('random forest:')
        print(metrics.flat_classification_report(
            y_test, y_pred, labels=('I', 'E'), digits=4
        ))

if __name__ == '__main__':
    unittest.main()
