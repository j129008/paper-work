import lightgbm as lgb
from lib.feature import *
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics

data = UniVec('./data/train.txt', k=1)
test_data = UniVec('./data/test.txt', k=1)

x_train, x_valid, y_train, y_valid = train_test_split(
    data.X, data.Y, test_size=0.1, shuffle=True
)

train_data = lgb.Dataset(x_train, y_train)
valid_data = lgb.Dataset(x_valid, y_valid, reference=train_data)

params = {
        'task'             : 'train',
        'boosting_type'    : 'gbdt',
        'objective'        : 'regression',
        'metric'           : {'l2', 'auc'},
        'num_leaves'       : 2047,
        'learning_rate'    : 0.05,
        'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8,
        'bagging_freq'     : 5,
        'verbose'          : 0
}
num_round = 1
bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data], early_stopping_rounds=10)

pred = bst.predict(test_data.X)
lab_pred = data.y2lab(pred)
lab_true = data.y2lab(y_test)
print(metrics.flat_classification_report(
    lab_true, lab_pred, labels=('I', 'E'), digits=4
))
