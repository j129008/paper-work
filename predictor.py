import pickle
from lib.death_book import txt_loader, book
from sklearn_crfsuite import metrics
from tqdm import tqdm as bar
import sys

def loadTestData(books):
    X, Y = [], []
    for book in bar( books ):
        X.extend( [ feature for feature in book.feature_list ] )
        Y.extend( [ end_lab for end_lab in book.end_label ] )
    return X, Y

if len(sys.argv) == 2:
    books = [ book( txt ) for txt in bar ( txt_loader(sys.argv[1]) ) ]
    X_private, Y_private = loadTestData(books)
else:
    X_private = pickle.load( open('./pickles/X_private.pkl', 'rb') )
    Y_private = pickle.load( open('./pickles/Y_private.pkl', 'rb') )

Y_pred = []
crf_model = pickle.load( open('./pickles/crf_best_model.pkl', 'rb') )
Y_pred_ideal = crf_model.predict(X_private)

print(metrics.flat_classification_report(
    Y_private, Y_pred_ideal, labels=('I', 'E'), digits=3
))
