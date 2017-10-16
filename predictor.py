import pickle
from lib.death_book import txt_loader, book
from sklearn_crfsuite import metrics

X_private = pickle.load( open('./pickles/X_private.pkl', 'rb') )
Y_private = pickle.load( open('./pickles/Y_private.pkl', 'rb') )
Y_pred = []

label_cache = [ 'I', 'I', 'I', 'I' , 'I']
crf_model = pickle.load( open('./pickles/crf_best_model.pkl', 'rb') )
for feature in X_private:
    feature_copy = feature.copy()
    feature_copy['y@-1'] = label_cache[-1]
    feature_copy['y@-2'] = label_cache[-2]
    feature_copy['y@-3'] = label_cache[-3]
    feature_copy['y@-4'] = label_cache[-4]
    feature_copy['y@-5'] = label_cache[-5]
    pred = crf_model.predict_single([feature_copy])[0]
    Y_pred.append(pred)
    label_cache.append(pred)
    label_cache.pop(0)

Y_pred_ideal = crf_model.predict_single(X_private)

print(metrics.flat_classification_report(
    Y_private, Y_pred, labels=('I', 'E'), digits=3
))
print( '============== ideal y@-1 ~ y@-4 ================' )
print(metrics.flat_classification_report(
    Y_private, Y_pred_ideal, labels=('I', 'E'), digits=3
))
