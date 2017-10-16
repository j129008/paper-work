import pickle
import re
from collections import Counter

crf  = pickle.load(open('./pickles/crf_best_model.pkl', 'rb'))
X    = pickle.load(open('./pickles/X_private.pkl', 'rb'))
Y    = pickle.load(open('./pickles/Y_private.pkl', 'rb'))
P    = crf.predict([X])[0]
text = "".join( [ele['@0'] for ele in X] )

def predict( x ):
    return crf.predict( [ x ] )[0]

def compare( i, j ):
    ts = ''
    ps = ''
    p = predict( X[i:j] )
    t = Y[i:j]
    txt = text[i:j]
    for wi in range( len( txt ) ):
        if ( t[wi], p[wi] ) == ( 'I', 'I' ):
            ts += txt[wi]
            ps += txt[wi]
        elif ( t[wi], p[wi] ) == ( 'E', 'E' ):
            ts += txt[wi] + '，'
            ps += txt[wi] + '，'
        elif ( t[wi], p[wi] ) == ( 'E', 'I' ):
            ts += txt[wi] + '，'
            ps += txt[wi] + '　'
        elif ( t[wi], p[wi] ) == ( 'I', 'E' ):
            ts += txt[wi] + '　'
            ps += txt[wi] + '，'
    return ( ts, ps )

ts, ps = compare(0, len(X))
over = re.findall( '(...)　', ts )
miss = re.findall( '(...)　', ps )

print( "not end but predict end" )
for tail, cnt in Counter(over).most_common(10):
    print( tail + '| cnt:', cnt, 'rate:', cnt/len(re.findall(tail, ts)) )

print('================================')

print( "miss end" )
for tail, cnt in Counter(miss).most_common(10):
    print( tail + '| cnt:', cnt, 'rate:', cnt/len(re.findall(tail, ps)) )

print('================================')

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))


print('================================')

print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(30))

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:])

print('================================')

err_index = []
for i in range( len(Y) ):
    if Y[i] != P[i]:
        err_index.append(i)

p = ''
for i in err_index:
    try:
        ts, ps = compare(i-10, i+10)
        p += 'T:' + ts + '\n' + 'P:' + ps + '\n'
    except:
        pass
print(p)
