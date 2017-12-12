import sklearn_crfsuite

class CRF(sklearn_crfsuite.CRF):
    def fit(self, x, y):
        super().fit([ [ele] for ele in x ], [ [ele] for ele in y ])

    def predict(self, x):
        res = super().predict([ [ele] for ele in x ])
        return [ ele[0] for ele in res ]
    def predict_prob(self, x):
        res = super().predict_marginals([ [ele] for ele in x ])
        return [ ele[0]['E'] for ele in res ]
