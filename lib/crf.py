import sklearn_crfsuite

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
