from lib.learner import Learner

class Crfsuite(Learner):
    def __init__(self, path):
        super().__init__(path)
    def crfSuite_transform(self):
        def transform(self, X, Y,output_file=None):
            key_list = sorted( list( X[0].keys() ) )
            data_pool = []
            print('transform to crfSuite:')
            for i in range(len(X)):
                instance = []
                instance.append( str( Y[i] ) )
                for key in key_list:
                    instance.append( str( X[i][key] ) )
                data_pool.append( '\t'.join( instance ) )
            print('write to file: ', output_file)
            open(output_file, 'w').write('\n'.join(data_pool))
        transform(self.X_train, self.Y_train, output_file='./train.txt')
        transform(self.X_private, self.Y_private, output_file='./test.txt')
