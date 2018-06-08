from lib.data import Data

path = '../data/tang_epitaph.txt'
data = Data(path)
print(data.text[0])
print(' '.join(data.Y[0]))

print('data len', len(data.text[0]))
print('data len', len(data.Y[0]))
