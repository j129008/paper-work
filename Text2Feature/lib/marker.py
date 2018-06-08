from glob import glob

name = []
text = open( './data2.txt', 'r' ).read()

for f_name in glob( './ref/known/*.txt' ):
    name += list( set( open( f_name, 'r' ).read().split('\n')[:-1] ) )

i = 0
for n in name:
    i += 1
    text = text.replace( n, '|' + n + '|' )
    if i%1000 == 0:
        print(i/len(name))
        f = open( 'all.txt' ,'w' )
        f.write(text)
        f.close()
