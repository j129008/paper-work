text = open( './ref/known/office2.txt_out' ).read()

stack = []
deep = 0
out = ''

for w in text:
    if w == '（':
        deep += 1
        if deep == 1:
            out+= w
        stack.append( w )
        continue
    if w == '）':
        deep -= 1
        if deep == 0:
            out+= w
        stack.pop()
        continue
    out += w


print( out )
