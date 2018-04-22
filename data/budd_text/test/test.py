text = open('./g005.txt', 'r')

for line in text:
    if len(line) > 30:
        print(line.strip(), file=open('./test.txt', 'a'))
