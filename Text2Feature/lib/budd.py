import xml.etree.cElementTree as ET
import re
from glob import glob
from itertools import chain
from pprint import pprint
ns = {
    'xml' : 'http://www.w3.org/XML/1998/namespace',
    'tei' : 'http://www.tei-c.org/ns/1.0'
}

def tag_finder(x, tag):
    tags = x.findall('tei:'+tag, ns)
    return list(chain(*[ [ele] + tag_finder(ele, tag) for ele in tags ]))

data_all = []
for path in glob('../data/buddhist/*.xml'):
    text = open(path, 'r').read()
    text = re.sub(r'<note[^>]*>[^<]*</note>', '', text)
    text = re.sub(r'<sic[^>]*>[^<]*</sic>', '', text)
    text = re.sub(r'<del[^>]*>[^<]*</del>', '', text)
    text = re.sub(r'<orig[^>]*>[^<]*</orig>', '', text)
    text = re.sub(r'<caesura/>', '，', text)
    text = re.sub(r'</l>', r'。</l>', text)
    tree = ET.fromstring(text)
    root = tree.find('tei:text/tei:body/tei:div', ns)
    data_list = [ ele for ele in root.findall('tei:div', ns) ]


    for chap in data_list:
        try:
            title = chap.find('tei:head', ns).text
        except:
            continue
        div_list = tag_finder(chap, 'div')
        hp_list = []
        for div in div_list:
            h = div.findall('tei:head', ns)
            h = ''.join(h[0].itertext()) if len(h)>0 else ''
            p = div.findall('tei:p', ns)
            if len(p) == 0:
                continue
            p = ''.join([ ''.join(_p.itertext()) for _p in p ])
            lg = div.findall('tei:lg', ns)
            lg = ''.join(lg[0].itertext()) if len(lg)>0 else ''
            hp_list.append('，'.join([h,p,lg]).replace('，，','，').replace('：，','：').replace('。，','。').replace(' ', '').replace('\n', '').replace('。。','。'))
        data_all.append( (title, hp_list) )

title_list = []
for title, hp_list in data_all:
    if title in title_list:
        continue
    title_list.append(title)
    chap = []
    print('\n'+str(title))
    for sentence in hp_list:
        #  if len(sentence) > 30:
        sentence = re.sub('^，','', sentence)
        #  chap.append(sentence)
        print(sentence)
    #  print('，'.join(chap))
