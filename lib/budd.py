import xml.etree.cElementTree as ET
import re
from glob import glob
from itertools import chain
from pprint import pprint
ns = {
    'xml' : 'http://www.w3.org/XML/1998/namespace',
    'tei' : 'http://www.tei-c.org/ns/1.0'
}

data_dict = {}
for path in glob('../data/buddhist/*.xml'):
    text = open(path, 'r').read()
    text = re.sub(r'<note[^>]*>[^<]*</note>', '', text)
    text = re.sub(r'<caesura/>', '，', text)
    tree = ET.fromstring(text)
    root = tree.find('tei:text/tei:body/tei:div', ns)
    data_list = [ ele for ele in root.findall('tei:div', ns) ]

    def tag_finder(x, tag):
        tags = x.findall('tei:'+tag, ns)
        return list(chain(*[ tag_finder(ele, tag) for ele in tags ])) + tags

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
            p = ''.join(p[0].itertext()) if len(p)>0 else ''
            lg = div.findall('tei:lg', ns)
            lg = ''.join(lg[0].itertext()) if len(lg)>0 else ''
            hp_list.append(h+'，'+p+lg)
        data_dict[title] = hp_list
for title in data_dict:
    for sentence in data_dict[title]:
        if len(sentence) > 30:
            print(sentence)
