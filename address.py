from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import StanfordNERTagger
from stemming.porter2 import stem
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import multiprocessing
import numpy as np
import string
import urllib
import sys
import os
import re

st = TreebankWordTokenizer()
stagger = StanfordNERTagger('/home/shivin/Documents/Travello-NLP/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', '/home/shivin/Documents/Travello-NLP/stanford-ner/stanford-ner.jar', encoding='utf-8')

# def clean():
#     os.chdir('./data/')
#     pool = multiprocessing.Pool()
#     files = os.listdir('.')
#     pool.map(cleanfile, files)


def parsepage(url):
    # f = open(filename, 'r')
    # raw = f.read()

    # url = raw_input("enter website to parse\n")
    soup = BeautifulSoup(urllib.urlopen(url).read(), 'lxml')

    if 'tripadvisor' in url:
        strt = soup.findAll("span", {"class" : 'street-address'})[0].get_text().encode('ascii', 'ignore')
        loc = soup.findAll("span", {"class" : 'locality'})[0].get_text().encode('ascii', 'ignore')
        count = soup.findAll("span", {"class" : 'country-name'})[0].get_text().encode\
        ('ascii', 'ignore')

        addr = strt + ', ' + loc + ', ' + count
        print strt, loc, count
        return [[[strt, loc, count]]]

    for elem in soup.findAll(['script', 'style']):
        elem.extract()

    raw = soup.get_text().encode('ascii', 'ignore')

    paragraphs = [p for p in raw.split('\n') if p]
    lens = [len(p) for p in paragraphs]

    raww = re.sub('[^\w]', ' ', raw)
    tok = st.tokenize(raww)

    tok1 = [t for t in tok if t not in stopwords.words('english') and len(t)>2 and
            not re.search(r'\d', t)]

    hier_addr = get_address(raw)
    print str(len(hier_addr)) + " addresses found!"
    print hier_addr

    # direct_addr = direct_address(raw)
    # print direct_addr
    return [hier_addr]


def get_address(text):
    paragraphs = [p for p in text.split('\n') if p]
    lens = [len(st.tokenize(p)) for p in paragraphs]
    regexp = re.compile(r'\+[0-9][0-9]*|\([0-9]{3}\)|[0-9]{4} [0-9]{4}')

    possible_addresses = []

    # to retrieve addresses which have phone number at the end
    for idx in range(len(paragraphs)):
        if bool(regexp.search(paragraphs[idx])) and lens[idx] <= 9:
            # to collect lines above the phone number
            poss = []

            while lens[idx] <= 9:
                poss.append(paragraphs[idx])
                idx-=1

            if len(poss) <= 15:
                possible_addresses.append(poss[::-1])

    return possible_addresses

def direct_address(text):
    paragraphs = [p for p in text.split('\n') if p]
    lens = [len(st.tokenize(p)) for p in paragraphs]
    cmms = np.array([p.count(',') for p in paragraphs])

    regexp = re.compile(r'\+[0-9][0-9]*|\([0-9]{3}\)|[0-9]{4} [0-9]{4}')

    paddridx = np.where(cmms>=2)[0]

    possible_addresses = []

    for idx in paddridx:
        possible_addresses.append(paragraphs[idx])

    for p in paragraphs:
        if bool(regexp.search(p)):
            possible_addresses.append(p)

    hopefully_addresses = []

    # print possible_addresses

    for posaddr in possible_addresses:
        if len(st.tokenize(posaddr)) <= 30:
            hopefully_addresses.append(posaddr)

    # print hopefully_addresses

    surely_addresses = []
    for addr in hopefully_addresses:
        classified_addr = stagger.tag(st.tokenize(addr))
        labels = [tag[1] for tag in classified_addr]
        if 'LOCATION' in labels:
            surely_addresses.append(addr)

    print str(len(surely_addresses)) + " addresses found"
    return [surely_addresses]


# def html2text(strText):
#     str1 = strText
#     int2 = str1.lower().find("<body")
#     if int2>0:
#        str1 = str1[int2:]
#     int2 = str1.lower().find("</body>")
#     if int2>0:
#        str1 = str1[:int2]
#     list1 = ['<br>',  '<tr',  '<td', '</p>', 'span>', 'li>', '</h', 'div>' ]
#     list2 = [chr(13), chr(13), chr(9), chr(13), chr(13),  chr(13), chr(13), chr(13)]
#     bolFlag1 = True
#     bolFlag2 = True
#     strReturn = ""
#     for int1 in range(len(str1)):
#       str2 = str1[int1]
#       for int2 in range(len(list1)):
#         if str1[int1:int1+len(list1[int2])].lower() == list1[int2]:
#            strReturn = strReturn + list2[int2]
#       if str1[int1:int1+7].lower() == '<script' or str1[int1:int1+9].lower() == '<noscript':
#          bolFlag1 = False
#       if str1[int1:int1+6].lower() == '<style':
#          bolFlag1 = False
#       if str1[int1:int1+7].lower() == '</style':
#          bolFlag1 = True
#       if str1[int1:int1+9].lower() == '</script>' or str1[int1:int1+11].lower() == '</noscript>':
#          bolFlag1 = True
#       if str2 == '<':
#          bolFlag2 = False
#       if bolFlag1 and bolFlag2 and (ord(str2) != 10) :
#         strReturn = strReturn + str2
#       if str2 == '>':
#          bolFlag2 = True
#       if bolFlag1 and bolFlag2:
#         strReturn = strReturn.replace(chr(32)+chr(13), chr(13))
#         strReturn = strReturn.replace(chr(9)+chr(13), chr(13))
#         strReturn = strReturn.replace(chr(13)+chr(32), chr(13))
#         strReturn = strReturn.replace(chr(13)+chr(9), chr(13))
#         strReturn = strReturn.replace(chr(13)+chr(13), chr(13))
#     strReturn = strReturn.replace(chr(13), '\n')
#     return strReturn


if __name__ == '__main__':
    parsepage()