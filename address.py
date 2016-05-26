from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import StanfordNERTagger
from stemming.porter2 import stem
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import multiprocessing
import numpy as np
import urllib
import sys
import os
import re

st = TreebankWordTokenizer()
stagger = StanfordNERTagger('/home/shivin/Documents/Travello-NLP/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', '/home/shivin/Documents/Travello-NLP/stanford-ner/stanford-ner.jar', encoding='utf-8')

def clean():
    os.chdir('./data/')
    pool = multiprocessing.Pool()
    files = os.listdir('.')
    pool.map(cleanfile, files)


def cleanfile(filename):
    f = open(filename, 'r')
    raw = f.read()

    '''url = 'http://edition.cnn.com/2015/09/16/travel/singapore-new-restaurants-2015/'
    html = urllib.urlopen(url).read()
    soup = BeautifulSoup(html)

    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    raw = soup.get_text()
    print raw'''

    paragraphs = [p for p in raw.split('\n') if p]
    lens = [len(p) for p in paragraphs]
    # print lens
    str1 = cleanString(raw)
    # new = open(filename+'.tok', 'a')
    print 'writing in ' + filename
    # print >> new, str1


def cleanString(str1):
    raw = re.sub('[^\w]', ' ', str1)
    tok = st.tokenize(raw)

    tok1 = [t for t in tok if t not in stopwords.words('english') and len(t)>2 and
            not re.search(r'\d', t)]

    # direct_addr = direct_address(str1)
    # print direct_addr

    hier_addr = get_address(str1)
    print hier_addr
    str1 = ' '.join(tok1)
    return str1


def get_address(text):
    paragraphs = [p for p in text.split('\n') if p]
    lens = [len(st.tokenize(p)) for p in paragraphs]
    regexp = re.compile(r'\+[0-9][0-9]*|\([0-9]{3}\)')

    possible_addresses = []

    # to retrieve addresses which have phone number at the end
    for idx in range(len(paragraphs)):
        if bool(regexp.search(paragraphs[idx])) and lens[idx] <= 9:
            possible_addresses.append([paragraphs[idx-2], paragraphs[idx-1], paragraphs[idx]])

    return possible_addresses

def direct_address(text):
    paragraphs = [p for p in text.split('\n') if p]
    lens = [len(st.tokenize(p)) for p in paragraphs]
    cmms = np.array([p.count(',') for p in paragraphs])

    regexp = re.compile(r'\+[0-9][0-9]*|\([0-9]{3}\)')

    possible_addresses = np.where(cmms>=2)[0]
    possible_addresses = []

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
    return surely_addresses


if __name__ == '__main__':
    clean()