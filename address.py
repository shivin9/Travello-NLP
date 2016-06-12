from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import StanfordNERTagger
from stemming.porter2 import stem
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import multiprocessing
import pandas as pd
import numpy as np
import urllib2
import string
import json
import sys
import os
import re

st = TreebankWordTokenizer()
stagger = StanfordNERTagger('/home/shivin/Documents/Travello-NLP/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', '/home/shivin/Documents/Travello-NLP/stanford-ner/stanford-ner.jar', encoding='utf-8')

# addr = pd.read_csv('./database/locs.csv', dtype=str)
# countries = pd.read_csv('./database/country-codes.csv', dtype=str)
# countries =  countries.set_index('CountryCode')['CountryName'].to_dict()
# addr = addr.fillna('')
# states = {}
# cities = {}
# countries = {}

with open('./database/streets.json', 'r') as f:
    streets = json.load(f)
with open('./database/states.json', 'r') as f:
    states = json.load(f)
with open('./database/cities.json', 'r') as f:
    cities = json.load(f)
with open('./database/countries.json', 'r') as f:
    countries = json.load(f)
# for state in addr.State:
#     if state not in stopwords.words('english') and len(state)>1:
#         states[state] = 1

# for citi in addr.Name:
#     if citi not in stopwords.words('english'):
#         cities[citi] = 1

# for count in addr.fname:
#     countries[count] = 1

def parsepage(url):
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')

    # if 'tripadvisor' in url:
    #     strt = soup.findAll("span", {"class" : 'street-address'})[0].get_text().encode('ascii', 'ignore')
    #     loc = soup.findAll("span", {"class" : 'locality'})[0].get_text().encode('ascii', 'ignore')
    #     count = soup.findAll("span", {"class" : 'country-name'})[0].get_text().encode\
    #     ('ascii', 'ignore')

    #     addr = strt + ', ' + loc + ', ' + count
    #     print strt, loc, count
    #     return [[[strt, loc, count]]]

    for elem in soup.findAll(['script', 'style']):
        elem.extract()

    raw = soup.get_text().encode('ascii', 'ignore')
    raw = raw.replace('\t', '')
    # paragraphs = raw.splitlines()
    # paragraphs = [p.strip() for p in raw.split('\n') if len(p) > 2]

    raww = re.sub('[^\w]', ' ', raw)
    tok = st.tokenize(raww)

    tok1 = [t for t in tok if t not in stopwords.words('english') and len(t)>2 and
            not re.search(r'\d', t)]

    hier_addr = new_address(raw)
    # print str(len(hier_addr)) + " addresses found!"
    # print hier_addr
    # direct_addr = direct_address(raw)
    # print direct_addr
    return [hier_addr]


def get_address(text):
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 2]
    lens = [len(st.tokenize(p)) for p in paragraphs]
    regexp = re.compile(r'\+[0-9][0-9]*|\([0-9]{3}\)|[0-9]{4} [0-9]{4}|([0-9]{3,4}[- ]){2}[0-9]{3,4}')
    possible_addresses = []

    # to retrieve addresses which have phone number at the end
    for idx in range(len(paragraphs)):
        if bool(regexp.search(paragraphs[idx])):# and lens[idx] <= 9:
            # to collect lines above the phone number
            poss = []
            poss.append(paragraphs[idx])
            temp = idx-1
            while lens[temp] <= 9:
                poss.append(paragraphs[temp])
                temp-=1

            # address cant be that long
            if len(poss) <= 15:
                possible_addresses.append(poss[::-1])

    return possible_addresses

def direct_address(text):
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 2]
    lens = [len(st.tokenize(p)) for p in paragraphs]
    cmms = np.array([p.count(',') for p in paragraphs])

    regexp = re.compile(r'\+[0-9][0-9]*|\([0-9]{3}\)|([0-9]{3,4}[- ]){2}[0-9]{3,4}')
    paddridx = np.where(cmms>=2)[0]
    possible_addresses = []

    for idx in paddridx:
        possible_addresses.append(paragraphs[idx])

    for p in paragraphs:
        if bool(regexp.search(p)):
            possible_addresses.append(p)

    hopefully_addresses = []
    for posaddr in possible_addresses:
        if len(st.tokenize(posaddr)) <= 30:
            hopefully_addresses.append(posaddr)

    surely_addresses = []
    for addr in hopefully_addresses:
        classified_addr = stagger.tag(st.tokenize(addr))
        labels = [tag[1] for tag in classified_addr]
        if 'LOCATION' in labels:
            surely_addresses.append(addr)

    print str(len(surely_addresses)) + " addresses found"
    return [surely_addresses]

def new_address(text):
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 2]
    tokked = [st.tokenize(p) for p in paragraphs]
    lens = [len(tokked) for p in paragraphs]
    lens = [len(st.tokenize(p)) for p in paragraphs]
    regexp = re.compile(r'\+[0-9][0-9]*|\([0-9]{3}\)|[0-9]{4} [0-9]{4}|([0-9]{3,4}[- ]){2}[0-9]{3,4}|[0-9]{10}')
    # print paragraphs
    print lens
    possible_addresses = []
    idx = 0
    while idx < len(paragraphs):
        # first filter the paragraphs by phone number
        if bool(regexp.search(paragraphs[idx])):
            poss = []
            poss.append(paragraphs[idx])
            temp = idx-1
            print paragraphs[idx]
            # go back till we are seeing an address
            while isAddr(paragraphs[temp]) and lens[temp] < 10:
                poss.append(paragraphs[temp])
                temp-=1

            if len(poss) <= 15:
                possible_addresses.append(poss[::-1])
                idx+=1

        else:
            # some random number for max. length of an address
            poss = []
            if lens[idx] <= 20:
                if isAddr(paragraphs[idx]):
                    temp = idx
                    while isAddr(paragraphs[temp]) or bool(regexp.search(paragraphs[temp])):
                        if lens[temp] >= 10:
                            break
                        poss.append(paragraphs[temp])
                        temp+=1
                        if temp >= len(paragraphs):
                            break
                    # address less than 10 lines
                    if len(poss) < 10:
                        possible_addresses.append(poss)
                        idx = temp
        idx+=1
    return possible_addresses

# use ML techniques to fix the score increments
def isAddr(test_addr):
    score = 0
    numterm = 0
    for terms in st.tokenize(test_addr):
        numterm+=1
        # terms = terms.lower()
        if terms in states:
            print "state " + terms + " found!"
            score+=1
        if terms.lower() in streets:
            print "street " + terms + " found!"
            score+=3
        if terms in cities:
            print "city " + terms + " found!"
            score+=1
        if terms in countries:
            print "country " + terms + " found!"
            score+=1
    return float(score)/numterm > 0.4

if __name__ == '__main__':
    url = raw_input("enter website to parse\n")
    parsepage(url)