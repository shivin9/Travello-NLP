from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import StanfordNERTagger
from sklearn.cluster import KMeans
from stemming.porter2 import stem
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import theano.tensor as T
import multiprocessing
import numpy as np
import lasagne
import urllib2
import string
import theano
import json
import sys
import os
import re
sys.path.insert(0, './database/')
from create_training import getvec



# GLOBAL PARAMETERS
SEQ_LENGTH = 1

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 5

# Number of epochs to train the net
NUM_EPOCHS = 50

# Batch Size
BATCH_SIZE = 256


st = TreebankWordTokenizer()
stagger = StanfordNERTagger('/home/shivin/Documents/Travello-NLP/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', '/home/shivin/Documents/Travello-NLP/stanford-ner/stanford-ner.jar', encoding='utf-8')

with open('./database/streets.json', 'r') as f:
    streets = json.load(f)
with open('./database/states.json', 'r') as f:
    states = json.load(f)
with open('./database/cities.json', 'r') as f:
    cities = json.load(f)
with open('./database/countries.json', 'r') as f:
    countries = json.load(f)

def parsepage(url):
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')
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
    predictrnn(url)

    # raw = soup.get_text().encode('ascii', 'ignore')
    # raw = raw.replace('\t', '')
    # hier_addr = new_address(raw)
    # print str(len(hier_addr)) + " addresses found!"
    # print hier_addr
    # # direct_addr = direct_address(raw)
    # # print direct_addr
    # return [hier_addr]

# hierarchical addresses
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

# one line addresses
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
            # print "state " + terms + " found!"
            score+=1
        if terms.lower() in streets:
            # print "street " + terms + " found!"
            score+=3
        if terms in cities:
            # print "city " + terms + " found!"
            score+=1
        if terms in countries:
            # print "country " + terms + " found!"
            score+=1
    return float(score)/numterm > 0.4

def parseurl(url):
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')
    for elem in soup.findAll(['script', 'style']):
            elem.extract()
    raw = soup.get_text().encode('ascii', 'ignore')
    paragraphs = [p.strip() for p in raw.split('\n') if len(p.strip()) > 2]
    return paragraphs


def getaddr(url, pred):
    paras = parseurl(url)
    data = np.zeros((BATCH_SIZE, SEQ_LENGTH, 8))
    for bn in range(BATCH_SIZE):
        for s in range(SEQ_LENGTH):
            if bn*SEQ_LENGTH + s >= len(paras):
                break
            data[bn, s, :] = np.array(getvec([paras[bn*SEQ_LENGTH+s]]))
    res = pred(data)
    res = res.flatten()
    for i in range(len(paras)):
        print (paras[i], res[i])
    return res, paras


def predictrnn(url):
    l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, SEQ_LENGTH, 8))

    l_forward = lasagne.layers.RecurrentLayer(
            l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)

    l_backward = lasagne.layers.RecurrentLayer(
            l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh,
            only_return_final=True, backwards=True)

    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
    l_dense = lasagne.layers.DenseLayer(
        l_concat, num_units=SEQ_LENGTH, nonlinearity=lasagne.nonlinearities.tanh)

    l_out = lasagne.layers.DenseLayer(l_concat, num_units=SEQ_LENGTH, nonlinearity=lasagne.nonlinearities.tanh)

    target_values = T.dmatrix('target_output')
    network_output = lasagne.layers.get_output(l_out)
    all_param_values = np.load('./models/rnnmodel-old.npy')

    all_params = lasagne.layers.get_all_params(l_out)
    for p, v in zip(all_params, all_param_values):
        p.set_value(v)

    pred = theano.function([l_in.input_var],network_output,allow_input_downcast=True)
    res, parag = getaddr(url, pred)
    res = res.reshape(-1,1)
    est = KMeans(n_clusters = 3)
    est.fit(res)
    labels = est.labels_
    dict = {}
    dict[0] = []
    dict[1] = []
    dict[2] = []
    print len(labels), len(parag)
    for i in range(len(parag)):
        dict[labels[i]].append(parag[i])

    print dict


if __name__ == '__main__':
    while 1:
        url = raw_input("enter website to parse\n")
        parsepage(url)