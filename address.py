from nltk.tokenize import TreebankWordTokenizer
from sklearn.cluster import KMeans
from bs4 import BeautifulSoup
import theano.tensor as T
import multiprocessing
import numpy as np
import datefinder
import lasagne
import urllib2
import string
import theano
import json
import sys
import os
import re
from create_training import getvec
from utils import parsePage

sys.path.insert(0, './database/')

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

# Number of Clusters
NUM_CLUST = 3

NUM_FEATURES = 9

params = {"LEARNING_RATE": 0.01,
          "GRAD_CLIP"    : 100,
          "PRINT_FREQ"   : 5,
          "NUM_EPOCHS"   : 50,
          "BATCH_SIZE"   : 256,
          "NUM_CLUST"    : 3,
          "NUM_FEATURES" : 8}

st = TreebankWordTokenizer()

with open('./database/hard_data/streets.json', 'r') as f:
    streets = json.load(f)
with open('./database/hard_data/states.json', 'r') as f:
    states = json.load(f)
with open('./database/hard_data/cities.json', 'r') as f:
    cities = json.load(f)
with open('./database/hard_data/countries.json', 'r') as f:
    countries = json.load(f)


def parsepage(url):
    soup, paras, paradict = parsePage(url)

    if 'tripadvisor' in url:
        strt = soup.findAll(
            "span", {"class": 'street-address'})[0].get_text().encode('ascii', 'ignore')
        loc = soup.findAll("span", {"class": 'locality'})[0].get_text().encode('ascii', 'ignore')
        count = soup.findAll("span", {"class" : 'country-name'})[0].get_text().encode('ascii', 'ignore')

        addr = strt + ', ' + loc + ', ' + count
        print strt, loc, count
        return [[[strt, loc, count]]]

    pred1 = set(predictrnn(paras))
    pred2 = set(predictlstm(paras))
    print pred1
    print "################"
    print pred2
    pred = pred1.intersection(pred2)
    addresses = sorted(pred1, key=lambda x: x[1])
    final = accuAddr(addresses)
    print final
    return final
    # raw = soup.get_text().encode('ascii', 'ignore')
    # raw = raw.replace('\t', '')
    # hier_addr = new_address(raw)
    # print str(len(hier_addr)) + " addresses found!"
    # print hier_addr
    # # direct_addr = direct_address(raw)
    # # print direct_addr
    # return [hier_addr]


def accuAddr(addresses):
    i = 0
    final = []
    while i < len(addresses):
        accued = [addresses[i][0]]
        while i + 1 < len(addresses) and (addresses[i + 1][1] - addresses[i][1]) <= 2:
            accued += [addresses[i + 1][0]]
            i += 1
        if not hasdate(accued):
            final += [accued]
        i += 1
    return [final]

# function for removing dates from addresses


def hasdate(address):
    str1 = " ".join(address)
    matches = datefinder.find_dates(str1, strict=True)
    for match in matches:
        return True
    return False

# hierarchical addresses


def get_address(text):
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 2]
    lens = [len(st.tokenize(p)) for p in paragraphs]
    regexp = re.compile(
        r'\+[0-9][0-9]*|\([0-9]{3}\)|[0-9]{4} [0-9]{4}|([0-9]{3,4}[- ]){2}[0-9]{3,4}')
    possible_addresses = []

    # to retrieve addresses which have phone number at the end
    for idx in range(len(paragraphs)):
        if bool(regexp.search(paragraphs[idx])):  # and lens[idx] <= 9:
            # to collect lines above the phone number
            poss = []
            poss.append(paragraphs[idx])
            temp = idx - 1
            while lens[temp] <= 9:
                poss.append(paragraphs[temp])
                temp -= 1

            # address cant be that long
            if len(poss) <= 15:
                possible_addresses.append(poss[::-1])

    return possible_addresses

# one line addresses


def direct_address(text):
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 2]
    lens = [len(st.tokenize(p)) for p in paragraphs]
    cmms = np.array([p.count(',') for p in paragraphs])

    regexp = re.compile(
        r'\+[0-9][0-9]*|\([0-9]{3}\)|([0-9]{3,4}[- ]){2}[0-9]{3,4}')
    paddridx = np.where(cmms >= 2)[0]
    possible_addresses = []

    for idx in paddridx:
        possible_addresses.append(paragraphs[idx])

    for p in paragraphs:
        if bool(regexp.search(p)):
            possible_addresses.append(p)

    hopefully_addresses = - []
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
    regexp = re.compile(
        r'\+[0-9][0-9]*|\([0-9]{3}\)|[0-9]{4} [0-9]{4}|([0-9]{3,4}[- ]){2}[0-9]{3,4}|[0-9]{10}')
    # print paragraphs
    print lens
    possible_addresses = []
    idx = 0
    while idx < len(paragraphs):
        # first filter the paragraphs by phone number
        if bool(regexp.search(paragraphs[idx])):
            poss = []
            poss.append(paragraphs[idx])
            temp = idx - 1
            print paragraphs[idx]
            # go back till we are seeing an address
            while isAddr(paragraphs[temp]) and lens[temp] < 10:
                poss.append(paragraphs[temp])
                    temp -= 1

            if len(poss) <= 15:
                possible_addresses.append(poss[::-1])
                idx += 1

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
                        temp += 1
                        if temp >= len(paragraphs):
                            break
                    # address less than 10 lines
                    if len(poss) < 10:
                        possible_addresses.append(poss)
                        idx = temp
        idx += 1
    return possible_addresses

# use ML techniques to fix the score increments


def isAddr(test_addr):
    score = 0
    numterm = 0
    for terms in st.tokenize(test_addr):
        numterm += 1
        # terms = terms.lower()
        if terms in states:
            # print "state " + terms + " found!"
            score += 1
        if terms.lower() in streets:
            # print "street " + terms + " found!"
            score += 3
        if terms in cities:
            # print "city " + terms + " found!"
            score += 1
        if terms in countries:
            # print "country " + terms + " found!"
            score += 1
    return float(score) / numterm > 0.4


def getData(paras, seql):
    len1 = len(paras)
    batches = len1 / (BATCH_SIZE * seql) + 1
    data1 = np.zeros((BATCH_SIZE * (batches) * seql, NUM_FEATURES))
    for i in range(len1):
        data1[i] = np.array(getvec([paras[i]]))

    data2 = np.zeros((BATCH_SIZE * (batches), seql, NUM_FEATURES))
    for i in range(len(data1)):
        data2[i / seql, i % seql, :] = data1[i]
    del(data1)
    return data2


def predictrnn(parag):
    # GLOBAL PARAMETERS
    SEQ_LENGTH = 1

    # Number of units in the two hidden (LSTM) layers
    N_HIDDEN = 512
    l_in = lasagne.layers.InputLayer(
        shape=(BATCH_SIZE, SEQ_LENGTH, NUM_FEATURES))

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

    l_out = lasagne.layers.DenseLayer(
        l_concat, num_units=SEQ_LENGTH, nonlinearity=lasagne.nonlinearities.tanh)

    target_values = T.dmatrix('target_output')
    network_output = lasagne.layers.get_output(l_out)
    all_param_values = np.load('./models/rnnmodel.npy')

    all_params = lasagne.layers.get_all_params(l_out)
    for p, v in zip(all_params, all_param_values):
        p.set_value(v)

    pred = theano.function(
        [l_in.input_var], network_output, allow_input_downcast=True)
    data = getData(parag, SEQ_LENGTH)
    numbat = len(data) / BATCH_SIZE
    res = np.zeros((numbat, BATCH_SIZE, SEQ_LENGTH))
    for i in range(numbat):
        res[i, :] = pred(data[BATCH_SIZE * i:BATCH_SIZE * (i + 1), :, :])
    res = res.flatten()
    for i in range(len(parag)):
        print (parag[i], res[i])

    return printAddresses(res, parag)


def predictlstm(parag):
    SEQ_LENGTH = 4
    N_HIDDEN = 64
    l_in = lasagne.layers.InputLayer(
        shape=(BATCH_SIZE, SEQ_LENGTH, NUM_FEATURES))

    gate_parameters = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.))

    cell_parameters = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        # Setting W_cell to None denotes that no cell connection will be used.
        W_cell=None, b=lasagne.init.Constant(0.),
        # By convention, the cell nonlinearity is tanh in an LSTM.
        nonlinearity=lasagne.nonlinearities.tanh)

    l_lstm = lasagne.layers.recurrent.LSTMLayer(
        l_in, N_HIDDEN,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=100.)

    l_lstm_back = lasagne.layers.recurrent.LSTMLayer(
        l_in, N_HIDDEN, ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=100., backwards=True)

    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.
    l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm, l_lstm_back])

    l_reshape = lasagne.layers.ReshapeLayer(l_sum, (-1, N_HIDDEN))

    l_dense = lasagne.layers.DenseLayer(
        l_reshape, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

    l_out = lasagne.layers.ReshapeLayer(l_dense, (BATCH_SIZE, SEQ_LENGTH))

    target_values = T.dmatrix('target_output')

    network_output = lasagne.layers.get_output(l_out)
    cost = T.mean((network_output - target_values)**2)

    all_params = lasagne.layers.get_all_params(l_out)
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    pred = theano.function(
        [l_in.input_var], network_output, allow_input_downcast=True)

    all_param_values = np.load('./models/lstmodel-old.npy')

    all_params = lasagne.layers.get_all_params(l_out)
    for p, v in zip(all_params, all_param_values):
        p.set_value(v)

    data = getData(parag, SEQ_LENGTH)
    numbat = len(data) / BATCH_SIZE
    res = np.zeros((numbat, BATCH_SIZE, SEQ_LENGTH))
    for i in range(numbat):
        res[i, :] = pred(data[BATCH_SIZE * i:BATCH_SIZE * (i + 1), :, :])
    res = res.flatten()
    for i in range(len(parag)):
        print (parag[i], res[i])
    return printAddresses(res, parag)


def printAddresses(res, parag):
    res = res.reshape(-1, 1)
    est = KMeans(n_clusters=NUM_CLUST)
    est.fit(res)
    labels = est.labels_
    dict = {}

    for i in range(NUM_CLUST):
        dict[i] = []

    bestaddr = np.argmax(res)
    if res[bestaddr] < 0.5:
        return []
    for i in range(len(parag)):
        dict[labels[i]].append((parag[i], i))
    return dict[labels[bestaddr]]

# if __name__ == '__main__':
#     while 1:
#         try:
#             url = raw_input("enter website to parse\n")
#         except:
#             print "invalid url"
#         parsepage(url)
