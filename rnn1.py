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

import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys
sys.path.insert(0, './database/')
sys.path.insert(0, './database/features')
from datavec1 import X1
from datavec2 import X2
from labels1 import y1
from labels2 import y2
from create_training import getvec

SEQ_LENGTH = 4

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 64

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 5

# Number of epochs to train the net
NUM_EPOCHS = 10

# Batch Size
BATCH_SIZE = 256

NUM_FEATURES = 8

def gen_data(p, X, y, batch_size=BATCH_SIZE):
    x = np.zeros((batch_size, SEQ_LENGTH, NUM_FEATURES))
    X = np.array(X)
    y = np.array(y)
    yout = np.zeros((batch_size, SEQ_LENGTH))
    cnt = p
    for n in range(batch_size):
        for i in range(SEQ_LENGTH):
            if cnt == len(y):
                break
            x[n, i, :] = X[cnt]
            cnt+=1
        for i in range(len(y[p + n*SEQ_LENGTH : cnt])):
            yout[n, i] = y[p + n*SEQ_LENGTH : cnt][i]
    return x, np.array(yout)

print "creating layers"

l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, SEQ_LENGTH, NUM_FEATURES))

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

print('compiling the network...')
train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)

pred = theano.function([l_in.input_var],network_output,allow_input_downcast=True)
data_size = len(X1)


with open('testdoc', 'r') as f1:
        res = f1.read()
paragraphs = [p.strip() for p in res.split('\n') if len(p.strip()) > 2][:-2]
out = []
for p in paragraphs:
    out.append(getvec(p))
ans = [0,0,1,1,1,0,0,0,0,0,0,0][:-2]
out = np.array(out)
ans = np.array(ans)
xval, yval = gen_data(0, out, ans)


print "training the network..."
p = 0
flag = 0
for it in xrange(data_size * NUM_EPOCHS/ BATCH_SIZE):
    avg_cost = 0
    for _ in range(PRINT_FREQ):
        try:
            x1, y11 = gen_data(p, X1, y1)
            x2, y22 = gen_data(p, X2, y2)
        except:
            flag = 1
            break
        p+=BATCH_SIZE*SEQ_LENGTH/NUM_EPOCHS
        if p+BATCH_SIZE*SEQ_LENGTH>=data_size:
            break
        avg_cost += train(x1, y11)
        avg_cost += train(x2, y22)
    valerr = compute_cost(xval,yval)
    if flag == 1:
        break
    if valerr < 0.005 and it > 50:
        break
    print ("Epoch {} validation cost = {}".format(it*1.0*PRINT_FREQ/data_size*BATCH_SIZE, valerr))

def parsepage(url):
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


def getaddr(url):
        paras = parsepage(url)
        data = np.zeros((BATCH_SIZE, SEQ_LENGTH, NUM_FEATURES))
        for bn in range(BATCH_SIZE):
            for s in range(SEQ_LENGTH):
                if bn*SEQ_LENGTH + s >= len(paras):
                    break
                data[bn, s, :] = np.array(getvec([paras[bn*SEQ_LENGTH+s]]))
        res = pred(data)
        res = res.flatten()
        for i in range(len(paras)):
            print (paras[i], res[i])
        return data

all_param_values = [p.get_value() for p in all_params]
np.save("lstmodel", all_param_values)


while 1:
    url = raw_input("enter website to parse\n")
    try:
        getaddr(url)
    except:
        print "invalid website"