import numpy as np
import time
import sys
import re

import theano
import theano.tensor as T
import lasagne

sys.path.insert(0, './database/')
sys.path.insert(0, './database/features')
from datavec1 import X1
from datavec2 import X2
from labels1 import y1
from labels2 import y2
from create_training import getvec

# Optimization learning rate
LEARNING_RATE = .05

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 5

# Number of epochs to train the net
NUM_EPOCHS = 2

# Batch Size
BATCH_SIZE = 256

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

NUM_FEATURES = 9

WINDOW_SIZE = 5

def load_dataset(X, y):
    # y_new = []

    for i in range(len(X)):
        X[i] = np.array(X[i])

    X = np.array(X)
    y = np.array(y, dtype='int32')

    X_train = X[:-1000]
    y_train = y[:-1000]

    X_val = X[-1000:]
    y_val = y[-1000:]

    return X_train, y_train, X_val, y_val

def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))


def mlp(input_var = None, depth=2, width=N_HIDDEN, drop_input=.2,
                     drop_hidden=.5):

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(BATCH_SIZE, NUM_FEATURES*WINDOW_SIZE), input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)

    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)

    # Output layer:
    nonlinlayer = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, BATCH_SIZE, nonlinearity=nonlinlayer)
    return network

def iterate_minibatches(inputs, targets, batchsize=BATCH_SIZE, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


if __name__ == '__main__':
    print "loading data..."
    X_train1, y_train1, X_val1, y_val1 = load_dataset(X1, y1)
    X_train2, y_train2, X_val2, y_val2 = load_dataset(X2, y2)

    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')
    model = 'mlp'

    print "creating network..."

    if model == 'mlp':
        network = mlp(input_var=input_var, depth=2, width=N_HIDDEN, drop_input=0.2, drop_hidden=0.2)

    prediction = lasagne.layers.get_output(network)
    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    # loss = T.neg(T.dot(T.argmax(prediction, axis=1), target_var))
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adagrad(loss, params, LEARNING_RATE)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    # test_loss = T.neg(T.dot(T.argmax(test_prediction, axis=1), target_var))
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
        dtype=theano.config.floatX)

    pred = theano.function([input_var], test_prediction, allow_input_downcast=True)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print "starting training..."

    for epoch in range(NUM_EPOCHS):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X_train1, y_train1, BATCH_SIZE, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        for batch in iterate_minibatches(X_train2, y_train2, BATCH_SIZE, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0

        for batch in iterate_minibatches(X_val1, y_val1, BATCH_SIZE, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} (Composite addresses) took {:.3f}s".format(
            epoch + 1, NUM_EPOCHS, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err/train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err/val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc/val_batches * 100))

        for batch in iterate_minibatches(X_val2, y_val2, BATCH_SIZE, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} (OneLine addresses) took {:.3f}s".format(
            epoch + 1, NUM_EPOCHS, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err/train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err/val_batches))
        print("  validation accuracy:\t\t{:.2f} %\n".format(val_acc/val_batches * 100))

    print "saving the parameters..."
    all_param_values = [p.get_value() for p in params]
    np.save("newrolling", all_param_values)


    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_val1, y_val1, BATCH_SIZE, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    all_param_values = np.load('./newrolling.npy')

    all_params = lasagne.layers.get_all_params(network)
    for p, v in zip(all_params, all_param_values):
        p.set_value(v)

    # while(1):
    #     try:
    #         url = raw_input("enter website to parse\n")

    #     except:
            # print "invalid url"