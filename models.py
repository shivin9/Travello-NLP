import time
import theano.tensor as T
import numpy as np
import lasagne
import theano
import sys

sys.path.insert(0, './database/features')
sys.path.insert(0, './database/')

from datavec1 import X1
from datavec2 import X2
from labels1 import y1
from labels2 import y2

def _load_dataset(X, y):
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


def getModel(params):
    if params['NAME'] == "rnn":
        pass

    elif params['NAME'] == "lstm":
        pass

    elif params['NAME'] == "rnn+lstm":
        pass

    elif params['NAME'] == "rnnboost":
        pass


def getrnn(input_var, params):
    print "creating layers"

    l_in = lasagne.layers.InputLayer(shape=(params['BATCH_SIZE'], params['SEQ_LENGTH'], params['NUM_FEATURES']), input_var=input_var)

    l_forward = lasagne.layers.RecurrentLayer(
        l_in, params['N_HIDDEN'], grad_clipping=params['GRAD_CLIP'],
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)

    l_backward = lasagne.layers.RecurrentLayer(
        l_in, params['N_HIDDEN'], grad_clipping=params['GRAD_CLIP'],
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True, backwards=True)

    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
    l_out = lasagne.layers.DenseLayer(l_concat, num_units=params['SEQ_LENGTH'],\
            nonlinearity=lasagne.nonlinearities.tanh)

    return l_out


def train(params):
    if params['NAME'] == "rnn":
        pass

    elif params['NAME'] == "lstm":
        pass

    elif params['NAME'] == "rnn+lstm":
        pass

    elif params['NAME'] == "rnnboost":
        pass


def trainrnn(params):
    print "loading data..."
    X_train1, y_train1, X_val1, y_val1 = _load_dataset(X1, y1)
    X_train2, y_train2, X_val2, y_val2 = _load_dataset(X2, y2)

    input_var = T.dtensor3('input_var')
    l_out = getrnn(input_var, params)

    target_values = T.dmatrix('target_output')
    network_output = lasagne.layers.get_output(l_out)
    cost = T.mean((network_output - target_values)**2)

    all_params = lasagne.layers.get_all_params(l_out)
    updates = lasagne.updates.adagrad(cost, all_params, params['LEARNING_RATE'])

    print('compiling the network ' + params['NAME'])

    train = theano.function([input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    validate = theano.function([input_var, target_values], cost, allow_input_downcast=True)
    pred = theano.function([input_var], network_output, allow_input_downcast=True)


    for epoch in range(params['NUM_EPOCHS']):
        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X_train1, y_train1, params['BATCH_SIZE'],\
         params['SEQ_LENGTH'], shuffle=False):
            if train_batches%50 == 0:
                print "batch number " + str(train_batches)
            inputs, targets = batch
            train_err += train(inputs, targets)
            train_batches += 1

        for batch in iterate_minibatches(X_train2, y_train2, params['BATCH_SIZE'],\
         params['SEQ_LENGTH'], shuffle=False):
            if train_batches%50 == 0:
                print "batch number " + str(train_batches)
            inputs, targets = batch
            train_err += train(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0

        for batch in iterate_minibatches(X_val1, y_val1, params['BATCH_SIZE'],\
         params['SEQ_LENGTH'], shuffle=False):
            inputs, targets = batch
            err = validate(inputs, targets)
            val_err += err
            # val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} (Composite addresses) took {:.3f}s".format(
            epoch + 1, params['NUM_EPOCHS'], time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        # print("  validation accuracy:\t\t{:.2f} %".format(val_acc/val_batches * 100))

        val_err = 0
        val_acc = 0
        val_batches = 0

        for batch in iterate_minibatches(X_val2, y_val2, params['BATCH_SIZE'],\
         params['SEQ_LENGTH'], shuffle=False):
            inputs, targets = batch
            err = validate(inputs, targets)
            val_err += err
            # val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} (OneLine addresses) took {:.3f}s".format(
            epoch + 1, params['NUM_EPOCHS'], time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err/train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err/val_batches))
        # print("  validation accuracy:\t\t{:.2f} %\n".format(val_acc/val_batches * 100))

    print "saving the parameters..."
    all_param_values = [p.get_value() for p in all_params]
    np.save("./models/"+str(params), all_param_values)
    return pred

def iterate_minibatches(inputs, targets, batchsize, SEQ_LENGTH=None, shuffle=False):
    assert len(inputs) == len(targets)
    num_feat = inputs.shape[1]
    if SEQ_LENGTH:
        batches = len(inputs) / (batchsize * SEQ_LENGTH) + 1
        X = np.zeros((batchsize * (batches), SEQ_LENGTH, num_feat))
        y = np.zeros((batchsize * (batches), SEQ_LENGTH))

        for i in range(len(inputs)):
            X[i / SEQ_LENGTH, i % SEQ_LENGTH, :] = inputs[i]
            y[i / SEQ_LENGTH, i % SEQ_LENGTH] = targets[i]

        for i in range(batches):
            yield X[i*batchsize : (i+1)*batchsize], y[i*batchsize : (i+1)*batchsize]

        # start_idx = 0
        # while start_idx < len(inputs):
        #     for i in range(batchsize):
        #         data2 = np.zeros(batchsize, SEQ_LENGTH, num_feat)
        #         data2[i/start_idx, i % SEQ_LENGTH, :] = data1[i]


    else:
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)

        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]


def getlstm():
    pass


def getboth():
    pass


def getrnnboost():
    pass

