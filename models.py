from utils import load_dataset
from nltk.tokenize import TreebankWordTokenizer
import re
import time
import theano.tensor as T
import numpy as np
import lasagne
import theano
import sys

sys.path.insert(0, './database/features')
sys.path.insert(0, './database/')
st = TreebankWordTokenizer()

from datavec1 import X1
from datavec2 import X2
from labels1 import y1
from labels2 import y2


def getModel(params, filename=None):
    if params['NAME'] == "RNN":
        predictor = getRNN(params, filename)

    elif params['NAME'] == "LSTM":
        predictor = getLSTM(params, filename)

    elif params['NAME'] == "CNN":
        predictor = getCNN(params, filename)

    elif params['NAME'] == "BoostedRNN":
        pass

    elif params['NAME'] == "RULE":
        predictor = get_address

    return predictor


def cnn(params, input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(params['BATCH_SIZE'], 1, params[
                                        'CONV'], params['NUM_FEATURES']), input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 2x2. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=16, filter_size=(3, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.2),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.2),
        num_units=params['CONV'],
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def rnn(input_var, params):
    print "creating layers"

    l_in = lasagne.layers.InputLayer(shape=(params['BATCH_SIZE'], params[
        'SEQ_LENGTH'], params['NUM_FEATURES']), input_var=input_var)

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
    l_out = lasagne.layers.DenseLayer(l_concat, num_units=params['SEQ_LENGTH'],
                                      nonlinearity=lasagne.nonlinearities.tanh)

    return l_out


def lstm(input_var, params):

    l_in = lasagne.layers.InputLayer(shape=(params['BATCH_SIZE'], params[
                                     'SEQ_LENGTH'], params['NUM_FEATURES']), input_var=input_var)

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
        l_in, params['N_HIDDEN'],
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=100.)

    l_lstm_back = lasagne.layers.recurrent.LSTMLayer(
        l_in, params[
            'N_HIDDEN'], ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=100., backwards=True)

    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.
    l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm, l_lstm_back])

    l_reshape = lasagne.layers.ReshapeLayer(l_sum, (-1, params['N_HIDDEN']))

    l_dense = lasagne.layers.DenseLayer(
        l_reshape, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

    l_out = lasagne.layers.ReshapeLayer(
        l_dense, (params['BATCH_SIZE'], params['SEQ_LENGTH']))
    return l_out


def getCNN(params, filename=None):

    X_train1, y_train1, X_val1, y_val1 = load_dataset(X1, y1)
    X_train2, y_train2, X_val2, y_val2 = load_dataset(X2, y2)

    input_var = T.tensor4('inputs')
    l_out = cnn(input_var, params)

    target_values = T.fmatrix('target_output')
    network_output = lasagne.layers.get_output(l_out)
    cost = T.mean((network_output - target_values)**2)

    all_params = lasagne.layers.get_all_params(l_out)
    updates = lasagne.updates.adagrad(
        cost, all_params, params['LEARNING_RATE'])

    pred = theano.function([input_var], network_output,
                           allow_input_downcast=True)

    if filename:
        print "Loading a previously saved model..."
        all_param_values = np.load("./models/" + filename + '.npy')

        for i in range(len(all_param_values)):
            all_param_values[i] = all_param_values[i].astype('float32')

        all_params = lasagne.layers.get_all_params(l_out)
        for p, v in zip(all_params, all_param_values):
            p.set_value(v)

    else:
        print('compiling the ' + params['NAME'])

        train = theano.function([input_var, target_values],
                                cost, updates=updates, allow_input_downcast=True)
        validate = theano.function(
            [input_var, target_values], cost, allow_input_downcast=True)

        old_valerr = [10, 10]

        for epoch in range(params['NUM_EPOCHS']):
            print "Training the network..."
            train_err = 0
            train_batches = 0
            old_netout = l_out
            start_time = time.time()

            for batch in iterate_minibatches(X_train1, y_train1, params['BATCH_SIZE'], params['NUM_FEATURES'], SEQ_LENGTH=None, CONV=8, shuffle=False):
                # if train_batches % 50 == 0:
                #     print "batch number " + str(train_batches)
                inputs, targets = batch
                train_err += train(inputs, targets)
                train_batches += 1
                break

            for batch in iterate_minibatches(X_train2, y_train2, params['BATCH_SIZE'],
                                             params['NUM_FEATURES'],
                                             SEQ_LENGTH=None, CONV=8, shuffle=False):
                # if train_batches % 50 == 0:
                #     print "batch number " + str(train_batches)
                inputs, targets = batch
                train_err += train(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0

            for batch in iterate_minibatches(X_val1, y_val1, params['BATCH_SIZE'],
                                             params['NUM_FEATURES'],
                                             SEQ_LENGTH=None, CONV=8, shuffle=False):
                inputs, targets = batch
                err = validate(inputs, targets)
                val_err += err
                # val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} (Composite addresses) took {:.3f}s".format(
                epoch + 1, params['NUM_EPOCHS'], time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(
                train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            # print("  validation accuracy:\t\t{:.2f} %".format(val_acc/val_batches * 100))

            # to prevent overfitting
            # or val_err - old_valerr[0] < 0.001:
            # or old_valerr[0] - val_err < 0.001:
            if val_err - old_valerr[0] > 0.03:
                print "overfitting or model reached saturation...\n"
                print old_valerr
                l_out = old_netout
                break

            old_netout = l_out
            old_valerr[0] = val_err

            val_err = 0
            val_acc = 0
            val_batches = 0

            for batch in iterate_minibatches(X_val2, y_val2, params['BATCH_SIZE'],
                                             params['NUM_FEATURES'], SEQ_LENGTH=None, CONV=8, shuffle=False):
                inputs, targets = batch
                err = validate(inputs, targets)
                val_err += err
                # val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} (OneLine addresses) took {:.3f}s".format(
                epoch + 1, params['NUM_EPOCHS'], time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(
                train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            # print("  validation accuracy:\t\t{:.2f} %\n".format(val_acc/val_batches * 100))
            old_valerr[1] = val_err

        print "saving the parameters..."
        all_param_values = [p.get_value() for p in all_params]
        np.save("./models/" + str(params), all_param_values)

    return pred


def getRNN(params, filename=None):

    print "loading data..."

    # the simple rnn works with a single window
    X_train1, y_train1, X_val1, y_val1 = load_dataset(X1, y1, wndw=params['SEQ_LENGTH'])
    X_train2, y_train2, X_val2, y_val2 = load_dataset(X2, y2, wndw=params['SEQ_LENGTH'])

    input_var = T.ftensor3('input_var')
    l_out = rnn(input_var, params)

    target_values = T.fmatrix('target_output')
    network_output = lasagne.layers.get_output(l_out)
    cost = T.mean((network_output - target_values)**2)

    all_params = lasagne.layers.get_all_params(l_out)
    updates = lasagne.updates.adagrad(
        cost, all_params, params['LEARNING_RATE'])

    pred = theano.function([input_var], network_output,
                           allow_input_downcast=True)

    if filename:
        print "Loading a previously saved " + params['NAME']
        all_param_values = np.load("./models/" + filename + '.npy')

        for i in range(len(all_param_values)):
            all_param_values[i] = all_param_values[i].astype('float32')

        all_params = lasagne.layers.get_all_params(l_out)
        for p, v in zip(all_params, all_param_values):
            p.set_value(v)

    else:
        print('compiling the ' + params['NAME'])

        train = theano.function([input_var, target_values],
                                cost, updates=updates, allow_input_downcast=True)
        validate = theano.function(
            [input_var, target_values], cost, allow_input_downcast=True)

        old_valerr = [10, 10]

        for epoch in range(params['NUM_EPOCHS']):
            print "Training the network..."
            train_err = 0
            train_batches = 0
            old_netout = l_out
            start_time = time.time()

            for batch in iterate_minibatches(X_train1, y_train1, params['BATCH_SIZE'],
                                             params['NUM_FEATURES'], params['SEQ_LENGTH'], shuffle=False):
                # if train_batches % 50 == 0:
                #     print "batch number " + str(train_batches)
                inputs, targets = batch
                train_err += train(inputs, targets)
                train_batches += 1

            for batch in iterate_minibatches(X_train2, y_train2, params['BATCH_SIZE'],
                                             params['NUM_FEATURES'], params['SEQ_LENGTH'], shuffle=False):
                # if train_batches % 50 == 0:
                #     print "batch number " + str(train_batches)
                inputs, targets = batch
                train_err += train(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0

            for batch in iterate_minibatches(X_val1, y_val1, params['BATCH_SIZE'],
                                             params['NUM_FEATURES'], params['SEQ_LENGTH'], shuffle=False):
                inputs, targets = batch
                err = validate(inputs, targets)
                val_err += err
                # val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} (Composite addresses) took {:.3f}s".format(
                epoch + 1, params['NUM_EPOCHS'], time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(
                train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            # print("  validation accuracy:\t\t{:.2f} %".format(val_acc/val_batches * 100))

            # to prevent overfitting
            # or val_err - old_valerr[0] < 0.001:
            # or old_valerr[0] - val_err < 0.001:
            if val_err - old_valerr[0] > 0.03:
                print "overfitting or model reached saturation...\n"
                print old_valerr
                l_out = old_netout
                break

            old_netout = l_out
            old_valerr[0] = val_err

            val_err = 0
            val_acc = 0
            val_batches = 0

            for batch in iterate_minibatches(X_val2, y_val2, params['BATCH_SIZE'],
                                             params['NUM_FEATURES'], params['SEQ_LENGTH'], shuffle=False):
                inputs, targets = batch
                err = validate(inputs, targets)
                val_err += err
                # val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} (OneLine addresses) took {:.3f}s".format(
                epoch + 1, params['NUM_EPOCHS'], time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(
                train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            # print("  validation accuracy:\t\t{:.2f} %\n".format(val_acc/val_batches * 100))
            old_valerr[1] = val_err

        print "saving the parameters..."
        all_param_values = [p.get_value() for p in all_params]
        np.save("./models/" + str(params), all_param_values)

    return pred


def getLSTM(params, filename):

    print "loading data..."
    X_train1, y_train1, X_val1, y_val1 = load_dataset(X1, y1, wndw=params['SEQ_LENGTH'])
    X_train2, y_train2, X_val2, y_val2 = load_dataset(X2, y2, wndw=params['SEQ_LENGTH'])

    input_var = T.ftensor3('input_var')
    l_out = lstm(input_var, params)

    target_values = T.fmatrix('target_output')

    network_output = lasagne.layers.get_output(l_out)
    cost = T.mean((network_output - target_values)**2)

    all_params = lasagne.layers.get_all_params(l_out)
    updates = lasagne.updates.adagrad(
        cost, all_params, params['LEARNING_RATE'])

    pred = theano.function([input_var], network_output,
                           allow_input_downcast=True)

    if filename:
        print "Loading a previously saved " + params['NAME']
        all_param_values = np.load("./models/" + filename + '.npy')

        for i in range(len(all_param_values)):
            all_param_values[i] = all_param_values[i].astype('float32')

        all_params = lasagne.layers.get_all_params(l_out)
        for p, v in zip(all_params, all_param_values):
            p.set_value(v)

    else:

        print('compiling the ' + params['NAME'])

        train = theano.function([input_var, target_values],
                                cost, updates=updates, allow_input_downcast=True)
        validate = theano.function(
            [input_var, target_values], cost, allow_input_downcast=True)

        old_valerr = [10, 10]

        for epoch in range(params['NUM_EPOCHS']):
            print "Training the network..."
            train_err = 0
            train_batches = 0
            old_netout = l_out
            start_time = time.time()

            for batch in iterate_minibatches(X_train1, y_train1, params['BATCH_SIZE'],
                                             params['NUM_FEATURES'], params['SEQ_LENGTH'], shuffle=False):
                # if train_batches % 50 == 0:
                #     print "batch number " + str(train_batches)
                inputs, targets = batch
                train_err += train(inputs, targets)
                train_batches += 1

            for batch in iterate_minibatches(X_train2, y_train2, params['BATCH_SIZE'],
                                             params['NUM_FEATURES'], params['SEQ_LENGTH'], shuffle=False):
                # if train_batches % 50 == 0:
                #     print "batch number " + str(train_batches)
                inputs, targets = batch
                train_err += train(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0

            for batch in iterate_minibatches(X_val1, y_val1, params['BATCH_SIZE'],
                                             params['NUM_FEATURES'], params['SEQ_LENGTH'], shuffle=False):
                inputs, targets = batch
                err = validate(inputs, targets)
                val_err += err
                # val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} (Composite addresses) took {:.3f}s".format(
                epoch + 1, params['NUM_EPOCHS'], time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(
                train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            # print("  validation accuracy:\t\t{:.2f} %".format(val_acc/val_batches * 100))

            # to prevent overfitting
            # or val_err - old_valerr[0] < 0.001:
            # or old_valerr[0] - val_err < 0.001:
            if val_err - old_valerr[0] > 0.03:
                print "overfitting or model reached saturation...\n"
                print old_valerr
                l_out = old_netout
                break

            old_netout = l_out
            old_valerr[0] = val_err

            val_err = 0
            val_acc = 0
            val_batches = 0

            for batch in iterate_minibatches(X_val2, y_val2, params['BATCH_SIZE'],
                                             params['NUM_FEATURES'], params['SEQ_LENGTH'], shuffle=False):
                inputs, targets = batch
                err = validate(inputs, targets)
                val_err += err
                # val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} (OneLine addresses) took {:.3f}s".format(
                epoch + 1, params['NUM_EPOCHS'], time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(
                train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            # print("  validation accuracy:\t\t{:.2f} %\n".format(val_acc/val_batches * 100))
            old_valerr[1] = val_err

        print "saving the parameters..."
        all_param_values = [p.get_value() for p in all_params]
        np.save("./models/" + str(params), all_param_values)

    return pred


def getlstm():
    pass


def getboth():
    pass


def getrnnboost():
    pass


# The rule based address extractor... hard-coded but still a gem...
def rulEx(paragraphs):
    lens = [len(st.tokenize(p)) for p in paragraphs]
    regexp = re.compile(
        r'\+[0-9][0-9]*|\([0-9]{3}\)|[0-9]{4} [0-9]{4}|([0-9]{3,4}[- ]){2}[0-9]{3,4}')
    possible_addresses = []

    # to retrieve addresses which have phone number at the end
    for idx in range(len(paragraphs)):
        if bool(regexp.search(paragraphs[idx])):  # and lens[idx] <= 9:
            # to collect lines above the phone number
            poss = []
            poss.append((paragraphs[idx], idx))
            temp = idx - 1
            while lens[temp] <= 9:
                poss.append((paragraphs[temp].encode("ascii"), temp))
                temp -= 1

            # address cant be that long
            if len(poss) <= 15:
                possible_addresses += poss

    return set(possible_addresses)


def iterate_minibatches(inputs, targets, batchsize, NUM_FEATURES, SEQ_LENGTH=None,
                        CONV=None, shuffle=False):
    assert len(inputs) == len(targets)
    num_feat = NUM_FEATURES
    if SEQ_LENGTH:
        batches = len(inputs) / (batchsize) + 1
        X = np.zeros((batchsize * (batches), SEQ_LENGTH, num_feat))
        y = np.zeros((batchsize * batches))

        for i in range(len(inputs)):
            # to generate rolling data. We get SEQ_LENGTHs vectors and flatten
            # them
            X[i / SEQ_LENGTH, i % SEQ_LENGTH, :] = inputs[i: i + SEQ_LENGTH].flatten()[:num_feat]
            y[i] = targets[i]

        for i in range(batches):
            yield X[i * batchsize: (i + 1) * batchsize], y[i * batchsize: (i + 1) * batchsize]

    elif CONV:
        batches = len(inputs) / (batchsize * CONV) + 1
        X = np.zeros((batchsize * (batches), 1, CONV, num_feat))
        y = np.zeros((batchsize * (batches), CONV))

        for i in range(len(inputs)):
            X[i / CONV, 0, i % CONV, :] = inputs[i][:num_feat]
            y[i / CONV, i % CONV] = targets[i]

        for i in range(batches):
            yield X[i * batchsize: (i + 1) * batchsize], y[i * batchsize: (i + 1) * batchsize]

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
