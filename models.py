import theano.tensor as T
import numpy as np
import lasagne
import theano
import sys

sys.path.insert(0, './database/')


class model(object):
    def __init__(self, name):
        self.name = name

    def getModel(self, name, params):
        if name == "rnn":
            pass

        elif name == "lstm":
            pass

        elif name == "rnn+lstm":
            pass

        elif name == "rnnboost":
            pass

    def _getrnn(self, params):
        print "creating layers"

        l_in = lasagne.layers.InputLayer(shape=(params['BATCH_SIZE'], params['SEQ_LENGTH'], params['NUM_FEATURES']))

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

        target_values = T.dmatrix('target_output')
        network_output = lasagne.layers.get_output(l_out)
        cost = T.mean((network_output - target_values)**2)

        all_params = lasagne.layers.get_all_params(l_out)
        updates = lasagne.updates.adagrad(cost, all_params, params['LEARNING_RATE'])

        print('compiling the network...')
        train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
        compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)
        pred = theano.function([l_in.input_var], network_output, allow_input_downcast=True)

        return all_params, train, pred, compute_cost

    def _getlstm():
        pass

    def _getboth():
        pass

    def _getrnnboost():
        pass

