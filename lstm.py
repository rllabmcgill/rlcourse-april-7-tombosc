import theano
import theano.tensor as T
from theano import shared
from theano.tensor.nnet import sigmoid
from theano import scan, shared, function
from theano.compile.nanguardmode import NanGuardMode
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import lasagne
from lasagne.updates import adam, momentum
import pickle
from utils import get_activation_function
from collections import OrderedDict

floatX = theano.config.floatX

def init_weights_bias(shape, activation):
    n_in, n_out = shape
    if activation in ['sigmoid','tanh','softmax','log_softmax','linear']:
        if activation in ['sigmoid','softmax','log_softmax','linear']:
            glorot_coefficient = 4.
        elif activation == 'tanh':
            glorot_coefficient = 1.
        bound = glorot_coefficient * np.sqrt(6. / (n_in + n_out))
        init_weights = np.random.uniform(-bound, bound, (n_in, n_out))
        init_bias = np.random.uniform(-0.05, 0.05, (1, n_out))
    elif activation in ['relu', 'elu']:
        init_weights = np.random.normal(0, 1, (n_in, n_out))
        init_weights *= np.sqrt(2.0/float(n_out))
        init_bias = np.zeros((1,n_out))
    return (init_weights.astype(theano.config.floatX),
            init_bias.astype(theano.config.floatX))

class RLLSTM():
    def __init__(self, n_in, n_h, n_out, output_activation, 
                 gamma, lambda_, K, lr, eps):
        """
        n_in: dim of observation space
        n_h: size of hidden state
        n_out: dim of action space
        output_activation: 'softmax', 'sigmoid', ...
        lr: learning rate
        """
        print "in:", n_in
        print "hid:", n_h
        print "out:", n_out
        self.n_h = n_h
        self.eps = eps
        self.lr = shared(lr)

        # i, f, o gate go together as they are sigmoided
        inner_activations = 'sigmoid'
        iW, ib = init_weights_bias((n_in, 3*n_h), inner_activations)
        iR, _ = init_weights_bias((n_h, 3*n_h), inner_activations)
        W_ifo = shared(iW, name='W_ifo') 
        R_ifo = shared(iR, name='R_ifo') 
        b_ifo = shared(ib, name='b_ifo', broadcastable=(True, False))

        iW, ib = init_weights_bias((n_in, n_h), inner_activations)
        iR, _ = init_weights_bias((n_h, n_h), inner_activations)
        W_z = shared(iW, name='W_z') 
        R_z = shared(iR, name='R_z') 
        b_z = shared(ib, name='b_z', broadcastable=(True, False))

        iW, ib = init_weights_bias((n_h, n_out), output_activation)
        W_out = shared(iW, name='W_out') 
        b_out = shared(ib, name='b_out', broadcastable=(True, False))

        self.c = shared(np.zeros(n_h), name='c')
        self.h = shared(np.zeros(n_h), name='h')

        self.params = [W_ifo, R_ifo, b_ifo, W_z, R_z,
                       b_z, W_out, b_out, self.h, self.c]
        self.params_no_bias = [W_ifo, R_ifo, W_z, R_z, W_out]

        # create eligibility traces as shared variables
        self.e_traces = []
        for p in self.params:
            e_name = 'e' + p.name[1:]
            init_val = np.zeros_like(p.eval())
            self.e_traces.append(shared(init_val, name=e_name))

        #rng = RandomStreams(np.random.RandomState(0).randint(2**30)) 
        
        def compute_advantage(x, h, c):
            """
            x: observation at time t
            h: hidden state at time t-1
            c: cell state at time t-1
            """
            #x = theano.gradient.disconnected_grad(x) 
            #h = theano.gradient.disconnected_grad(h) 
            ifo = T.tanh(T.dot(x, W_ifo) + T.dot(h, R_ifo) + b_ifo)
            i = ifo[:,:n_h]
            f = ifo[:,n_h:-n_h]
            o = ifo[:,-n_h:]

            z = T.tanh(T.dot(x, W_z) + T.dot(h, R_z) + b_z)
            next_c = i * z + f * c
            next_h = o * T.tanh(next_c)

            f = get_activation_function(output_activation)
            A = f(T.dot(next_h, W_out) + b_out)
            #next_x = rng.multinomial(pvals=next_p, dtype=floatX)
            #next_x = disconnected_grad(next_x)
            updates = OrderedDict()
            updates[c] = next_c.flatten()
            updates[h] = next_h.flatten()
            return A.flatten(), updates

        def update(o_t, A_t, a_t, o_t_1, r, g, l, alpha, K):
            """
            updates: updates of
                - params
                - eligibility traces
                - temperature net
            """
            # compute TD error E_td
            A_t_1, _ = compute_advantage(o_t_1, self.h, self.c)
            V_t = T.max(A_t)
            V_t_1 = T.max(A_t_1)
            E_td = V_t + (r + V_t_1 - V_t)/K - A_t[a_t]

            updates = OrderedDict()
            # compute eligibility traces and param updates
            for p, e in zip(self.params, self.e_traces):
                next_e = g * l * e + T.grad(A_t_1[a_t], p)
                updates[e] = next_e

                next_p = p + alpha * E_td * next_e
                if p.broadcastable[0]:
                    updates[p] = T.addbroadcast(next_p, 0)
                else:
                    updates[p] = next_p
            
            return (E_td, V_t, V_t_1, e), updates

        o_t = T.vector('o_t', dtype=floatX)
        o_t_1 = T.vector('o_t_1', dtype=floatX)
        a_t = T.scalar('a_t', dtype='int32')

        A_t, updates_states = compute_advantage(o_t, self.h, self.c)
        r = T.scalar('r', dtype=floatX)
        values, updates_params = update(o_t, A_t, a_t, o_t_1, r, 
                                        gamma, lambda_, self.lr, K)

        self.advantage_f = function([o_t], A_t, updates=updates_states)
        self.update_f = function([o_t, a_t, o_t_1, r],
                              values, 
                              updates=updates_params)

    def reset(self, o):
        self._reset_traces()
        self.c.set_value(np.zeros(self.n_h))
        self.h.set_value(np.zeros(self.n_h))

    def sample_action(self, o):
        next_A = self.advantage_f(o)
        greedy = np.random.choice(2, p=[self.eps, 1-self.eps])
        if greedy == 1:
            picked_a = np.argmax(next_A)
        else:
            picked_a = np.random.choice(len(next_A))
            self._reset_traces() # Watkins or Peng, I don't remember
        return picked_a, next_A

    def _reset_traces(self):
        for e in self.e_traces:
            e.set_value(np.zeros_like(e.get_value()))
    
    def update(self, o_t, a_t, o_t_1, r, done):
        return self.update_f(o_t, a_t, o_t_1, r)

class ErrorPredictor(object):
    def __init__(self, feature_size, lr, beta):
        self.beta = beta
        self.input_var = T.matrix('inputs', dtype=floatX)
        target_var = T.vector('targets', dtype=floatX)
        network = self._build_mlp(feature_size, 6)
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.squared_error(prediction, target_var)
        mean_loss = loss.mean()
        var_loss = loss.var()
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = momentum(mean_loss, params, learning_rate=lr, momentum=0.2)

        self.predict_f = theano.function([self.input_var], prediction)
        self.train_f = theano.function([self.input_var, target_var],
                                       [mean_loss, var_loss], updates=updates)

    def _build_mlp(self, feature_size, hid_size):
        l_in = lasagne.layers.InputLayer(shape=(None, feature_size),
                                         input_var=self.input_var)
        l_hid = lasagne.layers.DenseLayer(
                l_in, num_units=8,
                nonlinearity=lasagne.nonlinearities.sigmoid,
                W=lasagne.init.GlorotUniform())

        l_out = lasagne.layers.DenseLayer(
                l_hid, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)

        return l_out

    def train(self, observations, errors, n_iter = 10):
        observations = np.asarray(observations)
        y_v = self.predict_f(observations[1:]).flatten()
        targets = np.abs(np.asarray(errors[:-1])) + self.beta * y_v
        return self.train_f(observations[:-1], targets)
