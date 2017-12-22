#!/usr/bin/env python

import logging
import numpy as np
from utils.general_utils import Progbar
from random import shuffle

def weight(cooccur, x_max, alpha):
    if cooccur < x_max:
        return (cooccur * 1.0 / x_max) ** alpha
    else:
        return 1.


def run_iter(data, learning_rate=0.05, x_max=100, alpha=0.75):
    """
    Run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.

    `data` is a pre-fetched data / weights list where each element is of
    the form

        (v_main, v_context,
         b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence)

    as produced by the `train_glove` function. Each element in this
    tuple is an `ndarray` view into the data structure which contains
    it.

    See the `train_glove` function for information on the shapes of `W`,
    `biases`, `gradient_squared`, `gradient_squared_biases` and how they
    should be initialized.

    The parameters `x_max`, `alpha` define our weighting function when
    computing the cost for two word pairs; see the GloVe paper for more
    details.

    Returns the cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.
    """
    shuffle(data)

    global_cost = 0
    prog = Progbar(target=len(data) // 1000)
    i = 1
    for v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,\
        gradsq_b_main, gradsq_b_context, cooccurrence in data:
        # weight of the sample
        w = weight(cooccurrence, x_max, alpha)
        # compute loss
        L = np.dot(v_main.T, v_context) + b_main[0] + b_context[0] - np.log(cooccurrence)
        sample_cost = L ** 2
        sample_cost *= w
        # gradients of vectors and bias
        grad_v_main = 2 * L * v_context * w
        grad_v_context = 2 * L * v_main * w
        grad_b_main = 2 * L * w
        grad_b_context = 2 * L * w
        # adagrad square cache
        gradsq_W_main += grad_v_main ** 2
        gradsq_W_context += grad_v_context ** 2
        gradsq_b_main += grad_b_main ** 2
        gradsq_b_context += grad_b_context ** 2
        # update vector and bias
        v_main -= learning_rate * grad_v_main / np.sqrt(gradsq_W_main)
        v_context -= learning_rate * grad_v_context / np.sqrt(gradsq_W_context)
        b_main -= learning_rate * grad_b_main / np.sqrt(gradsq_b_main)
        b_context -= learning_rate * grad_b_context / np.sqrt(gradsq_b_context)

        global_cost += sample_cost
        if i % 1000 == 0:
            prog.update(i // 1000, [("train loss", global_cost / i)])
        i += 1

    return global_cost


def train_glove(vocab, cooccurrences, iter_callback=None, vector_size=100,
                iterations=25, **kwargs):
    """
    Train GloVe vectors on the given generator `cooccurrences`, where
    each element is of the form

        (word_i_id, word_j_id, x_ij)

    where `x_ij` is a cooccurrence value $X_{ij}$ as presented in the
    matrix defined by `build_cooccur` and the Pennington et al. (2014)
    paper itself.

    If `iter_callback` is not `None`, the provided function will be
    called after each iteration with the learned `W` matrix so far.

    Keyword arguments are passed on to the iteration step function
    `run_iter`.

    Returns the computed word vector matrix `W`.
    """

    vocab_size = len(vocab)

    # Word vector matrix. This matrix is (2V) * d, where N is the size
    # of the corpus vocabulary and d is the dimensionality of the word
    # vectors. All elements are initialized randomly in the range (-0.5,
    # 0.5]. We build two word vectors for each word: one for the word as
    # the main (center) word and one for the word as a context word.
    #
    # It is up to the client to decide what to do with the resulting two
    # vectors. Pennington et al. (2014) suggest adding or averaging the
    # two for each word, or discarding the context vectors.
    W = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1)

    # Bias terms, each associated with a single vector. An array of size
    # $2V$, initialized randomly in the range (-0.5, 0.5].
    biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

    # Training is done via adaptive gradient descent (AdaGrad). To make
    # this work we need to store the sum of squares of all previous
    # gradients.
    #
    # Like `W`, this matrix is (2V) * d.
    #
    # Initialize all squared gradient sums to 1 so that our initial
    # adaptive learning rate is simply the global learning rate.
    gradient_squared = np.ones((vocab_size * 2, vector_size),
                               dtype=np.float64)

    # Sum of squared gradients for the bias terms.
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

    # Build a reusable list from the given cooccurrence generator,
    # pre-fetching all necessary data.
    #
    # NB: These are all views into the actual data matrices, so updates
    # to them will pass on to the real data structures
    #
    # (We even extract the single-element biases as slices so that we
    # can use them as views)
    data = [(W[i_main], W[i_context + vocab_size],
             biases[i_main : i_main + 1],
             biases[i_context + vocab_size : i_context + vocab_size + 1],
             gradient_squared[i_main], gradient_squared[i_context + vocab_size],
             gradient_squared_biases[i_main : i_main + 1],
             gradient_squared_biases[i_context + vocab_size
                                     : i_context + vocab_size + 1],
             cooccurrence)
            for i_main, i_context, cooccurrence in cooccurrences]

    for i in range(iterations):
        logging.info("\tBeginning iteration %i..", i)

        cost = run_iter(data, **kwargs)

        logging.info("\t\tDone (cost %f)", cost)

        if iter_callback is not None:
            iter_callback(W)

    return W





