#!/usr/bin/env python
import logging
import time
import random
import numpy as np
from sgd import sgd
from utils.sigmoid import sigmoid

def word2vec_model(args, dataset):
    tokens = dataset.tokens()
    nWords = len(tokens)

    startTime=time.time()
    wordVectors = np.concatenate(
        ((np.random.rand(nWords, args.dimVectors) - 0.5) /
           args.dimVectors, np.zeros((nWords, args.dimVectors))),
        axis=0)
    wordVectors = sgd(
        lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, args.window_size,
            negSamplingCostAndGradient),
        wordVectors, 0.3, 40000, None, True, PRINT_EVERY=1)
    # Note that normalization is not called here. This is not a bug,
    # normalizing during training loses the notion of length.

    logging.info("training took %d seconds" % (time.time() - startTime))

    # concatenate the input and output word vectors
    wordVectors = np.concatenate(
        (wordVectors[:nWords,:], wordVectors[nWords:,:]),
        axis=0)
    # wordVectors = wordVectors[:nWords,:] + wordVectors[nWords:,:]


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    # cost from target sample
    uo = outputVectors[target, ].reshape((1, -1))                   # (1, d)
    vc = predicted.reshape((-1, 1))                                 # (d, 1)
    uovc = np.dot(uo, vc)
    s_uovc = sigmoid(uovc)
    cost_from_ts = -1 * np.log(s_uovc)

    # cost from negative sample
    uk = outputVectors[indices[1:], ]                               # (mk, d)
    ukvc = np.dot(uk, vc)                                           # (mk, 1)
    s_ukvc = sigmoid(-1 * ukvc)                                     # (mk, 1)
    cost_from_ns = -1 * np.sum(np.log(s_ukvc))

    cost = cost_from_ts + cost_from_ns

    # gradient of the predict vectors
    grad = np.zeros_like(outputVectors)
    gradPred = ((s_uovc - 1) * outputVectors[target, ] - np.dot((s_ukvc.T - 1), uk)).reshape((-1, ))
    grad_k = -1 * np.dot((s_ukvc - 1.0), predicted.reshape((1, -1)))
    grad_target = (s_uovc - 1) * predicted

    for i, k in enumerate(indices[1:]):
        grad[k] += grad_k[i]
    grad[target] = grad_target

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=negSamplingCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    current_word_idx = tokens[currentWord]
    context_words_idx = [tokens[word] for word in contextWords]

    predict_vec = inputVectors[current_word_idx]

    for context_word in context_words_idx:
        target = context_word
        _cost, gradPred, grad = word2vecCostAndGradient(predict_vec, target, outputVectors, dataset)
        cost += _cost
        gradIn[current_word_idx] += gradPred
        gradOut += grad

    return cost, gradIn, gradOut


def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=negSamplingCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):

        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad
