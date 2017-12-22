#!/usr/bin/env python
import os
from utils.treebank import StanfordSentiment
import matplotlib
from argparse import ArgumentParser
from utils.data_utils import build_cooccur, make_id2word
from glove import train_glove
from functools import partial

matplotlib.use('agg')
import matplotlib.pyplot as plt
from word2vec import *

def parse_args():
    parser = ArgumentParser(
        description=('Build a GloVe vector-space model from the '
                     'provided corpus'))

    parser.add_argument('-m', '--model', choices=["glove", "word2vec"], default="glove", help="Type of model to train.")
    parser.add_argument('--corpus-path', help=('Path of training corpus'))

    parser.add_argument('-w', '--window-size', type=int, default=10,
                           help=('Number of context words to track to '
                                 'left and right of each word'))

    parser.add_argument('--vector-path',
                         help=('Path to which to save computed word '
                               'vectors'))
    parser.add_argument('-s', '--vector-size', type=int, default=100,
                         help=('Dimensionality of output word vectors'))
    parser.add_argument('--iterations', type=int, default=25,
                         help='For GloVe the arg means of training iterations over whole corpus. '
                              'Since Word2vec model is trained by sgd, '
                              'the arg means total random samples')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                         help='Initial learning rate')
    parser.add_argument('--save-every', default=False,
                         help=('Save vectors after every training '
                               'iteration'))
    parser.add_argument('--use-saved', default=True,
                         help=('resume previous train parameters'))

    return parser.parse_args()

def main(args):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    dataset = StanfordSentiment()
    print "Done, read total %d windows" % dataset.word_count()
    print 80 * "="
    print "TRAINING"
    print 80 * "="
    print "Training %s word vectors" % args.model
    if not os.path.exists(args.vector_path):
        os.makedirs(args.vector_path)

    if args.model == 'word2vec':
        word_vectors = word2vec_model(args, dataset)
    else:
        # glove model
        vocab = dataset.tokens()
        word_freq = dataset.tokenfreq()
        cooccur = build_cooccur(vocab, word_freq, dataset, window_size=10)
        word_vectors = train_glove(vocab, cooccur, args.vector_size, args.vector_path, iterations=args.iterations)

if __name__ == '__main__':
    main(parse_args())

#
# visualizeWords = [
#     "the", "a", "an", ",", ".", "?", "!", "``", "''", "--",
#     "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
#     "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
#     "annoying"]
#
# visualizeIdx = [tokens[word] for word in visualizeWords]
# visualizeVecs = wordVectors[visualizeIdx, :]
# temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
# covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
# U,S,V = np.linalg.svd(covariance)
# coord = temp.dot(U[:,0:2])
#
# for i in xrange(len(visualizeWords)):
#     plt.text(coord[i,0], coord[i,1], visualizeWords[i],
#         bbox=dict(facecolor='green', alpha=0.1))
#
# plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
# plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))
#
# plt.savefig('q3_word_vectors.png')
