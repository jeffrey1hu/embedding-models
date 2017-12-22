import logging
from utils.treebank import StanfordSentiment
from glove import train_glove
from utils import evaluate
from utils.data_utils import build_cooccur, make_id2word

logging.basicConfig(level=logging.INFO)

# Mock corpus (shamelessly stolen from Gensim word2vec tests)
test_corpus = ("""human interface computer
survey user computer system response time
eps user interface system
system human system eps
user response time
trees
graph trees
graph minors trees
graph minors survey
I like graph and stuff
I like trees and stuff
Sometimes I build a graph
Sometimes I build trees""").split("\n")


dataset = StanfordSentiment()
vocab = dataset.tokens()
word_freq = dataset.tokenfreq()
cooccur = build_cooccur(vocab, word_freq, dataset, window_size=10)
W = train_glove(vocab, cooccur, vector_size=10, iterations=500)

id2word = make_id2word(vocab)

W = evaluate.merge_main_context(W)

def test_similarity():
    similar = evaluate.most_similar(W, vocab, id2word, 'graph')
    logging.debug(similar)

    # assert_equal('trees', similar[0])

test_similarity()