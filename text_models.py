import itertools
from collections import namedtuple
from operator import attrgetter
from text_prepro import *
import numpy as np
from numpy.lib import math
from numpy.linalg import svd as singular_value_decomposition
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.reduction import ReductionSummarizer

SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order_in_document", "rank",))


def summarize_with_lsa(text, n=5):
    lsa_sentences_top = lsa(text, n)
    lsa_indices = [sent.order_in_document for sent in lsa_sentences_top]
    text = view_important_sent_in_text(text, lsa_indices)
    return text


def summarize_with_tf_idf(text, n=5):
    return text


def create_matrix(parsed_text):
    """
    Creates matrix of shape |unique words|×|sentences| where cells
    contains number of occurences of words (rows) in senteces (cols).
    """
    sentences, word_sentences, unique_words_dic = parsed_text

    words_count = len(unique_words_dic)
    sentences_count = len(sentences)

    # create matrix |unique words|×|sentences|
    matrix = np.zeros((words_count, sentences_count))
    for col, sentence in enumerate(word_sentences):
        for word in sentence:
            if word in unique_words_dic:
                row = unique_words_dic[word]
                matrix[row, col] += 1

    return matrix


def compute_term_frequency(matrix, smooth=0.4):
    """
    Computes TF metrics for each sentence (column) in the given matrix.
    You can read more about smoothing parameter at URL below:
    http://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html
    """
    assert 0.0 <= smooth < 1.0

    max_word_frequencies = np.max(matrix, axis=0)
    rows, cols = matrix.shape
    for row in range(rows):
        for col in range(cols):
            max_word_frequency = max_word_frequencies[col]
            if max_word_frequency != 0:
                frequency = matrix[row, col] / max_word_frequency
                matrix[row, col] = smooth + (1.0 - smooth) * frequency

    return matrix


def compute_ranks(sigma, v):
    dimensions = len(sigma)
    powered_sigma = tuple(s ** 2 if i < dimensions else 0.0 for i, s in enumerate(sigma))
    ranks = []
    # iterate over columns of matrix (rows of transposed matrix)
    for column_vector in v.T:
        rank = sum(s * v ** 2 for s, v in zip(powered_sigma, column_vector))
        ranks.append(math.sqrt(rank))
    return ranks


def get_top_sentences(sentences, number, rank):
    sentence_info = (SentenceInfo(sentence, order_in_document, rank(sentence)) for order_in_document, sentence in
                     enumerate(sentences))

    info = sorted(sentence_info, key=attrgetter("rank"), reverse=True)
    info = info[:int(number)]
    info = sorted(info, key=attrgetter("order_in_document"))
    return info


def group_sentences(summary):
    return str([str(sentence) for sentence in summary])


def lsa_preprocess(text):
    sentences = text2sentences(text)
    word_sentences = [normalize_sent(sentence, stopwords=True, stemming=True) for sentence in sentences]
    unique_words = frozenset(itertools.chain(*word_sentences))
    unique_words_dic = dict((w, i) for i, w in enumerate(unique_words))
    return (sentences, word_sentences, unique_words_dic)


def lsa(text, num_sentences=10):
    processed_text = lsa_preprocess(text)
    sentences, _, _ = processed_text

    matrix = create_matrix(processed_text)
    matrix = compute_term_frequency(matrix)
    u, sigma, v = singular_value_decomposition(matrix, full_matrices=False)

    ranks = iter(compute_ranks(sigma, v))

    return get_top_sentences(sentences, num_sentences, lambda s: next(ranks))


def text_rank_summarizer(text: str, summary_n_sent: int):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    init_sents = text2sentences(parser._text)
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, summary_n_sent)
    summary_ind = []
    for summary_sent in summary:
        summary_ind.append(init_sents.index(str(summary_sent)))
    return summary_ind


def apply_text_model(text: str, summary_n_sent: int, model_name=''):
    if model_name == 'text_rank':
        summary_ind = text_rank_summarizer(text, summary_n_sent)
    else:
        summary_ind = [0]
    return summary_ind
