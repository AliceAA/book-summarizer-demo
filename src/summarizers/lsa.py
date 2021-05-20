import itertools
import numpy as np
from numpy.lib import math
from numpy.linalg import svd
from collections import namedtuple
from operator import attrgetter
from nltk import word_tokenize

from summarizer_type import SummarizerType
from summarizers.text_summarizer import TextSummarizer
from text_prepro import text2sentences

SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order_in_document", "rank",))


class Lsa(TextSummarizer):
    def __init__(self, text, summary_size):
        TextSummarizer.__init__(self, text, summary_size)

    @staticmethod
    def type() -> SummarizerType:
        return SummarizerType.lsa

    def apply_text_model(self, text) -> []:
        text_tokenized = self.tokenize(text)
        matrix = self.create_matrix(text_tokenized)
        matrix = self.compute_term_frequency(matrix)
        u, sigma, v = svd(matrix, full_matrices=False)

        ranks = iter(self.compute_ranks(sigma, v))
        top_sentences = self.get_top_sentences(lambda s: next(ranks))
        indices = [sentence.order_in_document for sentence in top_sentences]
        return indices

    @staticmethod
    def tokenize(text):
        word_sentences = [word_tokenize(sentence) for sentence in text2sentences(text)]
        unique_words = frozenset(itertools.chain(*word_sentences))
        unique_words_dic = dict((w, i) for i, w in enumerate(unique_words))
        return word_sentences, unique_words_dic

    def create_matrix(self, text_tokenized):
        """
        Creates matrix of shape |unique words|×|sentences| where cells
        contains number of occurences of words (rows) in senteces (cols).
        """
        word_sentences, unique_words_dic = text_tokenized

        words_count = len(unique_words_dic)
        sentences_count = len(self.initial_text_sent)

        # create matrix |unique words|×|sentences|
        matrix = np.zeros((words_count, sentences_count))
        for col, sentence in enumerate(word_sentences):
            for word in sentence:
                if word in unique_words_dic:
                    row = unique_words_dic[word]
                    matrix[row, col] += 1

        return matrix

    @staticmethod
    def compute_term_frequency(matrix, smooth=0.4):
        """
        Computes TF metrics for each sentence (column) in the matrix.
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

    @staticmethod
    def compute_ranks(sigma, v):
        dimensions = len(sigma)
        powered_sigma = tuple(s ** 2 if i < dimensions else 0.0 for i, s in enumerate(sigma))
        ranks = []
        # iterate over columns of matrix (rows of transposed matrix)
        for column_vector in v.T:
            rank = sum(s * v ** 2 for s, v in zip(powered_sigma, column_vector))
            ranks.append(math.sqrt(rank))
        return ranks

    def get_top_sentences(self, rank):
        sentence_info = (SentenceInfo(sentence, order_in_document, rank(sentence)) for order_in_document, sentence in
                         enumerate(self.initial_text_sent))

        info = sorted(sentence_info, key=attrgetter("rank"), reverse=True)
        info = info[:int(self.summary_n_sent)]
        info = sorted(info, key=attrgetter("order_in_document"))
        return info
