# test_with_unittest.py

from unittest import TestCase

from src.text_prepro import *


class TryTesting(TestCase):

    def test_empty_string_preprocessing(self):
        self.assertTrue(text2sentences("") == [] and normalize_sent("") == [] and normalize_text("") == "")

    def test_empty_lists_preprocessing(self):
        self.assertTrue(
            split_words([]) == [] and remove_non_ascii([]) == [] and to_lowercase([]) == [] and
            remove_punctuation([]) == [] and remove_stopwords([]) == [] and lemmatize_nltk([]) == [] and
            # lemmatize_spacy([]) == [] and
            lemmatize_textblob([]) == [] and lemmatize([], 'nltk') == [] and
            stem([]) == [])

