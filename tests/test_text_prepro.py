from unittest import TestCase

from src import text_prepro


class PreprocessingTests(TestCase):

    def test_empty_string_preprocessing(self):
        self.assertTrue(text_prepro.text2sentences("") == [] and
                        text_prepro.normalize_sent("") == [] and
                        text_prepro.normalize_text("") == "")

    def test_empty_lists_preprocessing(self):
        self.assertTrue(
            text_prepro.split_words([]) == [] and
            text_prepro.remove_non_ascii([]) == [] and
            text_prepro.to_lowercase([]) == [] and
            text_prepro.remove_punctuation([]) == [] and
            text_prepro.remove_stopwords([]) == [] and
            text_prepro.lemmatize_nltk([]) == [] and
            text_prepro.lemmatize_spacy([]) == [] and
            text_prepro.lemmatize_textblob([]) == [] and
            text_prepro.lemmatize([], 'nltk') == [] and
            text_prepro.stem([]) == []
        )
