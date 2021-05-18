import unicodedata
import re

import spacy
import wordninja
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem.porter import PorterStemmer


def split_words(words: list) -> list:
    new_words = []
    for word in words:
        new_words.extend(wordninja.split(word))
    return new_words


def remove_non_ascii(words: list) -> list:
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words: list) -> list:
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words: list) -> list:
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words: list) -> list:
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def lemmatize_nltk(words: list) -> list:  ### not the best quality
    """Get the normal form of the words in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word)
        lemmas.append(lemma)
    return lemmas


def lemmatize_spacy(words: list) -> list:  ### great but veeeery slow :( 23m for 1 book
    lemmas = []
    nlp = spacy.load('en')
    doc = nlp(' '.join(words))
    for token in doc:
        lemmas.append(token.lemma_)
    return lemmas


def lemmatize_textblob(words: list) -> list:  ### same as nltk
    lemmas = []
    doc = TextBlob(' '.join(words))
    for token in doc.words:
        lemmas.append(token.lemmatize())
    return lemmas


def stem(words):
    """Cut off prefixes and suffixes of the words in list of tokenized words"""
    ps = PorterStemmer()
    new_words = []
    for word in words:
        word = ps.stem(word)
        new_words.append(word)
    return new_words
