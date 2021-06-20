import re
import unicodedata
import nltk
import spacy
import wordninja
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def text2sentences(text: str) -> list:
    text = text.replace('\\n', '')
    text = re.sub('([.,!?()])', r'\1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    text_sent = nltk.sent_tokenize(text)
    return text_sent


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


def lemmatize_nltk(words: list) -> list:
    """Get the normal form of the words in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word)
        lemmas.append(lemma)
    return lemmas


def lemmatize_spacy(words: list) -> list:
    lemmas = []
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(' '.join(words))
    for token in doc:
        lemmas.append(token.lemma_)
    return lemmas


def lemmatize_textblob(words: list) -> list:
    lemmas = []
    doc = TextBlob(' '.join(words))
    for token in doc.words:
        lemmas.append(token.lemmatize())
    return lemmas


def lemmatize(words: list, method: str) -> list:
    if method == 'nltk':
        return lemmatize_nltk(words)
    elif method == 'spacy':
        return lemmatize_spacy(words)
    elif method == 'textblob':
        return lemmatize_textblob(words)
    else:
        raise Exception(f'The method {method} is not supported for lemmatization')


def stem(words: list) -> list:
    """Cut off prefixes and suffixes of the words in list of tokenized words"""
    ps = PorterStemmer()
    new_words = []
    for word in words:
        word = ps.stem(word)
        new_words.append(word)
    return new_words


def normalize_sent(
        sentence: str,
        lowercase: bool = True,
        stemming: bool = False,
        lemmatize_method: str = ''
) -> list:
    words = nltk.word_tokenize(sentence)
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = split_words(words)
    if lowercase:
        words = to_lowercase(words)
    if lemmatize_method != '':
        words = lemmatize(words, lemmatize_method)
    if stemming:
        words = stem(words)
    return words


def normalize_text(text: str, lowercase: bool = True, stemming: bool = False, lemmatize_method: str = '') -> str:
    text_sent = text2sentences(text)
    text_sent_norm = []
    for sent in text_sent:
        sent_norm = normalize_sent(sent, lowercase=lowercase, stemming=stemming, lemmatize_method=lemmatize_method)
        text_sent_norm.append(sent_norm)
    return '. '.join(' '.join(sent) for sent in text_sent_norm)
