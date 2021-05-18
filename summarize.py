from text_prepro import *

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.reduction import ReductionSummarizer

summary_sent_color = 'red'
non_summary_sent_color = 'black'


def view_important_sent_in_text(book_sent, top_sent_inds):
    top_sent_inds = set(sorted(top_sent_inds))
    final_text = ''
    for i, sent in enumerate(book_sent):
        if i in top_sent_inds:
            final_text += f'<font style = "color: {summary_sent_color}"> {sent} </font>'
            final_text += sent
        else:
            final_text += f'<font style = "color: {non_summary_sent_color}"> {sent} </font>'
    return final_text


def text_rank_summarizer(sents: list, summary_n_sent: int):
    text = '. '.join(sents)
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, summary_n_sent)
    summary_ind = []
    for summary_sent in summary:
        summary_ind.append(sents.index(str(summary_sent)[:-1]))
    return summary_ind


class TextSummarizer:

    def __init__(self, text, summary_size=0.5):
        self.initial_text = text
        self.initial_text_sent = text2sentences(text)
        self.summary_n_sent = int(len(self.initial_text_sent) * summary_size)

    def summarize(self, lowercase, stemming, lemmatize_method, model_name=''):
        text_sent_norm = normalize_text(self.initial_text, lowercase=lowercase,
                                        stemming=stemming, lemmatize_method=lemmatize_method)
        summary_sent_ind = text_rank_summarizer(text_sent_norm, 10)
        # output = view_important_sent_in_text(self.initial_text_sent, summary_sent_ind)
        return str(summary_sent_ind) #output
