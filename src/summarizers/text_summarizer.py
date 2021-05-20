from abc import ABC, abstractmethod

from summarizer_type import SummarizerType
from text_prepro import text2sentences, normalize_text

summary_sent_color = 'red'
non_summary_sent_color = 'white'


def highlight_important_sent_in_text(book_sent, top_sent_inds):
    top_sent_inds = set(sorted(top_sent_inds))
    final_text = ''
    for i, sent in enumerate(book_sent):
        if i in top_sent_inds:
            final_text += f'<font style = "color: {summary_sent_color}"> {sent} </font>'
        else:
            final_text += f'<font style = "color: {non_summary_sent_color}"> {sent} </font>'
    return final_text


class TextSummarizer(ABC):
    def __init__(self, text, summary_size):
        self.initial_text = text
        self.initial_text_sent = text2sentences(text)
        self.summary_n_sent = max(1, int(len(self.initial_text_sent) * summary_size / 100))

    @staticmethod
    @abstractmethod
    def type() -> SummarizerType:
        pass

    @abstractmethod
    def apply_text_model(self, text) -> []:
        """Apply specific algorithm defined in derived class."""
        pass

    def summarize(self, lowercase, stemming, lemmatize_method):
        text_normalized = normalize_text(self.initial_text, lowercase=lowercase,
                                         stemming=stemming, lemmatize_method=lemmatize_method)

        summary_sent_ind = self.apply_text_model(text_normalized)
        output = highlight_important_sent_in_text(self.initial_text_sent, summary_sent_ind)
        return output
