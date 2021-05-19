from text_prepro import *
from text_models import *

summary_sent_color = 'red'
non_summary_sent_color = 'white'


def view_important_sent_in_text(book_sent, top_sent_inds):
    top_sent_inds = set(sorted(top_sent_inds))
    final_text = ''
    for i, sent in enumerate(book_sent):
        if i in top_sent_inds:
            final_text += f'<font style = "color: {summary_sent_color}"> {sent} </font>'
        else:
            final_text += f'<font style = "color: {non_summary_sent_color}"> {sent} </font>'
    return final_text


class TextSummarizer:

    def __init__(self, text, summary_size):
        self.initial_text = text
        self.initial_text_sent = text2sentences(text)
        self.summary_n_sent = max(1, int(len(self.initial_text_sent) * summary_size/100))

    def summarize(self, lowercase, stemming, lemmatize_method, model_name='text_rank'):
        text_sent_norm = normalize_text(self.initial_text, lowercase=lowercase,
                                        stemming=stemming, lemmatize_method=lemmatize_method)

        text_to_summarize = '. '.join(text_sent_norm)
        summary_sent_ind = apply_text_model(text_to_summarize, self.summary_n_sent, model_name)
        output = view_important_sent_in_text(self.initial_text_sent, summary_sent_ind)
        return output
