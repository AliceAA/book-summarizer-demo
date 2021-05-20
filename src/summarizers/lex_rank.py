from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from summarizer_type import SummarizerType
from summarizers.text_summarizer import TextSummarizer
from text_prepro import text2sentences


class LexRank(TextSummarizer):
    def __init__(self, text, summary_size):
        TextSummarizer.__init__(self, text, summary_size)

    @staticmethod
    def type() -> SummarizerType:
        return SummarizerType.lexrank

    def apply_text_model(self, text: str) -> []:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        init_sents = text2sentences(parser._text)
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, self.summary_n_sent)
        summary_ind = []
        for summary_sent in summary:
            summary_ind.append(init_sents.index(str(summary_sent)))
        return summary_ind
