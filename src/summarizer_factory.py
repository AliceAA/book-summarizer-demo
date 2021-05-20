from summarizer_type import SummarizerType
from summarizers.lsa import Lsa
from summarizers.text_rank import TextRank
from summarizers.lex_rank import LexRank
from summarizers.luhn import Luhn
from summarizers.kl_sum import KLSum

summarizers = [Lsa, TextRank, LexRank, Luhn, KLSum]


class SummarizerFactory:
    @staticmethod
    def create_summarizer(text: str, summary_size: int, stype: SummarizerType):
        for summarizer in summarizers:
            if summarizer.type() == stype:
                return summarizer(text, summary_size)

        return Lsa(text, summary_size)
