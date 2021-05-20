from summarizer_type import SummarizerType
from summarizers.lsa import Lsa
from summarizers.text_rank import TextRank

summarizers = [Lsa, TextRank]

class SummarizerFactory:
    @staticmethod
    def create_summarizer(text: str, summary_size: int, stype: SummarizerType):
        for summarizer in summarizers:
            if summarizer.type() == stype: return summarizer(text, summary_size)

        return Lsa(text, summary_size)



