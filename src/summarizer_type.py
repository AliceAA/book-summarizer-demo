from enum import Enum


class SummarizerType(Enum):
    lsa = 1,
    textrank = 2,
    lexrank = 3,
    luhn = 4,
    klsum = 5
