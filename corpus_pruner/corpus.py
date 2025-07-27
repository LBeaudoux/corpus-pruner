from collections import namedtuple
from typing import Iterable, Iterator

from iso639 import Lang
from wordfreq.tokens import tokenize

Sentence = namedtuple("Sentence", ["index", "text", "tokens"])


class Corpus:
    def __init__(self, language_code: str):
        self._lang = Lang(language_code)
        self._sentences = []

    def __iter__(self) -> Iterator[Sentence]:
        return iter(self._sentences)

    def add_sentences(self, sentences: Iterable[str]) -> None:
        start_index = len(self._sentences)
        for i, sentence_text in enumerate(sentences):
            standard_text = standardize_text(sentence_text)
            tokens = tokenize(standard_text, self._lang.pt1)
            sentence = Sentence(
                index=start_index + i, text=sentence_text, tokens=tokens
            )
            self._sentences.append(sentence)

    @property
    def lang(self) -> Lang:
        return self._lang


def standardize_text(any_text):
    return any_text.replace("’", "'").replace(" ", " ")
