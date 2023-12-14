import logging
from collections import Counter, defaultdict
from itertools import chain
from typing import Dict, Generator

import pandas as pd
from wordfreq import freq_to_zipf, word_frequency, zipf_frequency, zipf_to_freq

from .corpus import Corpus, Sentence
from .ngramfreq import NgramFreq, iter_ngrams

logger = logging.getLogger(__name__)


class FrequencyStats(pd.DataFrame):
    def filter_pervasive_tokens(
        self, min_count, min_zf, min_zf_diff, min_pervasiveness
    ):
        is_numerous = self["count"] >= min_count
        is_frequent = self["zf"] >= min_zf
        is_overrepresented = self["zf_diff"] >= min_zf_diff
        has_high_score = self["pervasiveness"] >= min_pervasiveness
        return self.loc[
            has_high_score & is_numerous & is_frequent & is_overrepresented
        ]


class CorpusPruner:
    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus
        self._nok = set()

    def sentences(self) -> Generator[Sentence, None, None]:
        for sentence in self._corpus:
            if sentence.index not in self._nok:
                yield sentence

    def pruned_sentences(self) -> Generator[Sentence, None, None]:
        for sentence in self._corpus:
            if sentence.index in self._nok:
                yield sentence

    @property
    def frequency_stats(self) -> FrequencyStats:
        token_counts = self._count_tokens()
        tot_tokens = sum(token_counts.values())

        def get_token_frequency_stats(token) -> Dict[str, float]:
            count = token_counts[token]
            ref_count = int(
                word_frequency(token, self._corpus.lang.pt1) * tot_tokens
            )
            count_diff = count - ref_count
            zf = freq_to_zipf(count / tot_tokens)
            ref_zf = zipf_frequency(token, self._corpus.lang.pt1)
            zf_diff = zf - ref_zf
            pervasiveness = zf + zf_diff
            return {
                "token": token,
                "count": count,
                "ref_count": ref_count,
                "count_diff": count_diff,
                "zf": zf,
                "ref_zf": ref_zf,
                "zf_diff": zf_diff,
                "pervasiveness": pervasiveness,
            }

        return FrequencyStats.from_records(
            map(get_token_frequency_stats, token_counts.keys()), index="token"
        )

    def prune_long_sentences(self, max_tokens: int) -> None:
        nok_sentence_indexes = set()
        for sentence in self.sentences():
            if len(sentence.tokens) > max_tokens:
                nok_sentence_indexes.add(sentence.index)
        self._nok |= nok_sentence_indexes

    def prune_unknown_tokens(self) -> None:
        nok_sentence_indexes = set()
        for sentence in self.sentences():
            try:
                for token in sentence.tokens:
                    assert zipf_frequency(token, self._corpus.lang.pt1) > 0
            except AssertionError:
                nok_sentence_indexes.add(sentence.index)
        self._nok |= nok_sentence_indexes

    def prune_pervasive_tekens(
        self, min_count: int = 0, min_zipf_diff: float = 1.0, epoch: int = 3
    ) -> None:
        def get_max_counts():
            token_counts = self._count_tokens()
            tot_tokens = sum(token_counts.values())
            max_counts = {}
            for token in token_counts.keys():
                zf = zipf_frequency(token, self._corpus.lang.pt1)
                max_zf = zf + min_zipf_diff
                max_count = zipf_to_freq(max_zf) * tot_tokens
                max_counts[token] = int(max_count)
            return max_counts

        for e in range(epoch):
            logger.debug(f"epoch #{e}")
            max_counts = get_max_counts()
            counts = defaultdict(int)
            nok_sentence_indexes = set()
            for sentence in self.sentences():
                sentence_counts = {tk: counts[tk] for tk in sentence.tokens}
                try:
                    for token in sentence.tokens:
                        sentence_counts[token] += 1
                        if sentence_counts[token] > min_count:
                            assert sentence_counts[token] < max_counts[token]
                except AssertionError:
                    nok_sentence_indexes.add(sentence.index)
                else:
                    counts.update(sentence_counts)
            self._nok |= nok_sentence_indexes

    def prune_pervasive_ngrams(
        self,
        n: int = 3,
        min_count: int = 1000,
        min_zipf_diff: float = 1.0,
        epoch: int = 3,
    ) -> None:
        def get_max_counts():
            ngram_freq = NgramFreq(self._corpus.lang.pt3, n)
            ngram_counts = self._count_ngrams()
            tot_ngrams = sum(ngram_counts.values())
            max_counts = {}
            for ngram in ngram_counts.keys():
                zf = ngram_freq.zipf_frequency(ngram)
                max_zf = zf + min_zipf_diff
                max_count = zipf_to_freq(max_zf) * tot_ngrams
                max_counts[ngram] = int(max_count)
            return max_counts

        for e in range(epoch):
            logger.debug(f"epoch #{e}")
            max_counts = get_max_counts()
            counts = defaultdict(int)
            nok_sentence_indexes = set()
            for sentence in self.sentences():
                sentence_counts = {
                    ngram: counts[ngram]
                    for ngram in iter_ngrams(sentence.tokens, n=n)
                }
                try:
                    for ngram in iter_ngrams(sentence.tokens, n=n):
                        sentence_counts[ngram] += 1
                        if sentence_counts[ngram] > min_count:
                            assert sentence_counts[ngram] < max_counts[ngram]
                except AssertionError:
                    nok_sentence_indexes.add(sentence.index)
                else:
                    counts.update(sentence_counts)
            self._nok |= nok_sentence_indexes

    def _count_tokens(self):
        return Counter(chain(*[s.tokens for s in self.sentences()]))

    def _count_ngrams(self, n: int = 3):
        ngram_counts = defaultdict(int)
        for sentence in self.sentences():
            for ngram in iter_ngrams(sentence.tokens, n=n):
                ngram_counts[ngram] += 1
        return ngram_counts
