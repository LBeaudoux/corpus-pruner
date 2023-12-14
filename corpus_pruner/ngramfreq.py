import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Tuple

from iso639 import Lang
from wordfreq import freq_to_zipf, tokenize

from .config import DATA_DIR

logger = logging.getLogger(__name__)


class NgramFreq:
    def __init__(self, language_code: str, n: int):
        self._lang = Lang(language_code)
        self._n = n
        self._freqs = self._load()

    def frequency(self, ngram: Tuple) -> float:
        return self._freqs.get(ngram, 0.0)

    def zipf_frequency(self, ngram: Tuple[str]) -> float:
        f = self.frequency(ngram)
        return freq_to_zipf(self.frequency(ngram)) if f > 0.0 else 0.0

    def build(self, file_path: Path, encoding: str = "utf-8") -> None:
        ngram_counts = defaultdict(int)
        with open(file_path, encoding=encoding) as f:
            for line in f:
                tokens = tokenize(line, self._lang.pt1)
                for ngram in iter_ngrams(tokens, n=self._n):
                    ngram_counts[ngram] += 1
        tot_ngrams = sum(ngram_counts.values())
        self._freqs = {
            ngram: cnt / tot_ngrams for ngram, cnt in ngram_counts.items()
        }
        self._save()

    def _save(self) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(self._freqs, f)

    def _load(self) -> Dict[str, float]:
        try:
            with open(self.path, "rb") as f:
                freqs = pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"{self.path} not found. Please build it.")
            return {}
        else:
            return freqs

    @property
    def path(self) -> Path:
        filename = f"{self._lang.pt3}_{self._n}gram_frequencies.pkl"
        return DATA_DIR.joinpath(filename)


def iter_ngrams(
    tokens: List[str], n: int = 3
) -> Generator[Tuple[str], None, None]:
    for i in range(0, len(tokens) - n + 1):
        yield tuple((tokens[i : i + n]))
