# Corpus Pruner

A Python library for pruning the sentences of a corpus.

## Installation

```sh
pip install git+https://github.com/LBeaudoux/corpus-pruner.git
```

## Usage

```python
from tatoebatools import tatoeba
from corpus_pruner.corpus import Corpus
from corpus_pruner.corpus_pruner import CorpusPruner

langcode = "eng"

corpus = Corpus(langcode)
corpus.add_sentences(map(lambda x: x.text, tatoeba.sentences_detailed(langcode)))

corpus_pruner = CorpusPruner(corpus)
corpus_pruner.prune_long_sentences(max_tokens=15)
corpus_pruner.prune_unknown_tokens()
corpus_pruner.prune_pervasive_tokens(threshold_count=20, threshold_zipf_diff=1.0)

ok_sentences = [sentence.text for sentence in corpus_pruner.sentences()]
nok_sentences = [sentence.text for sentence in corpus_pruner.pruned_sentences()]
```