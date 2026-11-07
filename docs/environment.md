# The environment and the agent's inputs

[← back to the README](../README.md)

## Rules

`ConnectionsEnv` holds the four groups as a private answer key.

`reset()` shuffles the sixteen words and returns an observation with the remaining words, the solved groups, mistakes left, the complete guess history and the game status. It never exposes unsolved categories.

- A **correct** guess reveals that group's words, title and color.
- A **wrong** guess reports whether it was *one away*, without revealing which group was nearly matched.
- **Failure does not reveal the remaining answers**, exactly as in the real game.

The agent selects four remaining words per step. `available_guesses()` lists the legal combinations and excludes combinations already guessed. A repeated guess, a removed word or any other invalid guess raises `ValueError` without consuming a turn. After solving three groups the agent must still submit the last four words to finish the puzzle.

## Rewards

| Result | Reward |
| --- | ---: |
| Correct group | +1 |
| One away | -0.2 |
| Other wrong guess | -1 |
| Fourth wrong guess (failure) | -5 only |
| Fourth correct group (completion) | +1 group reward and +5 bonus |

These are starting values and can be changed through `RewardConfig`.

```python
from connections_rl import ConnectionsEnv, load_puzzle

puzzle = load_puzzle("data/puzzles/example.json")
env = ConnectionsEnv(puzzle)
observation = env.reset(seed=7)
guess = env.available_guesses()[0]
result = env.step(guess)
print(result.reward, result.observation.guess_history[-1].feedback)
```

Run the checks with `python3 -m unittest discover -s tests -v`.

## Clue features

The agents score each legal four-word guess from the current observation plus cached prior knowledge about words.

`WordKnowledge` keeps a base vector and separate interpretations, each with a source, a description and a soft priority. `ClueEvidence` keeps each provider's raw score, its supporting words and an explanation. Scores from different providers are not assumed to share a scale; they are heuristics, not probabilities or final guess scores.

The providers:

- **`MeaningClueProvider`** reports WordNet shared-synset and shared-hypernym evidence separately from surface vector similarity. It can also check whether valid one-letter edits lead to words with a shared WordNet ancestor, keeping the original word, the edited word and the supporting sense visible in each clue.
- **`WordFormClueProvider`** reports shared prefixes and suffixes and valid one-letter additions and deletions, against a local lexicon.
- Entity and phrase providers can use the same interface later; their channels are masked as unavailable until they exist.

Knowledge sources:

- **`FastTextVectors`** reads a local fastText `.bin` model.
- **`FastTextSubsetVectors`** reads a compact SQLite cache extracted from the official pretrained `.vec.gz` release. The subset has vectors only for requested terms; unlike the full binary it cannot synthesize vectors for unseen words.
- **`WordNetKnowledgeCache`** keeps all WordNet lemma senses, their definition vectors and their taxonomy paths in memory for a run.

```python
from connections_rl import (
    FastTextVectors, MeaningClueProvider, WordFormClueProvider, WordNetKnowledgeCache
)

vectors = FastTextVectors.from_file("path/to/cc.en.300.bin")
knowledge = WordNetKnowledgeCache.from_nltk(vectors)
meanings = MeaningClueProvider(knowledge)
forms = WordFormClueProvider(knowledge.contains)
words = ("CAT", "DOG", "HORSE", "SHEEP")
direct_clues = meanings.find(words)
form_clues = forms.find(words)
deletion_clues = meanings.find_after_edits(words, form_clues, "delete_one_letter")
```

## What the network actually receives

The first network sees fixed-size `ChannelSummary` values: source coverage, evidence count, the two strongest clue strengths, support count, soft priority hints, and ordered relation-specific strengths. WordNet sense rank and tagged-use count are separate hints; neither filters out rare meanings. `ClueBatch` retains every individual evidence item for inspection and later experiments.

The full game observation retains exact words, but the first network compresses each prior group and title to fixed-vector averages and additionally receives their overlap with the proposed guess. It uses direct meaning and word-form summaries; semantic relations after spelling edits remain available to later encoders.

No answer key and no `puzzle_id` is ever part of the agent input.

## Optional dependencies

[fastText's Python package](https://fasttext.cc/docs/en/python-module.html), an [English pretrained binary model](https://fasttext.cc/docs/en/crawl-vectors.html) and [NLTK's WordNet corpus](https://www.nltk.org/howto/wordnet.html) are loaded only when requested; the base project and its tests download none of them.

If the NLTK downloader hits a certificate error, its [official corpus archive](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip) can be saved as `.venv/nltk_data/corpora/wordnet.zip`. NLTK reads that archive directly, and the project finds the local data directory on its own.
