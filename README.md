# Connections RL

A reinforcement learning project for training an agent to solve NYT Connections puzzles.

The goal is to explore how an RL agent can learn to identify semantic relationships, word patterns, and other connections between groups of words.

## Project structure

- `connections_rl/puzzle.py` defines and validates the puzzle data shape.
- `connections_rl/environment.py` runs one puzzle episode and applies feedback and rewards.
- `connections_rl/rewards.py` defines the configurable reward values.
- `connections_rl/agent_input.py` defines source-neutral knowledge and clue records for the planned agent encoder.
- `connections_rl/knowledge.py` adapts local fastText vectors and WordNet senses.
- `connections_rl/meanings.py` compares WordNet relationships and fixed vector similarity.
- `connections_rl/word_forms.py` provides shared affix and one-letter edit clues.
- `connections_rl/baseline.py` ranks legal guesses with inspectable clue rules.
- `connections_rl/demo.py` plays a full episode and prints its decisions.
- `data/puzzles/` stores JSON puzzle records. `example.json` is a synthetic example, not a historical NYT puzzle.
- `data/puzzles/nyt/` stores historical puzzles and their source manifest.
- `data/splits/nyt-v1.json` lists the fixed chronological training, validation, and test sets.
- `connections_rl/evaluate.py` runs the clue baseline over the saved benchmark.
- `connections_rl/network.py` builds cached candidate features and defines the small trainable scorer.
- `connections_rl/train.py` runs supervised warm start, game-playing DQN updates, and validation.
- `connections_rl/play_network.py` plays one puzzle with a saved network checkpoint.
- `connections_rl/llm_solver.py` plays puzzles with a Claude language model as a reference agent and upper bound.
- `tests/` checks the episode rules using the synthetic puzzle.

The neural network is implemented as an initial experiment; its training results are reported separately from the fixed-rule baseline.

## Try a puzzle

Create a local environment and install the small WordNet dependency:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[wordnet]'
.venv/bin/python -m nltk.downloader -d .venv/nltk_data wordnet
.venv/bin/python -m connections_rl.demo --show-top 3
```

If the NLTK downloader has a certificate error, its [official corpus archive](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip) can be saved as `.venv/nltk_data/corpora/wordnet.zip`; NLTK reads that archive directly. The demo automatically finds that project-local data directory. Pass `--puzzle path/to/puzzle.json` to try another puzzle and `--seed 7` to repeat the shuffle.

The command uses a clue-based baseline to exercise the environment, **not a trained agent**. It ranks only legal guesses, uses prior feedback, and prints the selected group's strongest clues and the game's response. The supplied puzzle is synthetic and deliberately simple, so solving it is a smoke test rather than evidence of general performance.

The default run uses WordNet only. To add the chosen fixed vector source, install the optional `fasttext` extra and supply a local pretrained `.bin` model:

```sh
.venv/bin/python -m pip install -e '.[fasttext]'
.venv/bin/python -m connections_rl.demo --fasttext path/to/cc.en.300.bin
```

See [fastText's model page](https://fasttext.cc/docs/en/crawl-vectors) for the English pretrained binary. The model is downloaded separately because it is large.

## Historical benchmark

`data/puzzles/nyt/` contains 717 dated historical word puzzles fetched from the [NYT Connections JSON endpoint](https://www.nytimes.com/svc/connections/v2/2024-01-01.json), spanning June 12, 2023–May 31, 2025. Three visual-card dates (April 1 and December 12, 2024, and April 1, 2025) were excluded because this agent uses words as input. The endpoint lists the four categories in color order but does not label the colors; the importer assigns yellow, green, blue, and purple in that order. `manifest.json` records dates, NYT numbers, source response hashes, and exclusions. The original 100-puzzle benchmark covered January 1–February 19, 2024 and January 1–February 19, 2025. Cross-checking the 717 saved puzzles against the [Eyefyre answer archive](https://github.com/Eyefyre/NYT-Connections-Answers) found 714 matches and three dates with differing words: July 19 and December 13, 2023, and September 20, 2024. The saved records use the current NYT responses as the source of record.

The fixed chronological split is in `data/splits/nyt-v1.json`: 476 training puzzles through September 30, 2024; 141 validation puzzles from October 1, 2024–February 19, 2025; and 100 held-out test puzzles from February 20–May 31, 2025. The test period was not used for the baseline results below. The split file includes the exact puzzle IDs and a manifest hash. Keep synthetic or generated puzzles in a separate training-only source so that test scores measure performance on real, unseen puzzles.

To run the WordNet-only baseline on **all** saved puzzles:

```sh
.venv/bin/python -m connections_rl.evaluate \
  --puzzles data/puzzles/nyt \
  --output reports/wordnet_baseline_all.json
```

The report records every guess and feedback result. With seed 7, this baseline won **6 of 100** puzzles, solved **0.70 groups per puzzle**, and earned **-5.84 mean reward**. It won 4 of the 50 puzzles from 2024 and 2 of the 50 from 2025. Before the first guess, 51 of the 400 true groups ranked in the top 10 candidates; the median true-group rank was 506.5 of 1,820 candidates. Purple groups were especially weak: only 4 of 100 ranked in the top 10. This run uses fixed clue rules with WordNet and spelling edits, without vectors or a trained network.

### Pretrained fastText vector comparison

The repo can extract only the needed English vectors from [fastText's pretrained text release](https://fasttext.cc/docs/en/crawl-vectors.html). The ignored local cache built for these 100 puzzles is `.venv/models/cc.en.300.subset.sqlite3`. It contains 10,409 matched terms, has 300 dimensions, and covers all 1,352 distinct displayed puzzle words. The source archive's SHA-256 is stored in the cache and the evaluation report. The full compressed `.vec.gz` archive was removed after extraction to conserve disk space. The subset must be rebuilt when the puzzle vocabulary changes; unlike a full `.bin` model, it has no subword vectors for unseen words.

To build the cache again, download the official archive and run the extraction script:

```sh
mkdir -p .venv/models
curl --fail --location --output .venv/models/cc.en.300.vec.gz \
  https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
.venv/bin/python scripts/build_fasttext_subset.py \
  --source .venv/models/cc.en.300.vec.gz \
  --puzzles data/puzzles/nyt \
  --output .venv/models/cc.en.300.subset.sqlite3
```

The extraction command refuses to overwrite an existing cache. To evaluate with vectors:

```sh
.venv/bin/python -m connections_rl.evaluate \
  --puzzles data/puzzles/nyt \
  --vector-cache .venv/models/cc.en.300.subset.sqlite3 \
  --output reports/fasttext_wordnet_baseline_all.json
```

| Same 100 puzzles, seed 7 | WordNet only | WordNet + fastText |
| --- | ---: | ---: |
| Puzzle wins | 6 | 9 |
| Mean groups solved | 0.70 | 0.86 |
| Mean reward | -5.84 | -5.27 |
| True groups in initial top 10, out of 400 | 51 | 66 |
| Median initial true-group rank, out of 1,820 | 506.5 | 402 |

These results compare fixed clue rules with and without pretrained vectors. They do not measure a trained neural network. The fastText vectors are distributed under [CC BY-SA 3.0](https://fasttext.cc/docs/en/crawl-vectors.html#license); the source archive and extracted cache are kept out of Git.

`scripts/ingest_nyt.py` can fetch other dated windows with `--start YYYY-MM-DD --days N`. It skips dates already in the manifest, validates each new puzzle, and preserves prior records when extending the collection. It accepts `--archive path/to/connections.json` for a color-order cross-check, but that check can reject dates where the archive and current NYT response disagree. Regenerate the split with `.venv/bin/python scripts/make_splits.py` after changing the collection. The endpoint and the archive are external sources, so reruns may differ if either source changes.

## LLM reference agent

`connections_rl/llm_solver.py` plays the same environment with a Claude language model instead of the trained network. It is a strong reference agent and an upper bound: it shows how much of the gap to a perfect solver is a modeling-capacity problem versus a representation problem in the clue features. It is **not** the trainable RL agent and does not learn from rewards.

Each turn sends the model the same observation any agent sees — the words still in play, mistakes remaining, and the full feedback history (correct guesses reveal their category; one-away and incorrect results are reported as the environment reports them). The model never receives the private answer key or `puzzle_id`. Its reply is parsed as a four-word JSON guess, matched case-insensitively back to the exact remaining words, and checked against the environment's rules; an illegal or unparseable reply is sent back for a limited number of retries before falling back to the first legal combination so a bad turn never stalls the game.

The solver talks to the model through a small `ChatModel` protocol, so the game logic is tested offline with scripted fake models and has no hard dependency on the network. It reaches the model with the Anthropic SDK and works with any API that speaks the Anthropic Messages format. Two providers are built in:

- `anthropic` (default): Claude, keyed by `ANTHROPIC_API_KEY`, default model `claude-opus-5-5`.
- `deepseek`: DeepSeek's [Anthropic-compatible endpoint](https://api-docs.deepseek.com/guides/anthropic_api) at `https://api.deepseek.com/anthropic`, keyed by `DEEPSEEK_API_KEY`, default model `deepseek-chat` (pass `--model deepseek-reasoner` for the reasoning model). DeepSeek does not accept the Anthropic-only `output_config` field, so reasoning `--effort` is omitted for this provider.

Install the optional client and run the validation split (the held-out test split is never loaded). The API key is read from the environment or a local `.env` file of `KEY=VALUE` lines (`.env` is gitignored):

```sh
.venv/bin/python -m pip install -e '.[llm]'
.venv/bin/python -m connections_rl.llm_solver \
  --provider deepseek \
  --limit 10 \
  --output reports/deepseek_validation_pilot.json
```

`--provider` picks the API, `--model`/`--effort`/`--max-tokens` tune it, `--env-file` points at a key file other than `.env`, `--limit` plays only the first N validation puzzles for a cheap pilot, and `--puzzle path/to/puzzle.json` plays a single file. **Each run calls the API once per turn and costs money**; start with `--limit` on a small sample before the full 141-puzzle validation set. The report records the provider and model, every guess, the per-puzzle outcome, aggregate wins/groups/reward by the same definitions as the baseline reports, token counts, and a rough cost estimate for the priced models.

## Puzzle data

Each JSON record has a unique `puzzle_id` and exactly four groups. Each group has a category `title`, one of the colors `yellow`, `green`, `blue`, or `purple`, and exactly four `words`. All 16 words must be distinct. The environment will hold the groups as a private answer key and will shuffle the words before showing them to the agent. It will reveal a group's words, title, and color only after a correct guess. The answers will not be revealed after a failed game.

```json
{
  "puzzle_id": "example-001",
  "groups": [
    {"title": "Colors", "color": "yellow", "words": ["RED", "BLUE", "GREEN", "GOLD"]},
    {"title": "Animals", "color": "green", "words": ["DOG", "CAT", "HORSE", "SHEEP"]},
    {"title": "Fruits", "color": "blue", "words": ["APPLE", "PEAR", "BANANA", "PEACH"]},
    {"title": "Shapes", "color": "purple", "words": ["CIRCLE", "SQUARE", "TRIANGLE", "OVAL"]}
  ]
}
```

Python loads this format into the `Puzzle` and `Group` types with `connections_rl.load_puzzle`. The loader converts JSON arrays to tuples; Python code constructing these types directly must also use tuples. The `puzzle_id` stays in the puzzle record for tracking and is not part of the agent's observation. Definitions, embeddings, and phrase data will live in a separate word-level cache so they can be reused across puzzles. Puzzle dates and numbers can be added when historical puzzles are collected.

## Training environment

`ConnectionsEnv` holds the groups as a private answer key. `reset()` shuffles the 16 words and returns an observation with remaining words, solved groups, mistakes left, complete guess history, and game status. It does not expose unsolved categories. A correct guess reveals that group's words, title, and color. A wrong guess reports whether it was one away, without revealing which group was nearly matched. Failure does not reveal the remaining answers.

The agent selects four remaining words per step. `available_guesses()` lists the legal combinations and excludes previously guessed combinations. A repeated guess, a removed word, or any other invalid guess raises `ValueError` without consuming a turn. The agent must submit the last four words after solving the first three groups to finish the puzzle.

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

## Agent input preparation

The planned agent will score each legal four-word guess using the current observation and cached prior knowledge. `WordKnowledge` keeps a base vector and separate interpretations, including source, description, and a soft priority. `ClueEvidence` keeps each provider's raw score, supporting words, and an explanation. Clue providers can cover meanings, named entities, phrases, and word forms. Scores from different providers are not assumed to share a scale.

The first network receives fixed-size `ChannelSummary` values: source coverage, evidence count, the two strongest clue strengths, support count, soft priority hints, and ordered relation-specific strengths. WordNet sense rank and tagged-use count are separate hints; neither filters out rare meanings. `ClueBatch` retains every individual evidence item for inspection and later experiments. Entity and phrase channels are masked as unavailable until their providers exist.

`FastTextVectors` reads an existing local fastText `.bin` model. `FastTextSubsetVectors` reads a compact SQLite cache extracted from the official pretrained fastText `.vec.gz` release. The subset has vectors only for requested terms; it cannot synthesize vectors for unseen words as the full binary can. `WordNetKnowledgeCache` keeps all WordNet lemma senses, their definition vectors, and their taxonomy paths in memory during a run. `WordFormClueProvider` uses a local lexicon to report shared prefixes/suffixes and valid one-letter additions/deletions. `MeaningClueProvider` reports WordNet shared-synset or shared-hypernym evidence separately from surface-vector similarity. It can also check whether valid one-letter edits lead to words with a shared WordNet ancestor. It keeps the original word, edited word, and supporting sense visible in each clue. These clue strengths are heuristics, not probabilities or final guess scores. Entity and phrase providers can use the same interface later. The first network uses direct meaning and word-form summaries; semantic relations after spelling edits remain available to later encoders. No puzzle answer key or `puzzle_id` is part of the agent input.

## First network training

Install the training dependencies with `.venv/bin/python -m pip install -e '.[train]'`. The ignored local vector cache for the expanded collection is `.venv/models/cc.en.300.all-718.sqlite3`. It was built before the all-emoji date was excluded, so it covers the 717 saved puzzles plus that date's unused terms. It includes displayed words, category title tokens, and WordNet definition tokens. The original 100-puzzle cache does not cover the larger collection.

Run `.venv/bin/python -m connections_rl.train` to train and validate. The command never loads puzzle IDs from the held-out test split. It trains a 321,793-parameter candidate scorer on four correct groups and sampled incorrect groups from each training puzzle, then plays training puzzles and updates a Q-value estimate from the environment rewards. Each legal guess is scored with a permutation-invariant pooled vector for its four words, direct meaning and word-form clue summaries, the current game status, and all prior guesses in order with their feedback and any revealed title/color. The full game observation still retains exact words; the first network compresses each prior group and title to fixed-vector averages and also receives its overlap with the proposed guess.

The run writes its checkpoint to `.venv/checkpoints/candidate_q.pt` and its validation report to `reports/network_training.json`. Cached answer-independent puzzle features live in `.venv/features/nyt-v1/`. The checkpoint and feature cache are ignored by Git. Validation is used to compare the warm start and DQN stages; the 100 test puzzles are excluded from model training and evaluation until a final test is deliberately run. The fixed vector cache includes their vocabulary so the same prior word source is available at test time.

Use `.venv/bin/python -m connections_rl.play_network --puzzle data/puzzles/nyt/nyt-2024-10-01.json` to watch one validation puzzle. Pass `--checkpoint .venv/checkpoints/candidate_q.supervised.pt` to compare the warm-start model with the DQN model on the same puzzle.

With seed 7, three supervised epochs and one DQN pass over 476 training puzzles produced these greedy-play results on the same 141 validation puzzles:

| Validation stage | Wins | Mean groups solved | Mean reward |
| --- | ---: | ---: | ---: |
| Supervised warm start | 14/141 | 0.93 | -4.83 |
| After DQN | 16/141 | 1.15 | -4.50 |

DQN increased the number of solved groups on 44 puzzles and decreased it on 20; eight new wins came with six lost wins. The model still fails most puzzles. These are validation results, not a final test score. See `reports/network_training.json` for the per-puzzle outcomes and `--checkpoint` to replay either stage.

### Validation diagnostics

Training reports now include every validation guess, its Q-value and clue strengths, feedback and reward, and the rank of each unsolved true group among legal guesses at that turn. The answer-aware ranks are calculated only for evaluation and never enter the model state or training loss. The summary counts initial true-group ranks by color, first-guess feedback, zero-group failures, and what the agent chose after wrong feedback. In particular, after an `incorrect` guess, another candidate sharing three of those words cannot be a true group; the report counts how often the agent still makes such a follow-up guess.

To replay validation diagnostics for a saved checkpoint without retraining:

```sh
.venv/bin/python -m connections_rl.train \
  --evaluate-checkpoint .venv/checkpoints/candidate_q.pt \
  --report reports/network_diagnostics_replay.json
```

Training with another `--seed` and distinct `--output` and `--report` paths checks whether an apparent improvement depends on initialization or exploration. `scripts/summarize_network_runs.py` compares completed reports against the same validation puzzle IDs and manifest. The held-out test split stays out of these diagnostics.

The diagnostic rerun used seeds 7, 17, and 27 on the same 141 validation puzzles. Seed 7 replayed the saved warm-start and DQN checkpoints; its per-puzzle outcomes matched the original report exactly. Seeds 17 and 27 were trained again from initialization. The combined data is in `reports/network_diagnostics_summary.json`, with individual turn traces in the seed reports.

| Seed | Warm-start wins | After DQN | Mean groups, warm → DQN | Zero-group failures, warm → DQN |
| --- | ---: | ---: | ---: | ---: |
| 7 | 14 | 16 | 0.93 → 1.15 | 70 → 49 |
| 17 | 12 | 22 | 0.88 → 1.30 | 73 → 44 |
| 27 | 17 | 19 | 1.01 → 1.27 | 64 → 45 |

Across all three runs, the share of follow-up guesses that reused at least three words from a known `incorrect` guess fell from 326/643 to 69/612 after DQN. This suggests the network learned to use some negative feedback. The initial ranking of purple groups remains weak: after DQN, only 4, 2, and 0 of 141 purple groups ranked in the top ten, depending on seed. The two added seeds support the improvement on this validation set, but three runs do not establish performance on the held-out test set.

### Repeated RL passes

To test whether more experience on the same training puzzles helps, a second experiment resumed the saved supervised checkpoints for seeds 7, 17, and 27. Each run used a fresh optimizer, played all 476 training puzzles three times, and measured greedy play on the same 141 validation puzzles after every pass. The clue features, network, rewards, and data split were unchanged. The within-run pass comparison is controlled; its one-pass scores can differ from the earlier runs, which carried optimizer state from supervised training into RL.

| RL passes | Wins, seeds 7 / 17 / 27 | Mean wins | Mean groups solved | Purple groups initially in top 10, mean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 14 / 12 / 17 | 14.3 | 0.94 | 3.3 |
| 1 | 22 / 18 / 16 | 18.7 | 1.19 | 2.3 |
| 2 | 22 / 22 / 19 | 21.0 | 1.25 | 2.0 |
| 3 | 25 / 20 / 18 | 21.0 | 1.24 | 1.3 |

The second pass improved average validation performance; the third did not improve it overall. Individual puzzle wins changed in both directions. Extra RL experience did not improve initial purple-group recognition, which needs better clue inputs. These are validation observations across only three seeds, not a claim about held-out test performance.

`--evaluate-each-rl-epoch` saves validation traces in the report and an ignored checkpoint after each pass. For example, with a saved seed-7 supervised checkpoint:

```sh
.venv/bin/python -m connections_rl.train \
  --resume-supervised .venv/checkpoints/candidate_q.supervised.pt \
  --seed 7 --rl-epochs 3 --evaluate-each-rl-epoch \
  --output .venv/checkpoints/repeated_rl_seed7.pt \
  --report reports/repeated_rl_seed7.json
```

`scripts/summarize_repeated_rl.py` checks that each report used the same validation puzzles and writes the comparison to `reports/repeated_rl_summary.json`. The three per-seed reports contain the per-puzzle traces; their checkpoints remain outside Git.

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

The optional sources are [fastText's Python package](https://fasttext.cc/docs/en/python-module.html), an [English pretrained binary model](https://fasttext.cc/docs/en/crawl-vectors.html), and [NLTK's WordNet corpus](https://www.nltk.org/howto/wordnet.html). They are loaded only when requested; the base project and tests do not download them.
