# Rule baselines and the trained network

[← back to the README](../README.md)

All numbers here are on the 141 validation puzzles with seed 7 unless stated otherwise. The 100 held-out test puzzles have not been used.

## The rule baseline

`connections_rl/baseline.py` ranks every legal guess with inspectable clue rules (WordNet shared synsets and hypernyms, shared prefixes and suffixes, one-letter edits) and plays the highest-ranked legal guess. There is no learning and no training.

```sh
.venv/bin/python -m connections_rl.evaluate \
  --split validation \
  --output reports/wordnet_baseline_validation.json
```

`--split {train,validation}` selects puzzles from `data/splits/nyt-v1.json` (the held-out test split is deliberately not selectable). Without `--split` it evaluates every puzzle in `--puzzles`. `--limit N` takes the first N sorted puzzles for a quick check. The report records every guess and feedback result.

| 141 validation puzzles, seed 7 | WordNet only | WordNet + fastText |
| --- | ---: | ---: |
| Puzzle wins | 6 | 10 |
| Mean groups solved | 0.65 | 0.82 |
| Mean reward | -6.15 | -5.69 |
| True groups in initial top 10, out of 564 | 70 | 86 |
| Median initial true-group rank, out of 1,820 | 548 | 382.5 |
| Groups solved by color | Y 27 · G 38 · B 17 · P 10 | Y 37 · G 43 · B 22 · P 13 |

Reports: `reports/wordnet_baseline_validation.json`, `reports/fasttext_wordnet_baseline_validation.json`.

An earlier pair of runs used a different 100-puzzle benchmark (January 1 – February 19 of 2024 and 2025) and is kept for reference in `reports/wordnet_baseline_100.json` and `reports/fasttext_wordnet_baseline_100.json`: 6 and 9 wins, 0.70 and 0.86 mean groups. Those puzzle sets are **not** the validation split, so they are not comparable with anything else in this repository.

## The fastText vector cache

Only the needed English vectors are extracted from [fastText's pretrained text release](https://fasttext.cc/docs/en/crawl-vectors.html). The archive's SHA-256 is stored in the cache and in every evaluation report. The full `.vec.gz` download and the extracted cache are kept out of Git.

```sh
mkdir -p .venv/models
curl --fail --location --output .venv/models/cc.en.300.vec.gz \
  https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
.venv/bin/python scripts/build_fasttext_subset.py \
  --source .venv/models/cc.en.300.vec.gz \
  --puzzles data/puzzles/nyt \
  --output .venv/models/cc.en.300.all-718.sqlite3
```

The extraction command refuses to overwrite an existing cache. The subset must be rebuilt when the puzzle vocabulary changes; unlike a full `.bin` model it has no subword vectors for unseen words. The cache used here, `.venv/models/cc.en.300.all-718.sqlite3`, covers displayed words, category title tokens and WordNet definition tokens for the whole collection. (It was built before the all-emoji date was excluded, so it also carries that date's unused terms.) The older 100-puzzle cache, `cc.en.300.subset.sqlite3`, holds 10,409 matched terms at 300 dimensions and does not cover the larger collection.

Evaluate with vectors by adding `--vector-cache`:

```sh
.venv/bin/python -m connections_rl.evaluate \
  --split validation \
  --vector-cache .venv/models/cc.en.300.all-718.sqlite3 \
  --output reports/fasttext_wordnet_baseline_validation.json
```

The fastText vectors are distributed under [CC BY-SA 3.0](https://fasttext.cc/docs/en/crawl-vectors.html#license).

## The trained network

Install the training dependencies with `.venv/bin/python -m pip install -e '.[train]'`, then:

```sh
.venv/bin/python -m connections_rl.train
```

The command never loads puzzle IDs from the held-out test split. It trains a **321,793-parameter** candidate scorer on the four correct groups and sampled incorrect groups from each training puzzle, then plays training puzzles and updates a Q-value estimate from the environment rewards.

Each legal guess is scored from a permutation-invariant pooled vector for its four words, direct meaning and word-form clue summaries, the current game status, and all prior guesses in order with their feedback and any revealed title and color.

Outputs: the checkpoint at `.venv/checkpoints/candidate_q.pt`, the validation report at `reports/network_training.json`, and answer-independent cached features under `.venv/features/nyt-v1/`. Checkpoints and the feature cache are gitignored.

Watch one validation puzzle:

```sh
.venv/bin/python -m connections_rl.play_network --puzzle data/puzzles/nyt/nyt-2024-10-01.json
```

Pass `--checkpoint .venv/checkpoints/candidate_q.supervised.pt` to compare the warm-start model with the DQN model on the same puzzle.

### Warm start vs. DQN

With seed 7, three supervised epochs and one DQN pass over the 476 training puzzles:

| Validation stage | Wins | Mean groups solved | Mean reward |
| --- | ---: | ---: | ---: |
| Supervised warm start | 14/141 | 0.93 | -4.83 |
| After DQN | 16/141 | 1.15 | -4.50 |

DQN increased solved groups on 44 puzzles and decreased it on 20; eight new wins came with six lost wins. See `reports/network_training.json` for per-puzzle outcomes.

### Seeds

| Seed | Warm-start wins | After DQN | Mean groups, warm → DQN | Zero-group failures, warm → DQN |
| --- | ---: | ---: | ---: | ---: |
| 7 | 14 | 16 | 0.93 → 1.15 | 70 → 49 |
| 17 | 12 | 22 | 0.88 → 1.30 | 73 → 44 |
| 27 | 17 | 19 | 1.01 → 1.27 | 64 → 45 |

Mean after DQN: 19.0 wins (range 16–22). Seed 7 replayed the saved warm-start and DQN checkpoints and its per-puzzle outcomes matched the original report exactly; seeds 17 and 27 were trained again from initialization. Combined data is in `reports/network_diagnostics_summary.json`, with per-turn traces in the seed reports.

Across the three runs, the share of follow-up guesses reusing at least three words from a known `incorrect` guess fell from 326/643 to 69/612 after DQN, so the network learned to use that negative evidence. Initial purple ranking stayed weak: after DQN only 4, 2 and 0 of 141 purple groups ranked in the top ten, depending on seed.

Three runs support the improvement on this validation set. They do not establish performance on the held-out test set.

### Validation diagnostics

Training reports include every validation guess, its Q-value and clue strengths, feedback and reward, and the rank of each unsolved true group among legal guesses at that turn. **The answer-aware ranks are computed only for evaluation** and never enter the model state or the training loss.

The summary counts initial true-group ranks by color, first-guess feedback, zero-group failures, and what the agent chose after wrong feedback, including how often it followed an `incorrect` guess with another candidate sharing three of those words, which cannot be a true group.

Replay diagnostics for a saved checkpoint without retraining:

```sh
.venv/bin/python -m connections_rl.train \
  --evaluate-checkpoint .venv/checkpoints/candidate_q.pt \
  --report reports/network_diagnostics_replay.json
```

`scripts/summarize_network_runs.py` compares completed reports against the same validation puzzle IDs and manifest.

### Repeated RL passes

A second experiment resumed the saved supervised checkpoints for seeds 7, 17 and 27, each with a fresh optimizer, played all 476 training puzzles three times, and measured greedy play on the 141 validation puzzles after every pass. Clue features, network, rewards and split were unchanged. The within-run pass comparison is controlled; its one-pass scores can differ from the runs above, which carried optimizer state from supervised training into RL.

| RL passes | Wins, seeds 7 / 17 / 27 | Mean wins | Mean groups solved | Purple groups initially in top 10, mean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 14 / 12 / 17 | 14.3 | 0.94 | 3.3 |
| 1 | 22 / 18 / 16 | 18.7 | 1.19 | 2.3 |
| 2 | 22 / 22 / 19 | 21.0 | 1.25 | 2.0 |
| 3 | 25 / 20 / 18 | 21.0 | 1.24 | 1.3 |

The second pass improved average validation performance; the third did not improve it overall. Individual puzzle wins changed in both directions. Extra RL experience did not improve initial purple-group recognition, which needs better clue inputs.

```sh
.venv/bin/python -m connections_rl.train \
  --resume-supervised .venv/checkpoints/candidate_q.supervised.pt \
  --seed 7 --rl-epochs 3 --evaluate-each-rl-epoch \
  --output .venv/checkpoints/repeated_rl_seed7.pt \
  --report reports/repeated_rl_seed7.json
```

`--evaluate-each-rl-epoch` saves validation traces in the report and a gitignored checkpoint after each pass. `scripts/summarize_repeated_rl.py` checks that each report used the same validation puzzles and writes the comparison to `reports/repeated_rl_summary.json`.

### The one-away blind spot

A "one away" proves exactly three of the four guessed words share a group, the strongest in-game hint available. The network almost never used it. Across the seed-7 validation run it kept three of the four words on only 24 of 180 follow-up guesses, and 41 of its 54 one-group losses contained a one-away it then abandoned.

An inference-time recovery policy for the network is the clearest unclaimed improvement here. The equivalent nudge for the language model is described in [the language-model doc](llm-agent.md#one-away-recovery) and helps much less there, because that agent already recovers on its own.
