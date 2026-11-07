# The language-model reference agent

[← back to the README](../README.md)

`connections_rl/llm_solver.py` plays the same environment with a language model instead of the trained network. It is a strong reference agent and an upper bound: it shows how much of the gap to a perfect solver is a modeling-capacity problem versus a representation problem in the clue features. It is **not** the trainable RL agent and it does not learn from rewards.

## How it plays

Each turn sends the model the same observation any agent sees: the words still in play, mistakes remaining, and the full feedback history. Correct guesses reveal their category; one-away and incorrect results are reported exactly as the environment reports them. The model never receives the private answer key or the `puzzle_id`.

Its reply is parsed as a four-word JSON guess, matched case-insensitively back to the exact remaining words, and checked against the environment's rules. An illegal or unparseable reply is sent back for a limited number of retries before falling back to the first legal combination, so a bad turn never stalls a game. Reports count these as `fallback_guesses`.

The solver talks to the model through a small `ChatModel` protocol, so the game logic is tested offline with scripted fake models and has no hard dependency on the network. It reaches the model with the Anthropic SDK and works with any API speaking the Anthropic Messages format. Two providers are built in:

- **`anthropic`** (default): Claude, keyed by `ANTHROPIC_API_KEY`, default model `claude-opus-5-5`.
- **`deepseek`**: DeepSeek's [Anthropic-compatible endpoint](https://api-docs.deepseek.com/guides/anthropic_api) at `https://api.deepseek.com/anthropic`, keyed by `DEEPSEEK_API_KEY`, default model `deepseek-chat` (`--model deepseek-reasoner` for the reasoning model). DeepSeek does not accept the Anthropic-only `output_config` field, so reasoning `--effort` is omitted for this provider.

## Running it

```sh
.venv/bin/python -m pip install -e '.[llm]'
.venv/bin/python -m connections_rl.llm_solver \
  --provider deepseek \
  --limit 10 \
  --output reports/deepseek_validation_pilot.json
```

The API key is read from the environment or a local `.env` file of `KEY=VALUE` lines (`.env` is gitignored). The held-out test split is never loaded.

`--provider` picks the API; `--model` / `--effort` / `--max-tokens` tune it; `--env-file` points at a key file other than `.env`; `--limit` plays only the first N validation puzzles for a cheap pilot; `--puzzle path/to/puzzle.json` plays a single file.

**Each run calls the API once per turn and costs money.** Start with `--limit` on a small sample before the full 141-puzzle validation set.

The report records the provider and model, every guess and its rationale, the per-puzzle outcome, aggregate wins/groups/reward by the same definitions as the baseline reports, a per-color solve rate in `solved_rate_by_color`, token counts, and a rough cost estimate for the priced models.

## Results

The model is nondeterministic, so a single run is noisy. Repeat it with distinct `--output` paths and compare with `scripts/summarize_llm_runs.py`, which groups repeats of one configuration into a mean and spread, and ranks distinct configurations into a leaderboard with a purple-group column and a per-puzzle win-consistency count. It requires every report to cover the same puzzle set.

```sh
.venv/bin/python scripts/summarize_llm_runs.py \
  reports/deepseek_chat_val_1.json \
  reports/deepseek_chat_val_2.json \
  reports/deepseek_chat_val_3.json \
  --output reports/deepseek_chat_variance.json
```

Three `deepseek-chat` runs over the 141 validation puzzles won **51, 38 and 48**, a mean of 45.7 (32.4%) with a standard deviation of 5.6. They solved a mean of 2.02 groups and cost about $0.11 each. Per-color solve rates averaged yellow 71%, green 55%, blue 41%, purple 35%.

The purple rate is the notable result. It is near zero for the fixed-rule baselines and the trained network, so a general language model recovers most of the purple-group signal the clue features miss.

Across the three runs only **21 puzzles were won every time while 72 were won by at least one**, which sets the repeat count needed for a fair comparison. These are validation observations, not a held-out test score, and the counts will drift because the model is not deterministic.

An earlier full validation run is kept in `reports/deepseek_validation.json`: 52 wins, 2.23 mean groups, $0.1176. It predates the per-color rate field, so the charts use the three variance runs instead.

The reasoning model (`--model deepseek-reasoner`) and `claude-opus-5-5` can be compared the same way, with more time and more cost.

## One-away recovery

A "one away" result proves that exactly three of the four guessed words share a group, the strongest in-game hint available.

By default the solver turns that into an explicit instruction after such a result: if the model can name the odd word, re-try the group with only that word swapped; if it cannot, lock in a different confident group first rather than perseverating on a group it cannot resolve. `active_one_away` finds the most recent one-away whose four words are all still in play, since a hint goes stale once one of its words is solved into another group. Pass `--no-one-away-hint` to disable it; the report records the setting in `one_away_hint`.

This nudge exists because of the trained network's blind spot: across the seed-7 validation run it kept three of the four words on only 24 of 180 follow-up guesses, and 41 of its 54 one-group losses contained a one-away it then abandoned.

For the language model the effect is much smaller. An earlier, blunter version that always told the model to keep three and swap one made it perseverate on groups it could not resolve, and hurt results. The current wording, which lets the model abandon an unidentifiable one-away, removed that harm. On a 40-puzzle slice:

| 40 validation puzzles, `deepseek-chat` | Wins | Mean groups | Fallback guesses | Report |
| --- | ---: | ---: | ---: | --- |
| Hint off | 16 | 2.30 | 3 | `reports/ab_hint_off.json` |
| Hint on, first wording | 14 | 2.15 | 6 | `reports/ab_hint_on.json` |
| Hint on, current wording | 17 | 2.35 | 2 | `reports/ab_hint_on_v2.json` |

Neutral-to-slightly-positive, and well within run-to-run noise at this sample size. The clearer opportunity is an inference-time recovery policy for the network, where the gap is largest.
