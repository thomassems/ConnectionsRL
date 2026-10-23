"""Play Connections with a Claude language model as the agent.

This is a strong reference agent and upper-bound baseline, not the trainable
RL network. Each turn sends the full game state (remaining words, mistakes
left, and the complete feedback history) to the model and asks for the next
four-word guess. The model never sees the private answer key: it works only
from the same observation the environment exposes to any agent.

The solver is decoupled from the Anthropic SDK through the ``ChatModel``
protocol so the game logic can be exercised offline with a fake model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Protocol

from .environment import ConnectionsEnv, GuessRecord, Observation
from .puzzle import load_puzzle
from .rewards import RewardConfig

FourWords = tuple[str, str, str, str]

# Published API prices in USD per million (input, output) tokens, for the
# rough cost estimate in the report. Unlisted models simply skip the estimate.
# DeepSeek rates are the standard-hours cache-miss prices and change often.
_MODEL_PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}

# Each provider is reached through the Anthropic SDK. DeepSeek ships an
# Anthropic-compatible endpoint but does not accept Anthropic-only request
# fields such as ``output_config`` (effort), so those are omitted there.
PROVIDERS = {
    "anthropic": {
        "base_url": None,
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-opus-5-5",
        "supports_effort": True,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/anthropic",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "supports_effort": False,
    },
}

SYSTEM_PROMPT = (
    "You are an expert solver of the New York Times Connections puzzle. "
    "Each puzzle has sixteen words that form four hidden groups of four. "
    "Groups can be joined by meaning, category, wordplay, fill-in-the-blank "
    "phrases, or other tricks, and puzzles often include trap words that look "
    "like they belong to more than one group. You get at most four mistakes. "
    "On each turn you are told the words still in play, how many mistakes "
    "remain, and the result of every previous guess. 'Correct' reveals that "
    "group's category. 'One away' means exactly three of the four words share "
    "a group. 'Incorrect' with no 'one away' means at most two were together, "
    "so no other group contains three of those four words. Use that feedback "
    "to deduce the remaining groups. Think it through, then commit to the four "
    "words you are most confident share a group."
)


@dataclass(frozen=True)
class Completion:
    """A model reply plus token accounting."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ChatModel(Protocol):
    """Anything that can turn a system prompt and messages into a reply."""

    def complete(self, system: str, messages: list[dict]) -> Completion:
        ...


class AnthropicChat:
    """A ``ChatModel`` backed by the Anthropic Messages API.

    The same client drives any provider that speaks the Anthropic Messages
    format; ``base_url`` and ``api_key`` point it at an alternative endpoint
    such as DeepSeek's Anthropic-compatible API. ``effort`` is sent as
    ``output_config`` only when not ``None`` — DeepSeek rejects that field.

    The SDK is imported lazily so the rest of the module (and its tests) do
    not depend on the optional ``anthropic`` package. Pass a preconstructed
    ``client`` to reuse one across puzzles or to inject a stub.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        max_tokens: int = 2048,
        effort: str | None = "medium",
        client: object | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        if client is None:
            import anthropic  # noqa: PLC0415 - optional dependency

            options = {}
            if base_url is not None:
                options["base_url"] = base_url
            if api_key is not None:
                options["api_key"] = api_key
            client = anthropic.Anthropic(**options)
        self._client = client

    def complete(self, system: str, messages: list[dict]) -> Completion:
        params: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
        }
        if self.effort is not None:
            params["output_config"] = {"effort": self.effort}
        response = self._client.messages.create(**params)
        if getattr(response, "stop_reason", None) == "refusal":
            return Completion("", _usage(response, "input"), _usage(response, "output"))
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return Completion(text, _usage(response, "input"), _usage(response, "output"))


def _usage(response: object, kind: str) -> int:
    usage = getattr(response, "usage", None)
    return int(getattr(usage, f"{kind}_tokens", 0) or 0) if usage is not None else 0


def _format_history(history: tuple[GuessRecord, ...]) -> str:
    if not history:
        return "No guesses yet."
    lines = []
    for turn, record in enumerate(history, 1):
        words = ", ".join(record.words)
        if record.feedback == "correct" and record.revealed_group is not None:
            group = record.revealed_group
            lines.append(
                f"{turn}. [{words}] -> CORRECT ({group.color} group: {group.title})"
            )
        elif record.feedback == "one_away":
            lines.append(f"{turn}. [{words}] -> ONE AWAY (three of these share a group)")
        else:
            lines.append(f"{turn}. [{words}] -> INCORRECT")
    return "\n".join(lines)


def build_prompt(observation: Observation) -> str:
    """Render one turn's game state as a user message."""
    remaining = ", ".join(observation.remaining_words)
    solved = len(observation.solved_groups)
    return (
        f"Words still in play ({len(observation.remaining_words)}): {remaining}\n"
        f"Groups solved: {solved} of 4\n"
        f"Mistakes remaining: {observation.mistakes_left} of 4\n\n"
        f"Guess history:\n{_format_history(observation.guess_history)}\n\n"
        "Choose the four words you are most confident form a group. Reply with "
        "only a JSON object of the form {\"words\": [\"W1\", \"W2\", \"W3\", \"W4\"]}, "
        "copying four distinct words exactly from the words still in play."
    )


def parse_guess(text: str) -> list[str] | None:
    """Pull a four-word list out of a model reply, or ``None`` if absent."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("words"), list):
            words = [str(word).strip() for word in data["words"]]
            if len(words) == 4:
                return words
    # Fall back to a bracketed list of quoted words if the JSON was malformed.
    quoted = re.findall(r'"([^"]+)"', text)
    if len(quoted) >= 4:
        return [word.strip() for word in quoted[:4]]
    return None


@dataclass
class _Legality:
    """Case-insensitive view of which guesses the environment will accept."""

    canonical: dict[str, str]  # casefolded remaining word -> exact remaining word
    guessed: set[frozenset[str]]

    def resolve(self, words: list[str]) -> FourWords | None:
        mapped = [self.canonical.get(word.casefold()) for word in words]
        if any(word is None for word in mapped) or len(set(mapped)) != 4:
            return None
        if frozenset(mapped) in self.guessed:
            return None
        return (mapped[0], mapped[1], mapped[2], mapped[3])


def _legality(observation: Observation) -> _Legality:
    canonical = {word.casefold(): word for word in observation.remaining_words}
    guessed = {frozenset(record.words) for record in observation.guess_history}
    return _Legality(canonical, guessed)


def _fallback_guess(observation: Observation) -> FourWords:
    guessed = {frozenset(record.words) for record in observation.guess_history}
    for combo in combinations(sorted(observation.remaining_words), 4):
        if frozenset(combo) not in guessed:
            return combo
    raise RuntimeError("no legal guess remains")


@dataclass
class LLMSolver:
    """Pick each guess by prompting a ``ChatModel`` and validating its reply."""

    model: ChatModel
    max_invalid_retries: int = 2
    input_tokens: int = field(default=0, init=False)
    output_tokens: int = field(default=0, init=False)

    def choose(self, observation: Observation) -> tuple[FourWords, dict]:
        legality = _legality(observation)
        messages = [{"role": "user", "content": build_prompt(observation)}]
        last_text = ""
        for attempt in range(self.max_invalid_retries + 1):
            completion = self.model.complete(SYSTEM_PROMPT, messages)
            self.input_tokens += completion.input_tokens
            self.output_tokens += completion.output_tokens
            last_text = completion.text
            words = parse_guess(completion.text)
            resolved = legality.resolve(words) if words else None
            if resolved is not None:
                return resolved, {"attempts": attempt + 1, "fell_back": False}
            messages.append({"role": "assistant", "content": completion.text or "(no reply)"})
            messages.append({
                "role": "user",
                "content": (
                    "That was not a legal guess. Reply with ONLY the JSON object "
                    "{\"words\": [...]} containing four distinct words copied "
                    "exactly from the words still in play, none from a group you "
                    "already solved and not a combination you already guessed."
                ),
            })
        return _fallback_guess(observation), {
            "attempts": self.max_invalid_retries + 1,
            "fell_back": True,
            "last_reply": last_text,
        }


def play_puzzle(
    path: Path, solver: LLMSolver, seed: int, rewards: RewardConfig | None = None
) -> dict:
    puzzle = load_puzzle(path)
    env = ConnectionsEnv(puzzle, rewards)
    observation = env.reset(seed=seed)
    total_reward = 0.0
    guesses: list[dict] = []
    while observation.status == "in_progress":
        guess, meta = solver.choose(observation)
        result = env.step(guess)
        observation = result.observation
        total_reward += result.reward
        feedback = observation.guess_history[-1]
        guesses.append({
            "words": list(guess),
            "feedback": feedback.feedback,
            "reward": result.reward,
            "revealed_color": feedback.revealed_group.color if feedback.revealed_group else None,
            "attempts": meta["attempts"],
            "fell_back": meta["fell_back"],
        })
    return {
        "puzzle_id": puzzle.puzzle_id,
        "status": observation.status,
        "groups_solved": len(observation.solved_groups),
        "mistakes": 4 - observation.mistakes_left,
        "turns": len(guesses),
        "total_reward": round(total_reward, 6),
        "guesses": guesses,
    }


def summarize(results: list[dict]) -> dict:
    from collections import Counter

    wins = sum(result["status"] == "won" for result in results)
    color_counts = Counter(
        guess["revealed_color"] for result in results for guess in result["guesses"]
        if guess["revealed_color"] is not None
    )
    count = len(results)
    return {
        "count": count,
        "wins": wins,
        "win_rate": wins / count if count else 0.0,
        "mean_groups_solved": sum(r["groups_solved"] for r in results) / count if count else 0.0,
        "mean_reward": sum(r["total_reward"] for r in results) / count if count else 0.0,
        "solved_groups_by_color": dict(sorted(color_counts.items())),
        "fallback_guesses": sum(
            guess["fell_back"] for result in results for guess in result["guesses"]
        ),
    }


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = _MODEL_PRICES.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return round(input_tokens / 1e6 * input_price + output_tokens / 1e6 * output_price, 4)


def load_env_file(path: Path) -> None:
    """Load ``KEY=VALUE`` lines into ``os.environ`` without overriding it."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _validation_paths(puzzles: Path, split: Path) -> tuple[list[Path], str]:
    data = json.loads(split.read_text(encoding="utf-8"))
    manifest = puzzles / "manifest.json"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
    if digest is not None and data.get("manifest_sha256") not in (None, digest):
        raise ValueError("split manifest hash does not match puzzle collection")
    return [puzzles / f"{name}.json" for name in data["splits"]["validation"]], digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzles", type=Path, default=Path("data/puzzles/nyt"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/nyt-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/llm_validation.json"))
    parser.add_argument("--provider", default="anthropic", choices=sorted(PROVIDERS),
                        help="which Anthropic-format API to call")
    parser.add_argument("--model", help="model id (defaults to the provider's)")
    parser.add_argument("--effort", default="medium",
                        choices=["low", "medium", "high", "xhigh", "max"],
                        help="reasoning effort; ignored by providers that lack it")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-invalid-retries", type=int, default=2)
    parser.add_argument("--env-file", type=Path, default=Path(".env"),
                        help="file of KEY=VALUE lines to load before reading the API key")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int,
                        help="play only the first N validation puzzles")
    parser.add_argument("--puzzle", type=Path,
                        help="play a single puzzle file instead of the validation split")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    provider = PROVIDERS[args.provider]
    model = args.model or provider["default_model"]
    effort = args.effort if provider["supports_effort"] else None
    load_env_file(args.env_file)
    api_key = os.environ.get(provider["api_key_env"])
    if not api_key:
        parser.error(
            f"set {provider['api_key_env']} (in the environment or {args.env_file}) "
            f"to call the {args.provider} API"
        )

    if args.puzzle is not None:
        paths = [args.puzzle]
        manifest_hash = None
    else:
        paths, manifest_hash = _validation_paths(args.puzzles, args.split)
        if args.limit is not None:
            paths = paths[:args.limit]
    if not paths:
        parser.error("no puzzles to play")

    chat = AnthropicChat(
        model=model, max_tokens=args.max_tokens, effort=effort,
        base_url=provider["base_url"], api_key=api_key,
    )
    solver = LLMSolver(chat, max_invalid_retries=args.max_invalid_retries)
    results = []
    for position, path in enumerate(paths, 1):
        results.append(play_puzzle(path, solver, args.seed))
        print(f"Played {position}/{len(paths)}: {results[-1]['puzzle_id']} "
              f"({results[-1]['status']}, {results[-1]['groups_solved']}/4)", flush=True)

    report = summarize(results)
    report.update({
        "agent": "llm",
        "provider": args.provider,
        "model": model,
        "effort": effort,
        "seed": args.seed,
        "split": "validation" if args.puzzle is None else "single",
        "test_count_used": 0,
        "manifest_sha256": manifest_hash,
        "input_tokens": solver.input_tokens,
        "output_tokens": solver.output_tokens,
        "estimated_cost_usd": estimate_cost(model, solver.input_tokens, solver.output_tokens),
        "results": results,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    cost = report["estimated_cost_usd"]
    print(
        f"Wins: {report['wins']}/{report['count']} ({report['win_rate']:.1%}); "
        f"mean groups: {report['mean_groups_solved']:.2f}; "
        f"mean reward: {report['mean_reward']:+.2f}"
        + (f"; est. cost ${cost:.2f}" if cost is not None else "")
    )
    print(f"Saved report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
