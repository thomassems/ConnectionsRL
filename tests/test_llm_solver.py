"""Exercise the LLM solver's game loop with a scripted fake model."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from connections_rl.environment import ConnectionsEnv
from connections_rl.llm_solver import (
    PROVIDERS,
    AnthropicChat,
    Completion,
    LLMSolver,
    build_prompt,
    estimate_cost,
    load_env_file,
    parse_guess,
    play_puzzle,
    summarize,
)
from connections_rl.puzzle import load_puzzle

EXAMPLE = Path(__file__).resolve().parent.parent / "data/puzzles/example.json"


class ScriptedModel:
    """Return queued replies; record the prompts it was given."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[list[dict]] = []

    def complete(self, system: str, messages: list[dict]) -> Completion:
        self.prompts.append(messages)
        reply = self._replies.pop(0) if self._replies else "{}"
        return Completion(reply, input_tokens=10, output_tokens=5)


class PerfectModel:
    """Solve whichever puzzle it is given from the groups in the file."""

    def __init__(self, path: Path) -> None:
        puzzle = load_puzzle(path)
        self._groups = [list(group.words) for group in puzzle.groups]

    def complete(self, system: str, messages: list[dict]) -> Completion:
        in_play = messages[-1]["content"].splitlines()[0]
        remaining = {word.strip() for word in in_play.split(":", 1)[1].split(",")}
        for words in self._groups:
            if set(words).issubset(remaining):
                return Completion(json.dumps({"words": words}), 20, 8)
        return Completion("{}", 20, 8)


class ParseGuessTests(unittest.TestCase):
    def test_reads_plain_json(self) -> None:
        self.assertEqual(
            parse_guess('{"words": ["RED", "BLUE", "GREEN", "GOLD"]}'),
            ["RED", "BLUE", "GREEN", "GOLD"],
        )

    def test_reads_json_wrapped_in_prose(self) -> None:
        text = 'I think the colors go together.\n{"words": ["RED","BLUE","GREEN","GOLD"]}\nDone.'
        self.assertEqual(parse_guess(text), ["RED", "BLUE", "GREEN", "GOLD"])

    def test_recovers_from_quoted_words(self) -> None:
        self.assertEqual(
            parse_guess('words are "RED" "BLUE" "GREEN" "GOLD" probably'),
            ["RED", "BLUE", "GREEN", "GOLD"],
        )

    def test_rejects_wrong_count(self) -> None:
        self.assertIsNone(parse_guess('{"words": ["RED", "BLUE"]}'))
        self.assertIsNone(parse_guess("no json here"))


class SolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.puzzle = load_puzzle(EXAMPLE)
        self.env = ConnectionsEnv(self.puzzle)
        self.observation = self.env.reset(seed=7)

    def test_case_insensitive_resolution_maps_to_exact_words(self) -> None:
        model = ScriptedModel(['{"words": ["red", "blue", "green", "gold"]}'])
        solver = LLMSolver(model)
        guess, meta = solver.choose(self.observation)
        self.assertEqual(frozenset(guess), frozenset({"RED", "BLUE", "GREEN", "GOLD"}))
        self.assertEqual(meta, {"attempts": 1, "fell_back": False})
        self.assertTrue(set(guess).issubset(self.observation.remaining_words))

    def test_retries_then_accepts_valid_guess(self) -> None:
        model = ScriptedModel([
            '{"words": ["NOPE", "BAD", "WRONG", "MISSING"]}',  # illegal words
            '{"words": ["RED", "BLUE", "GREEN", "GOLD"]}',
        ])
        solver = LLMSolver(model, max_invalid_retries=2)
        guess, meta = solver.choose(self.observation)
        self.assertEqual(frozenset(guess), frozenset({"RED", "BLUE", "GREEN", "GOLD"}))
        self.assertEqual(meta["attempts"], 2)
        self.assertFalse(meta["fell_back"])

    def test_falls_back_to_a_legal_guess_when_model_never_complies(self) -> None:
        model = ScriptedModel(["garbage", "still garbage", "no json"])
        solver = LLMSolver(model, max_invalid_retries=2)
        guess, meta = solver.choose(self.observation)
        self.assertTrue(meta["fell_back"])
        self.assertEqual(len(set(guess)), 4)
        self.assertTrue(set(guess).issubset(self.observation.remaining_words))

    def test_does_not_repeat_a_previously_guessed_combination(self) -> None:
        first = tuple(self.observation.remaining_words[:4])
        self.env.step(first)
        observation = self.env.observation()
        # Model proposes the already-guessed combination, then a fresh one.
        model = ScriptedModel([
            json.dumps({"words": list(first)}),
            json.dumps({"words": list(self.observation.remaining_words[4:8])}),
        ])
        solver = LLMSolver(model)
        guess, meta = solver.choose(observation)
        self.assertNotEqual(frozenset(guess), frozenset(first))
        self.assertFalse(meta["fell_back"])

    def test_tokens_accumulate_across_turns(self) -> None:
        model = PerfectModel(EXAMPLE)
        solver = LLMSolver(model)
        play_puzzle(EXAMPLE, solver, seed=7)
        self.assertEqual(solver.input_tokens, 80)  # four correct groups x 20
        self.assertEqual(solver.output_tokens, 32)


class PlayAndSummaryTests(unittest.TestCase):
    def test_perfect_model_wins_the_example(self) -> None:
        solver = LLMSolver(PerfectModel(EXAMPLE))
        result = play_puzzle(EXAMPLE, solver, seed=7)
        self.assertEqual(result["status"], "won")
        self.assertEqual(result["groups_solved"], 4)
        self.assertEqual(result["mistakes"], 0)
        self.assertEqual(result["turns"], 4)

    def test_summary_aggregates_results(self) -> None:
        solver = LLMSolver(PerfectModel(EXAMPLE))
        results = [play_puzzle(EXAMPLE, solver, seed=7)]
        report = summarize(results)
        self.assertEqual(report["wins"], 1)
        self.assertEqual(report["win_rate"], 1.0)
        self.assertEqual(report["mean_groups_solved"], 4.0)
        self.assertEqual(report["fallback_guesses"], 0)
        self.assertEqual(sum(report["solved_groups_by_color"].values()), 4)

    def test_build_prompt_lists_state_without_answers(self) -> None:
        prompt = build_prompt(self.observation())
        self.assertIn("Mistakes remaining: 4 of 4", prompt)
        self.assertIn("No guesses yet.", prompt)
        for group in load_puzzle(EXAMPLE).groups:
            self.assertNotIn(group.title, prompt)

    def observation(self):
        return ConnectionsEnv(load_puzzle(EXAMPLE)).reset(seed=7)


class CostTests(unittest.TestCase):
    def test_known_model_cost(self) -> None:
        self.assertEqual(estimate_cost("claude-opus-5-5", 1_000_000, 1_000_000), 24.0)

    def test_deepseek_model_cost(self) -> None:
        self.assertEqual(estimate_cost("deepseek-chat", 1_000_000, 1_000_000), 1.37)

    def test_unknown_model_returns_none(self) -> None:
        self.assertIsNone(estimate_cost("made-up-model", 10, 10))


class RecordingClient:
    """Capture the kwargs passed to ``messages.create``."""

    def __init__(self) -> None:
        self.kwargs: dict = {}
        self.messages = self

    def create(self, **kwargs):
        from types import SimpleNamespace

        self.kwargs = kwargs
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text='{"words": ["A","B","C","D"]}')],
            usage=SimpleNamespace(input_tokens=11, output_tokens=3),
        )


class ProviderWiringTests(unittest.TestCase):
    def test_effort_sent_when_present(self) -> None:
        client = RecordingClient()
        AnthropicChat(effort="high", client=client).complete("sys", [])
        self.assertEqual(client.kwargs["output_config"], {"effort": "high"})

    def test_effort_omitted_when_none(self) -> None:
        client = RecordingClient()
        AnthropicChat(effort=None, client=client).complete("sys", [])
        self.assertNotIn("output_config", client.kwargs)

    def test_deepseek_provider_has_compatible_endpoint(self) -> None:
        deepseek = PROVIDERS["deepseek"]
        self.assertEqual(deepseek["base_url"], "https://api.deepseek.com/anthropic")
        self.assertEqual(deepseek["api_key_env"], "DEEPSEEK_API_KEY")
        self.assertFalse(deepseek["supports_effort"])


class EnvFileTests(unittest.TestCase):
    def test_loads_keys_without_overriding_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                '# comment\nDEEPSEEK_API_KEY="sk-secret"\nEMPTY_LINE_BELOW=\n\nALREADY=new\n',
                encoding="utf-8",
            )
            self.assertNotIn("DEEPSEEK_API_KEY", os.environ)
            os.environ["ALREADY"] = "original"
            try:
                load_env_file(path)
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "sk-secret")
                self.assertEqual(os.environ["ALREADY"], "original")  # not overridden
            finally:
                os.environ.pop("DEEPSEEK_API_KEY", None)
                os.environ.pop("ALREADY", None)

    def test_missing_file_is_a_no_op(self) -> None:
        load_env_file(Path("/no/such/.env"))  # must not raise


if __name__ == "__main__":
    unittest.main()
