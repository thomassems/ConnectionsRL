"""Configurable rewards for a Connections episode."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    correct_group: float = 1.0
    one_away: float = -0.2
    incorrect: float = -1.0
    failure: float = -5.0
    completion: float = 5.0
