from __future__ import annotations

from random import Random
from typing import Callable, List, Sequence

from .library import apply_operator, operator_names

DEFAULT_ORDER = operator_names()


def sample_chain(
    rng: Random,
    *,
    length: int = 2,
    available: Sequence[str] | None = None,
) -> List[str]:
    ops = list(available) if available is not None else list(DEFAULT_ORDER)
    if length <= 0 or length > len(ops):
        raise ValueError(f"length must be in [1, {len(ops)}]")
    idx = sorted(rng.sample(range(len(ops)), k=length))
    return [ops[i] for i in idx]


def parse_ops(ops: str) -> List[str]:
    return [op.strip() for op in ops.split(",") if op.strip()]


def apply_chain(
    text: str,
    ops: Sequence[str],
    *,
    t: float,
    seed: int,
    level_fn: Callable[[str, float], float] | None = None,
) -> str:
    level_fn = level_fn or (lambda _op, t_val: t_val)
    out = text
    for i, op in enumerate(ops):
        level = level_fn(op, t)
        out = apply_operator(op, out, level, seed + i)
    return out
