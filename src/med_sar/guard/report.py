from __future__ import annotations

from collections import Counter
from typing import Iterable, Dict, Any

from .rules import GuardResult


def aggregate_guard_stats(results: Iterable[GuardResult]) -> Dict[str, Any]:
    total = 0
    passed = 0
    reasons = Counter()
    for res in results:
        total += 1
        if res.passed:
            passed += 1
        else:
            for r in res.reasons:
                reasons[r] += 1
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / max(1, total),
        "reasons": dict(reasons),
    }
