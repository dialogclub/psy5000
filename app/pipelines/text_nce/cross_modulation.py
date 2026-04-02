from __future__ import annotations

from typing import Dict

from app.domain.models import clamp

CROSS = {
    "FEAR": {"SEEKING": -0.15, "CALM": -0.15, "PLAY": -0.12},
    "ATTACHMENT": {"JOY": +0.10, "PLAY": +0.08},
    "CARE": {"ATTACHMENT": +0.08},
    "GRIEF": {"JOY": -0.15, "PLAY": -0.10},
    "RAGE": {"CALM": -0.15, "CARE": -0.05},
    "CALM": {"FEAR": -0.08, "RAGE": -0.08},
    "SEEKING": {"JOY": +0.05},
}


def apply_cross_modulation(vector: Dict[str, float]) -> Dict[str, float]:
    base = dict(vector)
    updated = dict(vector)
    for source, targets in CROSS.items():
        source_value = base[source]
        for target, coef in targets.items():
            updated[target] = clamp(updated[target] + source_value * coef)
    return updated
