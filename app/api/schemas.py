from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnalyzeTextRequest:
    text: str
    language: str = "ru"
    return_evidence: bool = True
    return_projections: bool = True
    return_clinical_hypotheses: bool = True
