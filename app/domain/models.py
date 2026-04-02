from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

CONTOURS: List[str] = [
    "SEEKING",
    "FEAR",
    "RAGE",
    "CARE",
    "GRIEF",
    "PLAY",
    "LUST",
    "CALM",
    "ATTACHMENT",
    "JOY",
]


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass
class EvidenceItem:
    source: str
    feature: str
    value: float | str | bool
    effect_on: str | None = None
    weight: float | None = None
    note: str | None = None


@dataclass
class QualityReport:
    signal_quality: float
    sufficiency: float
    noise_risk: float
    flags: List[str] = field(default_factory=list)


@dataclass
class ConfidenceReport:
    overall: float
    per_contour: Dict[str, float]


@dataclass
class ProjectionResult:
    system: str
    scores: Dict[str, float]
    confidence: float
    notes: List[str] = field(default_factory=list)


@dataclass
class ClinicalHypothesis:
    name: str
    probability: float
    severity: float
    evidence: List[EvidenceItem]
    disclaimer: str


@dataclass
class ModalityResult:
    modality: str
    raw_vector: Dict[str, float]
    adjusted_vector: Dict[str, float]
    confidence: ConfidenceReport
    quality: QualityReport
    evidence: List[EvidenceItem]
    warnings: List[str] = field(default_factory=list)
    version: str = "nce-text-0.1.0"


@dataclass
class PersonaResult:
    modality_results: List[ModalityResult]
    fused_vector: Dict[str, float]
    fusion_confidence: ConfidenceReport
    projections: List[ProjectionResult]
    clinical_hypotheses: List[ClinicalHypothesis]
    interpretive_summary: str
    limitations: List[str]
    trace_id: str
    version: str = "v7-skeleton-0.1.0"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
