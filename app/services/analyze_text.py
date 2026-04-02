from __future__ import annotations

from app.domain.models import PersonaResult
from app.pipelines.text_nce.pipeline import analyze_text as run_text_pipeline


def analyze_text(text: str) -> PersonaResult:
    return run_text_pipeline(text)
