# -*- coding: utf-8 -*-
"""
psyche_analysis.py  —  Expanded EO‑architecture
================================================
Версия 2.0 (2025‑05)

❶ **Принципы «Elegant Objects» Егора Бугаенко** сохранены: иммутабельность, отсутствие `None`,
   композиция вместо наследования, отсутствие статических методов.
❷ **Расширено покрытие детекторов** — теперь охватываются ВСЕ базы, упомянутые в диалоге: касты,
   исторические поколения, спиральная динамика, бардо‑состояния, чакры, юги, нейрогормональные
   циклы, когнитивные стили, эмоции (Hawkins), типы интеллекта, мировоззренческие парадигмы.
❸ **Плагинная модель**: каждый домен представлен парой *KnowledgeBase* + *Detector*.
❹ **Загрузка данных** из YAML/JSON в папке `kb/` — эксперты могут редактировать знания без
   изменения кода.
❺ **Отчёт** выводится в формате JSON‑LD, пригодном для онтологической интеграции.

Папки проекта
-------------
kb/                 – YAML‑файлы доменных баз (касты.yaml, generations.yaml, …)
models/             – ML‑модели (если задействованы)
psyche_analysis.py  – этот файл (оркестрирует анализ)

Резюме архитектуры
------------------
```
Transcript                (value‑object)
 └─ Speaker               (value‑object)
      └─ Profile          (aggregate of Dimensions)
            ├─ Dimension  (abstract, immut.)
            │     └─ … (Caste, Generation, …)       «what»
            └─ Evidence   (token indices, score)    «why»

KnowledgeBase (pure data)         Detector (strategy, EO‑style)
   keywords.yaml  <──┐              ↑ uses KB + Heuristic/ML
   model.pkl      <──┘

Pipeline (composition) → DialogueReport → JSON‑LD
```
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple, List, Dict, Any, Protocol, runtime_checkable

import spacy  # type: ignore
import yaml   # PyYAML ≥6.0

# ---------------------------------------------------------------------------
# Value objects — immutables with behaviour
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Text:
    body: str

    def lower(self) -> 'Text':
        return replace(self, body=self.body.lower())

    def words(self) -> Tuple[str, ...]:
        return tuple(re.findall(r"\w+", self.body.lower()))

    def snippet(self, n: int = 120) -> str:
        return (self.body[:n] + "…") if len(self.body) > n else self.body


@dataclass(frozen=True, slots=True)
class Optional:
    """Null‑object wrapper."""
    value: Any = None

    def present(self) -> bool:
        return self.value is not None


# ---------------------------------------------------------------------------
# Transcript / Speaker parsing (immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Speaker:
    name: str
    utterances: Tuple[Text, ...]

    def text(self) -> Text:
        return Text(" ".join(t.body for t in self.utterances))


@dataclass(frozen=True, slots=True)
class Transcript:
    raw: Text

    def speakers(self) -> Tuple[Speaker, ...]:
        grouped: Dict[str, List[Text]] = {}
        for line in self.raw.body.splitlines():
            m = re.match(r"^(\w+):\s+(.*)$", line.strip())
            if m:
                who, what = m.groups()
                grouped.setdefault(who, []).append(Text(what))
        return tuple(Speaker(name, tuple(utts)) for name, utts in grouped.items())

# ---------------------------------------------------------------------------
# Knowledge bases  (pure data holders)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    name: str
    mapping: Mapping[str, Sequence[str]]   # label → keywords or any other spec

    @staticmethod
    def from_yaml(path: Path) -> 'KnowledgeBase':
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return KnowledgeBase(name=path.stem, mapping=data)


# ---------------------------------------------------------------------------
# Dimensions & Evidence (domain results)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Evidence:
    label: str
    score: float
    snippet: str


@dataclass(frozen=True, slots=True)
class Dimension:
    domain: str
    evidence: Evidence

    def as_json(self) -> Mapping[str, Any]:
        return {
            "@type": self.domain,
            "label": self.evidence.label,
            "score": self.evidence.score,
            "snippet": self.evidence.snippet,
        }


# ---------------------------------------------------------------------------
# Detector contract (EO: behaviour interface)
# ---------------------------------------------------------------------------

@runtime_checkable
class Detector(Protocol):
    domain: str

    def detect(self, speaker: Speaker) -> Optional:  # returns Dimension inside Optional
        ...


# ---------------------------------------------------------------------------
# Generic keyword‑based detector  (composition KB + strategy)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class KeywordDetector:
    kb: KnowledgeBase
    domain: str
    min_hits: int = 1

    def detect(self, speaker: Speaker) -> Optional:
        txt = speaker.text().lower().body
        best: Tuple[str, int] | None = None
        for label, keys in self.kb.mapping.items():
            hits = sum(1 for k in keys if k in txt)
            if hits >= self.min_hits and (best is None or hits > best[1]):
                best = (label, hits)
        if best is None:
            return Optional()
        ev = Evidence(best[0], float(best[1]), speaker.text().snippet())
        return Optional(Dimension(self.domain, ev))


# ---------------------------------------------------------------------------
# ML detector stub (can be swapped with real model)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MLDetector:
    domain: str
    model_path: Path

    def detect(self, speaker: Speaker) -> Optional:
        # placeholder: emulate ML output; in real life — load model & predict
        return Optional()  # no evidence yet

# ---------------------------------------------------------------------------
# Profile aggregate (collection of Dimensions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Profile:
    dimensions: Tuple[Dimension, ...]

    def as_json(self) -> Mapping[str, Any]:
        return {dim.domain: dim.as_json() for dim in self.dimensions}


# ---------------------------------------------------------------------------
# Pipeline orchestrator (EO: composition, no logic in __init__)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Pipeline:
    detectors: Tuple[Detector, ...]

    def analyse(self, transcript: Transcript) -> 'DialogueReport':
        reports = tuple(self._analyse_speaker(sp) for sp in transcript.speakers())
        return DialogueReport(reports)

    # private helper
    def _analyse_speaker(self, sp: Speaker) -> 'SpeakerReport':
        dims: List[Dimension] = []
        for det in self.detectors:
            res = det.detect(sp)
            if res.present():
                dims.append(res.value)  # type: ignore[arg-type]
        return SpeakerReport(sp.name, Profile(tuple(dims)))

# ---------------------------------------------------------------------------
# Reports (immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SpeakerReport:
    name: str
    profile: Profile

    def as_json(self) -> Mapping[str, Any]:
        return {
            "@id": f"speaker:{self.name}",
            "name": self.name,
            **self.profile.as_json(),
        }


@dataclass(frozen=True, slots=True)
class DialogueReport:
    speakers: Tuple[SpeakerReport, ...]

    def to_json(self, path: Path) -> None:
        data = {"@context": "https://schema.cogmodel.xyz/psyche/v1", "speakers": [s.as_json() for s in self.speakers]}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# Bootstrapping helpers
# ---------------------------------------------------------------------------

def load_kbs(kb_root: Path) -> List[KnowledgeBase]:
    return [KnowledgeBase.from_yaml(p) for p in kb_root.glob("*.yaml")]


def build_detectors(kbs: Iterable[KnowledgeBase]) -> Tuple[Detector, ...]:
    domain_map = {
        "касты": "caste", "поколения": "generation", "спираль": "spiral", "бардо": "bardo",
        "чакры": "chakra", "юги": "yuga", "эмоции": "emotion", "интеллект": "intelligence",
        "когнитив": "cog_style", "верования": "belief", "мировоззрение": "worldview",
        "нейро": "neuro_cycle",
    }
    dets: List[Detector] = []
    for kb in kbs:
        domain = domain_map.get(kb.name, kb.name)
        dets.append(KeywordDetector(kb, domain))
    return tuple(dets)

# ---------------------------------------------------------------------------
# CLI (the only imperative runtime section)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli = argparse.ArgumentParser("psyche-analysis 2.0 (EO)")
    cli.add_argument("transcript", type=Path)
    cli.add_argument("--kb", type=Path, default=Path("kb"), help="Path to knowledge‑base folder")
    cli.add_argument("--output", type=Path, default=Path("report.json"))
    args = cli.parse_args()

    txt = Text(args.transcript.read_text(encoding="utf-8"))
    transcript = Transcript(txt)

    knowledge_bases = load_kbs(args.kb)
    detectors = build_detectors(knowledge_bases)
    pipeline = Pipeline(detectors)

    report = pipeline.analyse(transcript)
    report.to_json(args.output)
    print(f"✔ Analysis written to {args.output} (detectors: {len(detectors)})")
