"""Konfigurasyon yukleme.

settings.yaml + profile.<name>.yaml okunur, ortam degiskenleriyle ezilebilir:

    TWIN_STORAGE__BACKEND=elastic
    TWIN_PROFILE=steel
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

SETPOINT = "setpoint"
CONTEXT = "context"
MEASUREMENT = "measurement"


# --------------------------------------------------------------------------- #
# Profil (proses sozlugu)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variable:
    """Bir proses degiskeninin tanimi."""

    name: str
    role: str
    unit: str
    tag: str
    desc: str
    asset: str
    op_low: Optional[float] = None
    op_high: Optional[float] = None
    nominal: Optional[float] = None
    max_delta: Optional[float] = None

    @property
    def is_controllable(self) -> bool:
        return self.role == SETPOINT and self.op_low is not None and self.op_high is not None

    @property
    def bounds(self) -> tuple:
        return (float(self.op_low), float(self.op_high))


@dataclass
class Grade:
    code: str
    desc: str
    margin_try_per_ton: float
    share: float
    attrs: Dict[str, Any] = field(default_factory=dict)

    def spec(self, target: str):
        """Bir hedef degisken icin (min, max) spec araligi -- yoksa None."""
        for key in (f"{target}_spec", target.replace("_pct", "").replace("_c", "") + "_spec"):
            if key in self.attrs:
                lo, hi = self.attrs[key]
                return float(lo), float(hi)
        # kagit profilinde reel_moisture_pct -> moisture_spec
        alias = {"reel_moisture_pct": "moisture_spec", "tap_temp_c": "tap_temp_spec"}
        key = alias.get(target)
        if key and key in self.attrs:
            lo, hi = self.attrs[key]
            return float(lo), float(hi)
        return None


@dataclass
class TargetSpec:
    name: str
    kind: str  # regression | classification
    desc: str
    exclude_features: List[str] = field(default_factory=list)
    optimize_direction: Optional[str] = None
    constraint: Optional[str] = None
    horizon_min: Optional[int] = None
    risk_ceiling: Optional[float] = None


@dataclass
class Profile:
    name: str
    line_id: str
    plant: str
    area: str
    description: str
    variables: Dict[str, Variable]
    grades: Dict[str, Grade]
    targets: Dict[str, TargetSpec]
    constraints: Dict[str, Any]
    extras: Dict[str, Any] = field(default_factory=dict)

    # -- kolay erisimciler -------------------------------------------------- #
    def by_role(self, role: str) -> List[Variable]:
        return [v for v in self.variables.values() if v.role == role]

    @property
    def controllables(self) -> List[Variable]:
        return [v for v in self.variables.values() if v.is_controllable]

    @property
    def contexts(self) -> List[Variable]:
        return self.by_role(CONTEXT)

    @property
    def measurements(self) -> List[Variable]:
        return self.by_role(MEASUREMENT)

    def var(self, name: str) -> Variable:
        return self.variables[name]

    def grade(self, code: str) -> Grade:
        return self.grades[code]

    @property
    def tag_to_name(self) -> Dict[str, str]:
        """Musteri tag adi -> canonical degisken adi (ingest icin)."""
        return {v.tag: v.name for v in self.variables.values()}


def _load_profile(name: str) -> Profile:
    path = CONFIG_DIR / f"profile.{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profil bulunamadi: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    variables = {
        vname: Variable(
            name=vname,
            role=spec["role"],
            unit=str(spec.get("unit", "")),
            tag=spec.get("tag", vname),
            desc=spec.get("desc", ""),
            asset=spec.get("asset", raw["line_id"]),
            op_low=spec.get("op_low"),
            op_high=spec.get("op_high"),
            nominal=spec.get("nominal"),
            max_delta=spec.get("max_delta"),
        )
        for vname, spec in raw["variables"].items()
    }

    grades = {}
    for code, g in raw.get("grades", {}).items():
        attrs = {k: v for k, v in g.items() if k not in ("desc", "margin_try_per_ton", "share")}
        grades[code] = Grade(
            code=code,
            desc=g.get("desc", ""),
            margin_try_per_ton=float(g.get("margin_try_per_ton", 0.0)),
            share=float(g.get("share", 0.0)),
            attrs=attrs,
        )

    targets = {
        tname: TargetSpec(
            name=tname,
            kind=t["kind"],
            desc=t.get("desc", ""),
            exclude_features=list(t.get("exclude_features", [])),
            optimize_direction=t.get("optimize_direction"),
            constraint=t.get("constraint"),
            horizon_min=t.get("horizon_min"),
            risk_ceiling=t.get("risk_ceiling"),
        )
        for tname, t in raw.get("targets", {}).items()
    }

    known = {"line_id", "plant", "area", "description", "variables", "grades", "targets", "constraints"}
    extras = {k: v for k, v in raw.items() if k not in known}

    return Profile(
        name=name,
        line_id=raw["line_id"],
        plant=raw.get("plant", "PLANT"),
        area=raw.get("area", "AREA"),
        description=raw.get("description", ""),
        variables=variables,
        grades=grades,
        targets=targets,
        constraints=raw.get("constraints", {}),
        extras=extras,
    )


# --------------------------------------------------------------------------- #
# Ayarlar
# --------------------------------------------------------------------------- #
class Settings(dict):
    """Nokta ile erisilebilen sozluk: settings.get_path('storage.elastic.hosts')."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    """TWIN_STORAGE__BACKEND=elastic  ->  cfg['storage']['backend'] = 'elastic'"""
    for key, value in os.environ.items():
        if not key.startswith("TWIN_"):
            continue
        path = key[len("TWIN_") :].lower().split("__")
        node = cfg
        for part in path[:-1]:
            node = node.setdefault(part, {})
        parsed: Any = value
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            pass
        node[path[-1]] = parsed


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    path = Path(os.environ.get("TWIN_CONFIG", CONFIG_DIR / "settings.yaml"))
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    _apply_env_overrides(cfg)
    return Settings(cfg)


@lru_cache(maxsize=4)
def get_profile(name: Optional[str] = None) -> Profile:
    return _load_profile(name or get_settings().get_path("profile", "paper"))


def resolve_path(relative: str) -> Path:
    """Konfigurasyondaki goreli yollari proje kokune gore cozer."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
