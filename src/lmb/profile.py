"""Persist a known-good Local Model Buddy setup."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SavedProfile:
    model_key: str
    model_id: str
    quantization: str
    backend: str
    context_size: int
    threads: int
    gpu_layers: int
    batch_size: int
    mmap: bool
    kv_cache_quant: str
    model_path: str | None = None
    measured_tokens_per_second: float | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(
                self, "created_at", datetime.now(timezone.utc).isoformat()
            )


def default_profile_path() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "lmb" / "profile.json"


def save_profile(profile: SavedProfile, path: Path | None = None) -> Path:
    destination = path or default_profile_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(profile), indent=2) + "\n")
    temporary.replace(destination)
    return destination


def load_profile(path: Path | None = None) -> SavedProfile | None:
    source = path or default_profile_path()
    if not source.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(source.read_text())
        return SavedProfile(**data)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
