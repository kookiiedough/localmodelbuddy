"""Shared backend contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


class BackendError(RuntimeError):
    """A backend could not complete an operation."""


@dataclass(frozen=True)
class BackendStatus:
    name: str
    installed: bool
    available: bool
    version: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class LaunchConfig:
    model_id: str
    quantization: str
    context_size: int
    threads: int
    gpu_layers: int = 0
    batch_size: int = 128
    mmap: bool = True
    kv_cache_quant: str = "q8_0"
    model_path: Path | None = None
    extra: dict[str, str | int | float | bool] = field(default_factory=dict)


class Backend(ABC):
    name: str

    @abstractmethod
    def status(self) -> BackendStatus:
        """Return backend availability without changing the system."""

    @abstractmethod
    def install_instructions(self) -> str:
        """Return safe platform-specific installation guidance."""

    @abstractmethod
    def prepare(self, config: LaunchConfig, *, yes: bool = False) -> LaunchConfig:
        """Download or configure a model, after any required confirmation."""

    @abstractmethod
    def generate(
        self, config: LaunchConfig, prompt: str, *, max_tokens: int = 64
    ) -> str:
        """Generate text, raising BackendError on failure."""

    def stop(self) -> None:
        """Stop any process owned by this backend."""
