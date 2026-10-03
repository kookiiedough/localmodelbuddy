"""Validate a model setup and measure its practical performance."""

from __future__ import annotations

import time
from dataclasses import dataclass

import psutil

from .backends.base import Backend, BackendError, LaunchConfig


@dataclass(frozen=True)
class SmokeTestResult:
    passed: bool
    response: str
    elapsed_seconds: float
    estimated_tokens: int
    tokens_per_second: float
    available_ram_before: int
    available_ram_after: int
    error: str | None = None


def run_smoke_test(
    backend: Backend,
    config: LaunchConfig,
    *,
    prompt: str = "Reply with one short sentence confirming that you are ready.",
    max_tokens: int = 48,
) -> SmokeTestResult:
    before = psutil.virtual_memory().available
    started = time.perf_counter()
    try:
        response = backend.generate(config, prompt, max_tokens=max_tokens).strip()
    except (BackendError, MemoryError, OSError) as exc:
        elapsed = time.perf_counter() - started
        return SmokeTestResult(
            False, "", elapsed, 0, 0.0, before,
            psutil.virtual_memory().available, str(exc),
        )
    elapsed = max(time.perf_counter() - started, 0.001)
    # A whitespace estimate is deliberately labeled approximate. Runtime APIs
    # do not consistently expose tokenizer counts.
    tokens = max(1, round(len(response.split()) * 1.3)) if response else 0
    speed = tokens / elapsed
    passed = bool(response) and speed >= 0.25
    return SmokeTestResult(
        passed,
        response,
        elapsed,
        tokens,
        speed,
        before,
        psutil.virtual_memory().available,
        None if passed else "The model returned no text or was too slow to be usable.",
    )


def test_with_fallbacks(
    backend: Backend,
    configurations: list[LaunchConfig],
    *,
    prepare: bool = True,
    yes: bool = False,
) -> tuple[LaunchConfig | None, SmokeTestResult, list[str]]:
    """Try configurations in rank order until one behaves acceptably."""
    attempts: list[str] = []
    last = SmokeTestResult(False, "", 0, 0, 0, 0, 0, "No candidates supplied.")
    for config in configurations:
        try:
            ready = backend.prepare(config, yes=yes) if prepare else config
        except BackendError as exc:
            attempts.append(f"{config.model_id}: {exc}")
            last = SmokeTestResult(False, "", 0, 0, 0, 0, 0, str(exc))
            continue
        last = run_smoke_test(backend, ready)
        if last.passed:
            return ready, last, attempts
        attempts.append(f"{config.model_id}: {last.error}")
    return None, last, attempts
