"""vLLM backend for systems with a capable discrete GPU."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess

import httpx

from .base import Backend, BackendError, BackendStatus, LaunchConfig


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1") -> None:
        self.base_url = base_url.rstrip("/")

    def status(self) -> BackendStatus:
        installed = bool(shutil.which("vllm")) or importlib.util.find_spec("vllm") is not None
        running = False
        try:
            running = httpx.get(f"{self.base_url}/models", timeout=1.5).is_success
        except httpx.HTTPError:
            pass
        return BackendStatus(
            self.name,
            installed,
            running,
            detail="OpenAI-compatible API is ready" if running else "server is not running",
        )

    def install_instructions(self) -> str:
        return (
            "vLLM requires a supported GPU. See "
            "https://docs.vllm.ai/en/latest/getting_started/installation.html."
        )

    def prepare(self, config: LaunchConfig, *, yes: bool = False) -> LaunchConfig:
        status = self.status()
        if not status.installed:
            raise BackendError(self.install_instructions())
        if not status.available:
            command = (
                f"vllm serve {config.model_id} "
                f"--max-model-len {config.context_size}"
            )
            raise BackendError(f"Start the vLLM server first:\n{command}")
        return config

    def generate(
        self, config: LaunchConfig, prompt: str, *, max_tokens: int = 64
    ) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/completions",
                json={
                    "model": config.model_id,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                },
                timeout=300,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["text"])
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"vLLM generation failed: {exc}") from exc
