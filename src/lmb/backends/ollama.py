"""Ollama backend."""

from __future__ import annotations

import json
import shutil
import subprocess
import time

import httpx

from .base import Backend, BackendError, BackendStatus, LaunchConfig


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url.rstrip("/")
        self._server: subprocess.Popen[bytes] | None = None

    def status(self) -> BackendStatus:
        executable = shutil.which("ollama")
        version = None
        if executable:
            try:
                result = subprocess.run(
                    [executable, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                version = (result.stdout or result.stderr).strip() or None
            except (OSError, subprocess.TimeoutExpired):
                pass
        running = False
        try:
            response = httpx.get(f"{self.base_url}/api/version", timeout=1.5)
            running = response.is_success
        except httpx.HTTPError:
            pass
        detail = "API is ready" if running else "service is not running"
        return BackendStatus(self.name, bool(executable), running, version, detail)

    def install_instructions(self) -> str:
        return "Install from https://ollama.com/download, then run `ollama serve`."

    def prepare(self, config: LaunchConfig, *, yes: bool = False) -> LaunchConfig:
        status = self.status()
        if not status.installed:
            raise BackendError(self.install_instructions())
        if not yes:
            raise BackendError("Model download was not confirmed.")
        executable = shutil.which("ollama")
        assert executable
        if not status.available:
            try:
                self._server = subprocess.Popen(
                    [executable, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                raise BackendError(f"Could not start Ollama: {exc}") from exc
            for _ in range(20):
                if self.status().available:
                    break
                if self._server.poll() is not None:
                    raise BackendError("Ollama stopped while starting.")
                time.sleep(0.25)
            else:
                self.stop()
                raise BackendError("Ollama did not become ready within 5 seconds.")
        try:
            result = subprocess.run(
                [executable, "pull", config.model_id],
                timeout=None,
                check=False,
            )
        except OSError as exc:
            raise BackendError(f"Could not run Ollama: {exc}") from exc
        if result.returncode:
            raise BackendError(f"Ollama could not pull {config.model_id}.")
        return config

    def stop(self) -> None:
        if self._server and self._server.poll() is None:
            self._server.terminate()
            try:
                self._server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server.kill()
        self._server = None

    def generate(
        self, config: LaunchConfig, prompt: str, *, max_tokens: int = 64
    ) -> str:
        payload = {
            "model": config.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": config.context_size,
                "num_thread": config.threads,
                "num_gpu": config.gpu_layers,
                "num_predict": max_tokens,
            },
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate", json=payload, timeout=300
            )
            response.raise_for_status()
            return str(response.json().get("response", ""))
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise BackendError(f"Ollama generation failed: {exc}") from exc
