"""Direct llama.cpp backend for maximum control on modest hardware."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import httpx

from .base import Backend, BackendError, BackendStatus, LaunchConfig


class LlamaCppBackend(Backend):
    name = "llama.cpp"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or (
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lmb"
        )

    @staticmethod
    def _executable() -> str | None:
        return shutil.which("llama-cli") or shutil.which("main")

    def status(self) -> BackendStatus:
        executable = self._executable()
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
                version = (result.stdout or result.stderr).splitlines()[0]
            except (OSError, subprocess.TimeoutExpired, IndexError):
                pass
        return BackendStatus(
            self.name,
            bool(executable),
            bool(executable),
            version,
            "llama-cli found" if executable else "llama-cli not found",
        )

    def install_instructions(self) -> str:
        return (
            "Install llama.cpp from https://github.com/ggml-org/llama.cpp/"
            "releases and place `llama-cli` on PATH."
        )

    def prepare(self, config: LaunchConfig, *, yes: bool = False) -> LaunchConfig:
        if not self._executable():
            raise BackendError(self.install_instructions())
        if config.model_path and config.model_path.exists():
            return config
        repo = str(config.extra.get("hf_repo", ""))
        filename = str(config.extra.get("hf_file", ""))
        if not repo or not filename:
            raise BackendError("The catalog has no GGUF download for this model.")
        if not yes:
            raise BackendError("Model download was not confirmed.")
        destination = self.cache_dir / repo.replace("/", "--") / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 1024):
                        handle.write(chunk)
            partial.replace(destination)
        except (httpx.HTTPError, OSError) as exc:
            partial.unlink(missing_ok=True)
            raise BackendError(f"GGUF download failed: {exc}") from exc
        return replace(config, model_path=destination)

    def generate(
        self, config: LaunchConfig, prompt: str, *, max_tokens: int = 64
    ) -> str:
        executable = self._executable()
        if not executable or not config.model_path:
            raise BackendError("llama.cpp or the model file is not ready.")
        command = [
            executable,
            "-m",
            str(config.model_path),
            "-p",
            prompt,
            "-n",
            str(max_tokens),
            "-c",
            str(config.context_size),
            "-t",
            str(config.threads),
            "-ngl",
            str(config.gpu_layers),
            "-b",
            str(config.batch_size),
            "--no-display-prompt",
        ]
        if not config.mmap:
            command.append("--no-mmap")
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=300, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackendError(f"llama.cpp generation failed: {exc}") from exc
        if result.returncode:
            raise BackendError(result.stderr.strip() or "llama.cpp exited with an error.")
        return result.stdout.strip()
