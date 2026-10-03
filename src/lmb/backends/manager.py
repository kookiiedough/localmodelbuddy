"""Select the safest available runtime for the detected hardware."""

from __future__ import annotations

from collections.abc import Iterable

from .base import Backend
from .llamacpp import LlamaCppBackend
from .ollama import OllamaBackend
from .vllm import VLLMBackend


class BackendManager:
    def __init__(self, backends: Iterable[Backend] | None = None) -> None:
        self.backends = list(
            backends or (OllamaBackend(), LlamaCppBackend(), VLLMBackend())
        )

    def statuses(self):
        return [backend.status() for backend in self.backends]

    def get(self, name: str) -> Backend:
        normalized = name.lower().replace("_", "").replace("-", "").replace(".", "")
        for backend in self.backends:
            candidate = backend.name.lower().replace(".", "")
            if normalized == candidate:
                return backend
        raise KeyError(f"Unknown backend: {name}")

    def select(self, hardware, preferred: str | None = None) -> Backend:
        """Prefer an existing backend, using vLLM only with ample GPU memory."""
        if preferred:
            requested = self.get(preferred)
            if requested.status().installed:
                return requested

        installed = {backend.name: backend for backend in self.backends if backend.status().installed}
        gpus = getattr(hardware, "gpus", ())
        max_vram = max(
            (getattr(gpu, "memory_total", 0) or 0 for gpu in gpus),
            default=0,
        )

        if max_vram >= 8 * 1024**3 and "vllm" in installed:
            return installed["vllm"]
        if "ollama" in installed:
            return installed["ollama"]
        if "llama.cpp" in installed:
            return installed["llama.cpp"]
        # Ollama gives newcomers the simplest installation path.
        return self.get("ollama")
