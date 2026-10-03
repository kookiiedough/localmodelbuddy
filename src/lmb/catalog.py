"""Curated metadata for small and medium local GGUF models.

Sizes are approximate download/on-disk sizes, not runtime memory requirements.
They intentionally use decimal GB and are rounded to make the catalog useful for
planning rather than artifact verification.  Model metadata is kept as plain,
immutable values so callers and tests never depend on network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class QuantSpec:
    """A GGUF quantization option available for a model."""

    name: str
    size_gb: float
    quality: float
    bits: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("quantization name cannot be empty")
        if self.size_gb <= 0:
            raise ValueError("quantization size_gb must be positive")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quantization quality must be between 0 and 1")
        if self.bits <= 0:
            raise ValueError("quantization bits must be positive")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Metadata needed to compare, download, and run a local model."""

    id: str
    name: str
    parameter_count_b: float
    task_scores: Mapping[str, int] = field(hash=False)
    quality: float
    ollama_model: str
    ollama_tag: str
    hf_repo: str
    hf_filename_pattern: str
    context_length: int
    architecture: str
    layers: int
    quantizations: tuple[QuantSpec, ...]

    def __post_init__(self) -> None:
        if not self.id or not self.name:
            raise ValueError("model id and name cannot be empty")
        if self.parameter_count_b <= 0:
            raise ValueError("parameter_count_b must be positive")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be between 0 and 1")
        if self.context_length <= 0 or self.layers <= 0:
            raise ValueError("context_length and layers must be positive")
        scores = dict(self.task_scores)
        if not scores or any(not 0 <= score <= 100 for score in scores.values()):
            raise ValueError("task scores must be integers between 0 and 100")
        if not self.quantizations:
            raise ValueError("at least one quantization is required")
        if len({quant.name.casefold() for quant in self.quantizations}) != len(
            self.quantizations
        ):
            raise ValueError("quantization names must be unique")
        object.__setattr__(self, "task_scores", MappingProxyType(scores))

    @property
    def params_b(self) -> float:
        """Short alias for :attr:`parameter_count_b`."""

        return self.parameter_count_b

    @property
    def ollama_ref(self) -> str:
        """Return the directly usable Ollama model reference."""

        return f"{self.ollama_model}:{self.ollama_tag}"

    def quant(self, name: str) -> QuantSpec:
        """Look up one of this model's quantizations, case-insensitively."""

        wanted = name.casefold()
        for quantization in self.quantizations:
            if quantization.name.casefold() == wanted:
                return quantization
        raise KeyError(f"unknown quantization {name!r} for model {self.id!r}")


def _quants(q4_k_m: float, q5_k_m: float, q8_0: float) -> tuple[QuantSpec, ...]:
    return (
        QuantSpec("Q2_K", round(q4_k_m * 0.72, 2), 0.70, 2.6),
        QuantSpec("Q4_K_M", q4_k_m, 0.88, 4.5),
        QuantSpec("Q5_K_M", q5_k_m, 0.94, 5.5),
        QuantSpec("Q8_0", q8_0, 0.99, 8.0),
    )


def _scores(chat: int, reasoning: int, coding: int) -> Mapping[str, int]:
    return {"chat": chat, "reasoning": reasoning, "coding": coding}


# Scores are relative suitability estimates (0-100), not benchmark claims.
# Filenames use the naming convention of the referenced community GGUF repos.
CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        "tinyllama-1.1b-chat", "TinyLlama 1.1B Chat", 1.10,
        _scores(48, 32, 30), 0.48, "tinyllama", "1.1b-chat",
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "tinyllama-1.1b-chat-v1.0.{quant}.gguf", 2_048, "Llama", 22,
        _quants(0.67, 0.78, 1.17),
    ),
    ModelSpec(
        "llama-3.2-1b-instruct", "Llama 3.2 1B Instruct", 1.24,
        _scores(66, 51, 48), 0.67, "llama3.2", "1b",
        "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-{quant}.gguf", 131_072, "Llama 3", 16,
        _quants(0.81, 0.93, 1.32),
    ),
    ModelSpec(
        "llama-3.2-3b-instruct", "Llama 3.2 3B Instruct", 3.21,
        _scores(76, 64, 58), 0.77, "llama3.2", "3b",
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-{quant}.gguf", 131_072, "Llama 3", 28,
        _quants(2.02, 2.32, 3.42),
    ),
    ModelSpec(
        "qwen2.5-0.5b-instruct", "Qwen2.5 0.5B Instruct", 0.49,
        _scores(55, 45, 44), 0.56, "qwen2.5", "0.5b",
        "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
        "Qwen2.5-0.5B-Instruct-{quant}.gguf", 32_768, "Qwen2", 24,
        _quants(0.40, 0.46, 0.53),
    ),
    ModelSpec(
        "qwen2.5-1.5b-instruct", "Qwen2.5 1.5B Instruct", 1.54,
        _scores(69, 59, 57), 0.70, "qwen2.5", "1.5b",
        "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        "Qwen2.5-1.5B-Instruct-{quant}.gguf", 32_768, "Qwen2", 28,
        _quants(0.99, 1.14, 1.65),
    ),
    ModelSpec(
        "qwen2.5-3b-instruct", "Qwen2.5 3B Instruct", 3.09,
        _scores(77, 69, 66), 0.78, "qwen2.5", "3b",
        "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "Qwen2.5-3B-Instruct-{quant}.gguf", 32_768, "Qwen2", 36,
        _quants(1.93, 2.22, 3.29),
    ),
    ModelSpec(
        "qwen2.5-7b-instruct", "Qwen2.5 7B Instruct", 7.62,
        _scores(85, 79, 75), 0.86, "qwen2.5", "7b",
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "Qwen2.5-7B-Instruct-{quant}.gguf", 32_768, "Qwen2", 28,
        _quants(4.68, 5.44, 8.10),
    ),
    ModelSpec(
        "qwen2.5-coder-0.5b-instruct", "Qwen2.5-Coder 0.5B Instruct", 0.49,
        _scores(50, 43, 55), 0.57, "qwen2.5-coder", "0.5b",
        "bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        "Qwen2.5-Coder-0.5B-Instruct-{quant}.gguf", 32_768, "Qwen2", 24,
        _quants(0.40, 0.46, 0.53),
    ),
    ModelSpec(
        "qwen2.5-coder-1.5b-instruct", "Qwen2.5-Coder 1.5B Instruct", 1.54,
        _scores(61, 55, 69), 0.70, "qwen2.5-coder", "1.5b",
        "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "Qwen2.5-Coder-1.5B-Instruct-{quant}.gguf", 32_768, "Qwen2", 28,
        _quants(0.99, 1.14, 1.65),
    ),
    ModelSpec(
        "qwen2.5-coder-3b-instruct", "Qwen2.5-Coder 3B Instruct", 3.09,
        _scores(67, 65, 78), 0.79, "qwen2.5-coder", "3b",
        "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF",
        "Qwen2.5-Coder-3B-Instruct-{quant}.gguf", 32_768, "Qwen2", 36,
        _quants(1.93, 2.22, 3.29),
    ),
    ModelSpec(
        "qwen2.5-coder-7b-instruct", "Qwen2.5-Coder 7B Instruct", 7.62,
        _scores(73, 76, 88), 0.88, "qwen2.5-coder", "7b",
        "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "Qwen2.5-Coder-7B-Instruct-{quant}.gguf", 32_768, "Qwen2", 28,
        _quants(4.68, 5.44, 8.10),
    ),
    ModelSpec(
        "gemma-3-1b-it", "Gemma 3 1B IT", 1.00,
        _scores(64, 53, 48), 0.65, "gemma3", "1b",
        "bartowski/google_gemma-3-1b-it-GGUF",
        "google_gemma-3-1b-it-{quant}.gguf", 32_768, "Gemma 3", 26,
        _quants(0.81, 0.94, 1.18),
    ),
    ModelSpec(
        "gemma-3-4b-it", "Gemma 3 4B IT", 4.30,
        _scores(82, 75, 68), 0.83, "gemma3", "4b",
        "bartowski/google_gemma-3-4b-it-GGUF",
        "google_gemma-3-4b-it-{quant}.gguf", 131_072, "Gemma 3", 34,
        _quants(2.49, 2.87, 4.33),
    ),
    ModelSpec(
        "gemma-3-12b-it", "Gemma 3 12B IT", 12.22,
        _scores(89, 84, 76), 0.90, "gemma3", "12b",
        "bartowski/google_gemma-3-12b-it-GGUF",
        "google_gemma-3-12b-it-{quant}.gguf", 131_072, "Gemma 3", 48,
        _quants(7.30, 8.45, 12.90),
    ),
    ModelSpec(
        "phi-4-mini-instruct", "Phi-4 Mini Instruct", 3.84,
        _scores(79, 78, 70), 0.81, "phi4-mini", "3.8b",
        "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "microsoft_Phi-4-mini-instruct-{quant}.gguf", 131_072, "Phi 3", 32,
        _quants(2.49, 2.87, 4.16),
    ),
    ModelSpec(
        "smollm2-1.7b-instruct", "SmolLM2 1.7B Instruct", 1.71,
        _scores(65, 53, 52), 0.66, "smollm2", "1.7b",
        "bartowski/SmolLM2-1.7B-Instruct-GGUF",
        "SmolLM2-1.7B-Instruct-{quant}.gguf", 8_192, "Llama", 24,
        _quants(1.06, 1.23, 1.82),
    ),
    ModelSpec(
        "deepseek-r1-distill-qwen-1.5b", "DeepSeek-R1 Distill Qwen 1.5B", 1.78,
        _scores(64, 72, 63), 0.72, "deepseek-r1", "1.5b",
        "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "DeepSeek-R1-Distill-Qwen-1.5B-{quant}.gguf", 131_072, "Qwen2", 28,
        _quants(1.12, 1.29, 1.89),
    ),
    ModelSpec(
        "deepseek-r1-distill-qwen-7b", "DeepSeek-R1 Distill Qwen 7B", 7.62,
        _scores(77, 88, 80), 0.87, "deepseek-r1", "7b",
        "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "DeepSeek-R1-Distill-Qwen-7B-{quant}.gguf", 131_072, "Qwen2", 28,
        _quants(4.68, 5.44, 8.10),
    ),
    ModelSpec(
        "mistral-7b-instruct-v0.3", "Mistral 7B Instruct v0.3", 7.25,
        _scores(80, 70, 68), 0.80, "mistral", "7b-instruct-v0.3",
        "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "Mistral-7B-Instruct-v0.3-{quant}.gguf", 32_768, "Mistral", 32,
        _quants(4.37, 5.13, 7.70),
    ),
    ModelSpec(
        "granite-3.3-2b-instruct", "Granite 3.3 2B Instruct", 2.50,
        _scores(70, 64, 67), 0.72, "granite3.3", "2b",
        "bartowski/ibm-granite_granite-3.3-2b-instruct-GGUF",
        "ibm-granite_granite-3.3-2b-instruct-{quant}.gguf", 131_072,
        "Granite", 40, _quants(1.55, 1.79, 2.67),
    ),
    ModelSpec(
        "granite-3.3-8b-instruct", "Granite 3.3 8B Instruct", 8.10,
        _scores(84, 78, 81), 0.85, "granite3.3", "8b",
        "bartowski/ibm-granite_granite-3.3-8b-instruct-GGUF",
        "ibm-granite_granite-3.3-8b-instruct-{quant}.gguf", 131_072,
        "Granite", 40, _quants(4.88, 5.68, 8.60),
    ),
)

MODEL_CATALOG = CATALOG
"""Compatibility alias with a more explicit name."""

_BY_ID = {model.id: model for model in CATALOG}
if len(_BY_ID) != len(CATALOG):  # pragma: no cover - import-time data guard
    raise RuntimeError("catalog contains duplicate model ids")


def get_model(model_id: str) -> ModelSpec:
    """Return a model by stable id, raising ``KeyError`` when absent."""

    try:
        return _BY_ID[model_id.casefold()]
    except KeyError:
        raise KeyError(f"unknown model id: {model_id!r}") from None


def get_quant(model_id: str, quantization: str) -> QuantSpec:
    """Return a quantization for a model."""

    return get_model(model_id).quant(quantization)


def find_models(
    *,
    max_parameters_b: float | None = None,
    max_size_gb: float | None = None,
    quantization: str = "Q4_K_M",
    task: str | None = None,
    min_score: int = 0,
    architecture: str | None = None,
) -> tuple[ModelSpec, ...]:
    """Filter models while preserving the catalog's deterministic order.

    ``max_size_gb`` applies to the selected ``quantization``. Models without
    that quantization are excluded. ``task`` names are case-insensitive.
    """

    if max_parameters_b is not None and max_parameters_b <= 0:
        raise ValueError("max_parameters_b must be positive")
    if max_size_gb is not None and max_size_gb <= 0:
        raise ValueError("max_size_gb must be positive")
    if not 0 <= min_score <= 100:
        raise ValueError("min_score must be between 0 and 100")

    task_key = task.casefold() if task is not None else None
    architecture_key = architecture.casefold() if architecture is not None else None
    matches: list[ModelSpec] = []
    for model in CATALOG:
        if max_parameters_b is not None and model.parameter_count_b > max_parameters_b:
            continue
        if architecture_key is not None and model.architecture.casefold() != architecture_key:
            continue
        if task_key is not None and model.task_scores.get(task_key, -1) < min_score:
            continue
        if max_size_gb is not None:
            try:
                size = model.quant(quantization).size_gb
            except KeyError:
                continue
            if size > max_size_gb:
                continue
        matches.append(model)
    return tuple(matches)


def best_models(
    task: str,
    *,
    limit: int = 5,
    candidates: Iterable[ModelSpec] | None = None,
) -> tuple[ModelSpec, ...]:
    """Rank models by task score, then quality, then stable id."""

    if limit < 0:
        raise ValueError("limit cannot be negative")
    task_key = task.casefold()
    pool = CATALOG if candidates is None else tuple(candidates)
    if not any(task_key in model.task_scores for model in pool):
        raise KeyError(f"unknown task: {task!r}")
    ranked = sorted(
        (model for model in pool if task_key in model.task_scores),
        key=lambda model: (-model.task_scores[task_key], -model.quality, model.id),
    )
    return tuple(ranked[:limit])


def list_model_ids() -> tuple[str, ...]:
    """Return all stable ids in catalog order."""

    return tuple(model.id for model in CATALOG)


__all__ = [
    "CATALOG",
    "MODEL_CATALOG",
    "ModelSpec",
    "QuantSpec",
    "best_models",
    "find_models",
    "get_model",
    "get_quant",
    "list_model_ids",
]
