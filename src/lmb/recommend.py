"""Hardware-aware model ranking and runtime tuning."""

from __future__ import annotations

from dataclasses import dataclass

from .backends.base import LaunchConfig
from .catalog import CATALOG, ModelSpec, QuantSpec
from .hardware import HardwareProfile
from .tasks import TaskPriority, TaskRequest

GIB = 1024**3

TASK_SCORE_MAP = {
    "general": "chat",
    "chat": "chat",
    "coding": "coding",
    "writing": "chat",
    "summarization": "chat",
    "analysis": "reasoning",
    "translation": "chat",
    "question-answering": "reasoning",
}


def _upstream_model_id(model: ModelSpec) -> str:
    """Translate curated GGUF entries to their original vLLM repositories."""
    model_id = model.id
    if model_id.startswith("llama-3.2-"):
        size = model_id.split("-")[2].upper()
        return f"meta-llama/Llama-3.2-{size}-Instruct"
    if model_id.startswith("qwen2.5-coder-"):
        size = model_id.split("-")[2].upper()
        return f"Qwen/Qwen2.5-Coder-{size}-Instruct"
    if model_id.startswith("qwen2.5-"):
        size = model_id.split("-")[1].upper()
        return f"Qwen/Qwen2.5-{size}-Instruct"
    if model_id.startswith("gemma-3-"):
        size = model_id.split("-")[2]
        return f"google/gemma-3-{size}-it"
    if model_id == "phi-4-mini-instruct":
        return "microsoft/Phi-4-mini-instruct"
    if model_id == "smollm2-1.7b-instruct":
        return "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    if model_id.startswith("deepseek-r1-distill-qwen-"):
        size = model_id.rsplit("-", 1)[-1].upper()
        return f"deepseek-ai/DeepSeek-R1-Distill-Qwen-{size}"
    if model_id == "mistral-7b-instruct-v0.3":
        return "mistralai/Mistral-7B-Instruct-v0.3"
    if model_id.startswith("granite-3.3-"):
        size = model_id.split("-")[2]
        return f"ibm-granite/granite-3.3-{size}-instruct"
    if model_id == "tinyllama-1.1b-chat":
        return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    raise ValueError(f"No upstream repository is known for {model_id}.")


@dataclass(frozen=True)
class RuntimeSettings:
    context_size: int
    threads: int
    gpu_layers: int
    batch_size: int
    mmap: bool
    kv_cache_quant: str


@dataclass(frozen=True)
class Recommendation:
    model: ModelSpec
    quant: QuantSpec
    settings: RuntimeSettings
    estimated_ram_bytes: int
    memory_budget_bytes: int
    score: float
    reason: str

    @property
    def headroom_bytes(self) -> int:
        return self.memory_budget_bytes - self.estimated_ram_bytes

    def launch_config(self, backend: str) -> LaunchConfig:
        if backend == "ollama":
            model_id = self.model.ollama_ref
        elif backend == "vllm":
            model_id = _upstream_model_id(self.model)
        else:
            model_id = self.model.id
        filename = self.model.hf_filename_pattern.replace("{quant}", self.quant.name)
        return LaunchConfig(
            model_id=model_id,
            quantization=self.quant.name,
            context_size=self.settings.context_size,
            threads=self.settings.threads,
            gpu_layers=self.settings.gpu_layers,
            batch_size=self.settings.batch_size,
            mmap=self.settings.mmap,
            kv_cache_quant=self.settings.kv_cache_quant,
            extra={"hf_repo": self.model.hf_repo, "hf_file": filename},
        )


def memory_budget(profile: HardwareProfile) -> int:
    """Memory usable without pushing the desktop into swap thrashing."""
    total = profile.total_ram
    available = profile.available_ram
    reclaimable_floor = int(total * 0.55)
    return max(256 * 1024**2, min(int(total * 0.80), max(int(available * 0.85), reclaimable_floor)))


def choose_context(profile: HardwareProfile, model: ModelSpec) -> int:
    ram_gb = profile.total_ram / GIB
    requested = 2048 if ram_gb < 4 else 4096 if ram_gb < 12 else 8192
    return min(requested, model.context_length)


def estimate_memory(model: ModelSpec, quant: QuantSpec, context_size: int) -> int:
    """Conservative weights + KV cache + runtime workspace estimate."""
    weights = quant.size_gb * 1_000_000_000
    # KV cache varies by architecture/GQA. This approximation is intentionally
    # conservative for model selection, then the smoke test verifies reality.
    kv_cache = model.parameter_count_b * 22_000_000 * (context_size / 2048)
    runtime = 220 * 1024**2 + weights * 0.08
    return int(weights + kv_cache + runtime)


def _max_vram(profile: HardwareProfile) -> int:
    return max((gpu.memory_total or 0 for gpu in profile.gpus), default=0)


def _settings(profile: HardwareProfile, model: ModelSpec, quant: QuantSpec) -> RuntimeSettings:
    context = choose_context(profile, model)
    vram = _max_vram(profile)
    estimated = estimate_memory(model, quant, context)
    usable_vram = max(0, vram - 512 * 1024**2)
    gpu_layers = min(model.layers, int(model.layers * min(1.0, usable_vram / max(estimated, 1))))
    threads = max(1, min(profile.physical_cores or profile.logical_cores, 12))
    ram_gb = profile.total_ram / GIB
    return RuntimeSettings(
        context_size=context,
        threads=threads,
        gpu_layers=gpu_layers,
        batch_size=64 if ram_gb < 4 else 128 if ram_gb < 12 else 256,
        mmap=True,
        kv_cache_quant="q4_0" if ram_gb < 4 else "q8_0",
    )


def recommend_models(
    profile: HardwareProfile,
    request: TaskRequest,
    *,
    limit: int = 5,
) -> list[Recommendation]:
    budget = memory_budget(profile)
    task = TASK_SCORE_MAP.get(request.task.value, "chat")
    candidates: list[Recommendation] = []
    for model in CATALOG:
        for quant in model.quantizations:
            settings = _settings(profile, model, quant)
            usage = estimate_memory(model, quant, settings.context_size)
            if usage > budget or quant.size_gb * 1_000_000_000 > profile.disk_free:
                continue
            task_score = model.task_scores.get(task, model.task_scores.get("chat", 0))
            speed_bias = -model.parameter_count_b * 2.8
            quality_bias = model.quality * 28 + quant.quality * 12
            if request.priority == TaskPriority.SPEED:
                score = task_score * 0.55 + speed_bias * 2 + quant.bits * -0.6
            elif request.priority == TaskPriority.QUALITY:
                score = task_score * 0.75 + quality_bias * 1.3
            else:
                score = task_score * 0.68 + quality_bias + speed_bias
            reason = (
                f"{model.name} is strong for {request.task.value} and should use "
                f"about {usage / GIB:.1f} GiB of the {budget / GIB:.1f} GiB safe budget."
            )
            candidates.append(
                Recommendation(model, quant, settings, usage, budget, score, reason)
            )
    candidates.sort(
        key=lambda item: (item.score, item.quant.quality, item.model.quality),
        reverse=True,
    )
    # Keep one quant per model so fallbacks are meaningfully different.
    unique: list[Recommendation] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.model.id in seen:
            continue
        seen.add(candidate.model.id)
        unique.append(candidate)
        if len(unique) >= limit:
            break
    return unique
