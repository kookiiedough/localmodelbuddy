from lmb.hardware import GPUInfo, HardwareProfile
from lmb.catalog import CATALOG
from lmb.recommend import GIB, estimate_memory, memory_budget, recommend_models
from lmb.tasks import TaskPriority, TaskType, parse_task_request


def profile(ram_gb: int, available_gb: float, vram_gb: int = 0) -> HardwareProfile:
    gpus = (
        [GPUInfo("Test GPU", "NVIDIA", memory_total=vram_gb * GIB)]
        if vram_gb
        else []
    )
    return HardwareProfile(
        cpu_model="Test CPU",
        physical_cores=4,
        logical_cores=8,
        cpu_flags=["avx2"],
        total_ram=ram_gb * GIB,
        available_ram=int(available_gb * GIB),
        disk_free=100 * GIB,
        os="Test OS",
        gpus=gpus,
    )


def test_two_gb_machine_gets_a_safe_small_model() -> None:
    hardware = profile(2, 1.5)
    results = recommend_models(
        hardware, parse_task_request("chat with me", priority="speed")
    )
    assert results
    assert results[0].model.parameter_count_b <= 1.5
    assert all(item.estimated_ram_bytes <= item.memory_budget_bytes for item in results)


def test_eight_gb_coding_request_prefers_coding_model() -> None:
    results = recommend_models(
        profile(8, 6),
        parse_task_request("write and debug Python code", priority="balanced"),
    )
    assert results
    assert "coder" in results[0].model.id
    assert results[0].settings.context_size == 4096


def test_gpu_offload_is_enabled_when_vram_is_available() -> None:
    results = recommend_models(
        profile(16, 12, 8),
        parse_task_request("analyze a difficult question", priority="quality"),
    )
    assert results
    assert results[0].settings.gpu_layers > 0
    assert results[0].settings.context_size == 8192


def test_memory_estimate_grows_with_context() -> None:
    result = recommend_models(
        profile(8, 6), parse_task_request("", task=TaskType.GENERAL)
    )[0]
    low = estimate_memory(result.model, result.quant, 2048)
    high = estimate_memory(result.model, result.quant, 8192)
    assert high > low


def test_memory_budget_reserves_space_for_the_os() -> None:
    hardware = profile(8, 7.5)
    assert memory_budget(hardware) <= int(hardware.total_ram * 0.8)


def test_every_catalog_model_has_a_vllm_upstream_id() -> None:
    results = recommend_models(
        profile(64, 60, 24),
        parse_task_request("chat", priority="quality"),
        limit=len(CATALOG),
    )
    assert len(results) == len(CATALOG)
    assert all("/" in item.launch_config("vllm").model_id for item in results)
