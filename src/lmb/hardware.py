"""Cross-platform hardware discovery.

All capacities exposed by this module are measured in bytes.  Detection is
best-effort: optional operating-system tools may be absent, time out, or
return a format from a different driver version without preventing the rest
of the profile from being collected.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import psutil

_COMMAND_TIMEOUT = 3.0
_MIB = 1024 * 1024


@dataclass(frozen=True)
class GPUInfo:
    """Information about one graphics processor."""

    name: str
    vendor: str
    memory_total: Optional[int] = None
    memory_used: Optional[int] = None
    driver_version: Optional[str] = None
    index: Optional[int] = None
    uuid: Optional[str] = None


@dataclass(frozen=True)
class HardwareProfile:
    """A point-in-time snapshot of the host's relevant hardware."""

    cpu_model: str
    physical_cores: int
    logical_cores: int
    cpu_flags: list[str] = field(default_factory=list)
    total_ram: int = 0
    available_ram: int = 0
    swap_total: int = 0
    swap_free: int = 0
    disk_free: int = 0
    os: str = ""
    gpus: list[GPUInfo] = field(default_factory=list)


def _run(command: Sequence[str], timeout: float = _COMMAND_TIMEOUT) -> Optional[str]:
    """Run an optional system utility, returning output only on success."""
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def _linux_cpuinfo() -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value:
            values.setdefault(key.strip().lower(), []).append(value)
    return values


def _sysctl(name: str) -> Optional[str]:
    return _run(("sysctl", "-n", name))


def _cpu_model_and_flags() -> tuple[str, list[str]]:
    model = ""
    flags: set[str] = set()
    system = platform.system()

    if system == "Linux":
        info = _linux_cpuinfo()
        for key in ("model name", "hardware", "processor", "cpu model"):
            if info.get(key):
                model = info[key][0]
                break
        for key in ("flags", "features"):
            for value in info.get(key, []):
                flags.update(value.lower().split())
    elif system == "Darwin":
        model = _sysctl("machdep.cpu.brand_string") or ""
        for key in (
            "machdep.cpu.features",
            "machdep.cpu.leaf7_features",
            "machdep.cpu.extfeatures",
        ):
            value = _sysctl(key)
            if value:
                flags.update(value.lower().split())
    elif system == "Windows":
        model = platform.processor()
        # PowerShell is more widely available than WMIC on current Windows.
        if not model:
            output = _run(
                (
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 "
                    "-ExpandProperty Name)",
                )
            )
            model = output or ""
        identifier = os.environ.get("PROCESSOR_IDENTIFIER", "")
        # Windows does not expose a portable complete CPUID flag list.  Retain
        # architecture capabilities that can be established without native code.
        for flag in ("MMX", "SSE", "SSE2", "AVX", "AVX2"):
            if re.search(rf"\b{flag}\b", identifier, re.IGNORECASE):
                flags.add(flag.lower())

    model = " ".join((model or platform.processor() or platform.machine()).split())
    return model or "Unknown", sorted(flags)


def _parse_mib(value: Any) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not match:
        return None
    try:
        amount = float(match.group())
    except ValueError:
        return None
    return max(0, int(amount * _MIB))


def _nvidia_gpus() -> list[GPUInfo]:
    fields = "index,name,memory.total,memory.used,driver_version,uuid"
    output = _run(
        (
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        )
    )
    if not output:
        return []

    gpus: list[GPUInfo] = []
    for line in output.splitlines():
        # NVIDIA GPU names do not contain commas; maxsplit protects UUID-like
        # future fields if the utility adds unexpected trailing data.
        parts = [part.strip() for part in line.split(",", 5)]
        if len(parts) < 6:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            index = None
        gpus.append(
            GPUInfo(
                name=parts[1] or "Unknown NVIDIA GPU",
                vendor="NVIDIA",
                memory_total=_parse_mib(parts[2]),
                memory_used=_parse_mib(parts[3]),
                driver_version=parts[4] if parts[4] not in ("", "N/A") else None,
                index=index,
                uuid=parts[5] if parts[5] not in ("", "N/A") else None,
            )
        )
    return gpus


def _find_mapping_value(mapping: dict[str, Any], terms: Iterable[str]) -> Any:
    wanted = tuple(term.lower() for term in terms)
    for key, value in mapping.items():
        normalized = re.sub(r"[^a-z0-9]+", " ", str(key).lower())
        if all(term in normalized for term in wanted):
            return value
    return None


def _rocm_gpus() -> list[GPUInfo]:
    output = _run(
        (
            "rocm-smi",
            "--showproductname",
            "--showmeminfo",
            "vram",
            "--showdriverversion",
            "--json",
        )
    )
    if not output:
        return []
    try:
        payload = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []

    gpus: list[GPUInfo] = []
    for ordinal, (card, raw) in enumerate(payload.items()):
        if not isinstance(raw, dict):
            continue
        name = (
            _find_mapping_value(raw, ("card", "series"))
            or _find_mapping_value(raw, ("product", "name"))
            or _find_mapping_value(raw, ("device", "name"))
            or str(card)
        )
        total = _find_mapping_value(raw, ("vram", "total"))
        used = _find_mapping_value(raw, ("vram", "used"))
        driver = _find_mapping_value(raw, ("driver", "version"))
        index_match = re.search(r"\d+", str(card))
        index = int(index_match.group()) if index_match else ordinal

        # rocm-smi JSON normally reports memory in bytes despite labels varying
        # between releases.  Explicit MiB/GiB strings are handled separately.
        def rocm_bytes(value: Any) -> Optional[int]:
            if value is None:
                return None
            text = str(value)
            parsed = _parse_mib(text)
            if parsed is None:
                return None
            if re.search(r"\bGiB\b", text, re.IGNORECASE):
                return parsed * 1024
            if re.search(r"\bMiB\b|\bMB\b", text, re.IGNORECASE):
                return parsed
            number = re.search(r"\d+", text.replace(",", ""))
            return int(number.group()) if number else None

        gpus.append(
            GPUInfo(
                name=str(name).strip() or "Unknown AMD GPU",
                vendor="AMD",
                memory_total=rocm_bytes(total),
                memory_used=rocm_bytes(used),
                driver_version=str(driver).strip() if driver else None,
                index=index,
            )
        )
    return gpus


def _lspci_gpus(existing_vendors: set[str]) -> list[GPUInfo]:
    output = _run(("lspci", "-mm"))
    if not output:
        return []

    gpus: list[GPUInfo] = []
    for line in output.splitlines():
        if not re.search(r"(VGA compatible|3D controller|Display controller)", line, re.I):
            continue
        quoted = re.findall(r'"([^"]*)"', line)
        description = " ".join(quoted[1:]).strip() if len(quoted) > 1 else line.strip()
        lowered = line.lower()
        if "nvidia" in lowered:
            vendor = "NVIDIA"
        elif "amd" in lowered or "advanced micro devices" in lowered or "ati " in lowered:
            vendor = "AMD"
        elif "intel" in lowered:
            vendor = "Intel"
        else:
            vendor = quoted[1].strip() if len(quoted) > 1 else "Unknown"
        if vendor.casefold() in existing_vendors:
            continue
        gpus.append(GPUInfo(name=description or "Unknown GPU", vendor=vendor))
    return gpus


def detect_gpus() -> list[GPUInfo]:
    """Detect GPUs using vendor tools, then fill gaps from PCI data."""
    gpus = _nvidia_gpus()
    gpus.extend(_rocm_gpus())
    vendors = {gpu.vendor.casefold() for gpu in gpus}
    gpus.extend(_lspci_gpus(vendors))
    return gpus


def detect_hardware(disk_path: Optional[os.PathLike[str] | str] = None) -> HardwareProfile:
    """Collect a hardware profile without failing on optional-tool errors.

    ``disk_path`` selects the filesystem whose free space is reported.  By
    default this is the system drive on Windows and the root filesystem on
    POSIX hosts.
    """
    cpu_model, cpu_flags = _cpu_model_and_flags()
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    if disk_path is None:
        disk_path = Path(os.environ.get("SystemDrive", "C:") + os.sep) if os.name == "nt" else Path("/")
    try:
        disk_free = psutil.disk_usage(str(disk_path)).free
    except (OSError, ValueError):
        disk_free = 0

    os_description = platform.platform(aliased=True, terse=False) or platform.system()
    return HardwareProfile(
        cpu_model=cpu_model,
        physical_cores=int(physical),
        logical_cores=int(logical),
        cpu_flags=cpu_flags,
        total_ram=int(memory.total),
        available_ram=int(memory.available),
        swap_total=int(swap.total),
        swap_free=int(swap.free),
        disk_free=int(disk_free),
        os=os_description,
        gpus=detect_gpus(),
    )


def get_hardware_profile(
    disk_path: Optional[os.PathLike[str] | str] = None,
) -> HardwareProfile:
    """Backward-friendly alias for :func:`detect_hardware`."""
    return detect_hardware(disk_path)


__all__ = [
    "GPUInfo",
    "HardwareProfile",
    "detect_gpus",
    "detect_hardware",
    "get_hardware_profile",
]
