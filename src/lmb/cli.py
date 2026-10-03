"""Local Model Buddy command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .backends import BackendError, BackendManager, LaunchConfig
from .hardware import HardwareProfile, detect_hardware
from .profile import SavedProfile, load_profile, save_profile
from .recommend import GIB, Recommendation, recommend_models
from .smoketest import test_with_fallbacks
from .tasks import TaskPriority, TaskType, interview_task, parse_task_request

app = typer.Typer(
    name="lmb",
    help="Find and configure a local AI model that fits your PC.",
    no_args_is_help=False,
)
console = Console()


def _bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / GIB:.1f} GiB"


def _show_hardware(profile: HardwareProfile) -> None:
    table = Table(title="Your PC")
    table.add_column("Part", style="bold")
    table.add_column("Detected")
    table.add_row("CPU", f"{profile.cpu_model} ({profile.physical_cores} cores)")
    table.add_row("RAM", f"{_bytes(profile.total_ram)} total, {_bytes(profile.available_ram)} available")
    table.add_row("Disk", f"{_bytes(profile.disk_free)} free")
    flags = {flag.lower() for flag in profile.cpu_flags}
    acceleration = ", ".join(flag.upper() for flag in ("avx2", "avx512f") if flag in flags)
    table.add_row("CPU features", acceleration or "standard instruction set")
    if profile.gpus:
        for index, gpu in enumerate(profile.gpus, 1):
            table.add_row(
                f"GPU {index}", f"{gpu.name} ({_bytes(gpu.memory_total)} VRAM)"
            )
    else:
        table.add_row("GPU", "No supported discrete GPU detected — CPU mode is okay")
    console.print(table)


def _show_recommendations(items: list[Recommendation]) -> None:
    if not items:
        console.print(
            Panel(
                "No safe model fit was found with current free RAM and disk space. "
                "Close a few apps or free disk space, then run [bold]lmb recommend[/bold] again.",
                title="A little more room is needed",
            )
        )
        return
    table = Table(title="Models that fit safely")
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Quant")
    table.add_column("Est. RAM", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("Why")
    for number, item in enumerate(items, 1):
        table.add_row(
            str(number),
            item.model.name,
            item.quant.name,
            _bytes(item.estimated_ram_bytes),
            f"{item.settings.context_size:,}",
            item.reason,
        )
    console.print(table)


def _profile_from_config(
    recommendation: Recommendation,
    backend_name: str,
    config: LaunchConfig,
    speed: float,
) -> SavedProfile:
    return SavedProfile(
        model_key=recommendation.model.id,
        model_id=config.model_id,
        quantization=config.quantization,
        backend=backend_name,
        context_size=config.context_size,
        threads=config.threads,
        gpu_layers=config.gpu_layers,
        batch_size=config.batch_size,
        mmap=config.mmap,
        kv_cache_quant=config.kv_cache_quant,
        model_path=str(config.model_path) if config.model_path else None,
        measured_tokens_per_second=speed,
    )


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Launch the friendly guided setup when no command is supplied."""
    if ctx.invoked_subcommand is not None:
        return
    console.print(
        Panel(
            "You do not need a gaming PC to use local AI. I’ll inspect this "
            "computer, find a model that fits, and verify it before saving anything.",
            title="Local Model Buddy",
        )
    )
    profile = detect_hardware()
    _show_hardware(profile)
    request = interview_task(console)
    recommendations = recommend_models(profile, request)
    _show_recommendations(recommendations)
    if not recommendations:
        raise typer.Exit(1)

    choice = typer.prompt(
        "Choose a model",
        type=int,
        default=1,
    )
    if not 1 <= choice <= len(recommendations):
        console.print(f"Choose a number from 1 to {len(recommendations)}.")
        raise typer.Exit(2)
    selected = recommendations[choice - 1]
    manager = BackendManager()
    backend = manager.select(profile)
    status = backend.status()
    console.print(f"\nBest runtime for this PC: [bold]{backend.name}[/bold] ({status.detail})")
    if not status.installed:
        console.print(Panel(backend.install_instructions(), title="One-time runtime setup"))
        raise typer.Exit(2)
    if not typer.confirm(
        f"Download {selected.model.name} ({selected.quant.size_gb:.2f} GB) and test it?",
        default=True,
    ):
        console.print("No download was started.")
        return

    fallbacks = [selected] + [
        item for item in recommendations if item.model.id != selected.model.id
    ]
    configs = [item.launch_config(backend.name) for item in fallbacks]
    ready, result, attempts = test_with_fallbacks(backend, configs, yes=True)
    if ready is None:
        console.print(Panel("\n".join(attempts), title="Setup could not finish"))
        raise typer.Exit(1)
    matched = next(
        item
        for item in fallbacks
        if item.launch_config(backend.name).model_id == ready.model_id
    )
    destination = save_profile(_profile_from_config(matched, backend.name, ready, result.tokens_per_second))
    console.print(
        Panel(
            f"[bold green]It works.[/bold green] Approx. {result.tokens_per_second:.1f} "
            f"tokens/second.\nSaved setup: {destination}\nRun it again with [bold]lmb run[/bold].",
            title="Ready",
        )
    )


@app.command("scan")
def scan_command() -> None:
    """Inspect this PC without downloading or changing anything."""
    _show_hardware(detect_hardware())


@app.command("recommend")
def recommend_command(
    description: str = typer.Option("", "--task", "-t", help="What the model should do."),
    task_type: Optional[TaskType] = typer.Option(None, "--type"),
    priority: TaskPriority = typer.Option(TaskPriority.BALANCED, "--priority"),
    limit: int = typer.Option(5, min=1, max=20),
) -> None:
    """Rank models that fit. This command does not download anything."""
    profile = detect_hardware()
    request = parse_task_request(description, task_type, priority)
    _show_hardware(profile)
    _show_recommendations(recommend_models(profile, request, limit=limit))


@app.command("doctor")
def doctor_command() -> None:
    """Check runtimes and the last known-good setup."""
    profile = detect_hardware()
    _show_hardware(profile)
    table = Table(title="Runtime health")
    table.add_column("Runtime")
    table.add_column("Installed")
    table.add_column("Ready")
    table.add_column("Details")
    for status in BackendManager().statuses():
        table.add_row(
            status.name,
            "yes" if status.installed else "no",
            "yes" if status.available else "no",
            status.detail,
        )
    console.print(table)
    saved = load_profile()
    if saved:
        console.print(f"Saved model: [bold]{saved.model_id}[/bold] via {saved.backend}")
    else:
        console.print("No saved setup yet. Run [bold]lmb[/bold] to create one.")


@app.command("run")
def run_command(prompt: str = typer.Argument(..., help="Prompt to send to the model.")) -> None:
    """Run the saved model setup."""
    saved = load_profile()
    if not saved:
        console.print("No saved setup yet. Run [bold]lmb[/bold] first.")
        raise typer.Exit(2)
    config = LaunchConfig(
        model_id=saved.model_id,
        quantization=saved.quantization,
        context_size=saved.context_size,
        threads=saved.threads,
        gpu_layers=saved.gpu_layers,
        batch_size=saved.batch_size,
        mmap=saved.mmap,
        kv_cache_quant=saved.kv_cache_quant,
        model_path=Path(saved.model_path) if saved.model_path else None,
    )
    try:
        response = BackendManager().get(saved.backend).generate(
            config, prompt, max_tokens=256
        )
    except BackendError as exc:
        console.print(f"[red]Could not run the model:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(response)
