from pathlib import Path

from lmb.backends.base import Backend, BackendError, BackendStatus, LaunchConfig
from lmb.profile import SavedProfile, load_profile, save_profile
from lmb.smoketest import run_smoke_test, test_with_fallbacks as run_with_fallbacks
from lmb.tasks import TaskPriority, TaskType, infer_task_type, parse_task_request


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()

    def status(self) -> BackendStatus:
        return BackendStatus(self.name, True, True)

    def install_instructions(self) -> str:
        return ""

    def prepare(self, config: LaunchConfig, *, yes: bool = False) -> LaunchConfig:
        return config

    def generate(
        self, config: LaunchConfig, prompt: str, *, max_tokens: int = 64
    ) -> str:
        if config.model_id in self.failures:
            raise BackendError("simulated OOM")
        return "Ready and running smoothly."


def config(model: str) -> LaunchConfig:
    return LaunchConfig(model, "Q4_K_M", 2048, 4)


def test_plain_language_task_inference() -> None:
    assert infer_task_type("Please debug this Python function") == TaskType.CODING
    assert infer_task_type("Translate this note into French") == TaskType.TRANSLATION
    assert parse_task_request("").priority == TaskPriority.BALANCED


def test_smoke_test_passes_with_text() -> None:
    result = run_smoke_test(FakeBackend(), config("small"))
    assert result.passed
    assert result.tokens_per_second > 0


def test_fallback_uses_next_configuration() -> None:
    ready, result, attempts = run_with_fallbacks(
        FakeBackend({"large"}), [config("large"), config("small")], yes=True
    )
    assert ready is not None and ready.model_id == "small"
    assert result.passed
    assert len(attempts) == 1


def test_profile_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    original = SavedProfile(
        "small", "small:latest", "Q4_K_M", "ollama", 2048, 4, 0, 64, True, "q8_0"
    )
    save_profile(original, path)
    assert load_profile(path) == original
