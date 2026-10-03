"""Task classification and an accessible interactive task interview.

The parsing functions in this module are deliberately independent of the
interactive prompts.  Callers (and tests) can therefore build a
``TaskRequest`` without mocking terminal input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Mapping

from rich.console import Console
from rich.prompt import Prompt


class TaskType(str, Enum):
    """Tags used to describe the kind of work requested."""

    GENERAL = "general"
    CHAT = "chat"
    CODING = "coding"
    WRITING = "writing"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "question-answering"


class TaskPriority(str, Enum):
    """The user's preferred speed/quality trade-off."""

    SPEED = "speed"
    QUALITY = "quality"
    BALANCED = "balanced"


# Friendly labels are kept separate from enum values so a UI can present them
# without duplicating the canonical tags.
TASK_CHOICES: Mapping[str, TaskType] = {
    "general": TaskType.GENERAL,
    "chat": TaskType.CHAT,
    "coding": TaskType.CODING,
    "writing": TaskType.WRITING,
    "summarization": TaskType.SUMMARIZATION,
    "analysis": TaskType.ANALYSIS,
    "translation": TaskType.TRANSLATION,
    "question-answering": TaskType.QUESTION_ANSWERING,
}

PRIORITY_CHOICES: tuple[str, ...] = tuple(priority.value for priority in TaskPriority)
DEFAULT_TASK = TaskType.GENERAL
DEFAULT_PRIORITY = TaskPriority.BALANCED

# Ordered from more specific to more general.  Word-boundary matching avoids
# surprises such as classifying "encode" as coding because it contains "code".
TASK_KEYWORDS: Mapping[TaskType, tuple[str, ...]] = {
    TaskType.TRANSLATION: (
        "translate",
        "translation",
        "language",
        "localize",
        "localise",
    ),
    TaskType.SUMMARIZATION: (
        "summarize",
        "summarise",
        "summary",
        "condense",
        "shorten",
        "key points",
        "tl;dr",
    ),
    TaskType.CODING: (
        "code",
        "coding",
        "program",
        "function",
        "debug",
        "bug",
        "refactor",
        "script",
        "api",
        "test",
    ),
    TaskType.ANALYSIS: (
        "analyze",
        "analyse",
        "analysis",
        "compare",
        "evaluate",
        "investigate",
        "explain why",
        "reason",
    ),
    TaskType.WRITING: (
        "write",
        "rewrite",
        "draft",
        "edit",
        "proofread",
        "email",
        "article",
        "story",
    ),
    TaskType.QUESTION_ANSWERING: (
        "answer",
        "question",
        "what",
        "when",
        "where",
        "who",
        "how",
        "why",
    ),
    TaskType.CHAT: ("chat", "talk", "conversation", "brainstorm"),
}


@dataclass(frozen=True, slots=True)
class TaskRequest:
    """A normalized request ready for model selection or dispatch."""

    description: str
    task: TaskType = DEFAULT_TASK
    priority: TaskPriority = DEFAULT_PRIORITY

    @property
    def task_tag(self) -> str:
        """Return the canonical string tag expected by model routers."""

        return self.task.value


def _contains_keyword(text: str, keyword: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text) is not None


def infer_task_type(description: str | None) -> TaskType:
    """Infer a task tag from a plain-language description.

    Empty or unrecognized descriptions intentionally fall back to ``general``.
    """

    normalized = (description or "").strip().casefold()
    if not normalized:
        return DEFAULT_TASK

    for task_type, keywords in TASK_KEYWORDS.items():
        if any(_contains_keyword(normalized, keyword) for keyword in keywords):
            return task_type
    return DEFAULT_TASK


def _parse_task_type(value: TaskType | str | None, description: str) -> TaskType:
    if value is None:
        return infer_task_type(description)
    if isinstance(value, TaskType):
        return value

    normalized = value.strip().casefold().replace("_", "-").replace(" ", "-")
    aliases = {
        "auto": infer_task_type(description),
        "qa": TaskType.QUESTION_ANSWERING,
        "q&a": TaskType.QUESTION_ANSWERING,
        "question": TaskType.QUESTION_ANSWERING,
        "summarize": TaskType.SUMMARIZATION,
        "summarise": TaskType.SUMMARIZATION,
        "summary": TaskType.SUMMARIZATION,
        "code": TaskType.CODING,
        "write": TaskType.WRITING,
        "analyze": TaskType.ANALYSIS,
        "analyse": TaskType.ANALYSIS,
        "translate": TaskType.TRANSLATION,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return TASK_CHOICES[normalized]
    except KeyError as exc:
        choices = ", ".join(TASK_CHOICES)
        raise ValueError(f"Unknown task type {value!r}; choose one of: {choices}") from exc


def _parse_priority(value: TaskPriority | str | None) -> TaskPriority:
    if value is None:
        return DEFAULT_PRIORITY
    if isinstance(value, TaskPriority):
        return value
    try:
        return TaskPriority(value.strip().casefold())
    except ValueError as exc:
        choices = ", ".join(PRIORITY_CHOICES)
        raise ValueError(f"Unknown priority {value!r}; choose one of: {choices}") from exc


def parse_task_request(
    description: str | None = None,
    task: TaskType | str | None = None,
    priority: TaskPriority | str | None = None,
) -> TaskRequest:
    """Parse a task request without reading input, suitable for tests and CLIs."""

    clean_description = (description or "").strip()
    return TaskRequest(
        description=clean_description,
        task=_parse_task_type(task, clean_description),
        priority=_parse_priority(priority),
    )


def interview_task(console: Console | None = None) -> TaskRequest:
    """Conduct an accessible terminal interview using Rich prompts.

    Prompts use named choices rather than color or cursor position as the only
    signal, making them usable with screen readers and plain terminals.
    """

    console = console or Console()
    description = Prompt.ask(
        "Describe what you would like the model to do (blank is okay)",
        default="",
        show_default=False,
        console=console,
    ).strip()

    inferred = infer_task_type(description)
    task_choices = ["auto", *TASK_CHOICES.keys()]
    task = Prompt.ask(
        f"Task type; 'auto' detected [bold]{inferred.value}[/bold]",
        choices=task_choices,
        default="auto",
        show_choices=True,
        console=console,
    )
    priority = Prompt.ask(
        "Priority",
        choices=list(PRIORITY_CHOICES),
        default=DEFAULT_PRIORITY.value,
        show_choices=True,
        console=console,
    )

    return parse_task_request(description, task, priority)


# Short aliases retained for callers that use "tag" or "interactive" language.
TaskTag = TaskType
infer_task_tag = infer_task_type
prompt_for_task = interview_task


__all__ = [
    "DEFAULT_PRIORITY",
    "DEFAULT_TASK",
    "PRIORITY_CHOICES",
    "TASK_CHOICES",
    "TASK_KEYWORDS",
    "TaskPriority",
    "TaskRequest",
    "TaskTag",
    "TaskType",
    "infer_task_tag",
    "infer_task_type",
    "interview_task",
    "parse_task_request",
    "prompt_for_task",
]
