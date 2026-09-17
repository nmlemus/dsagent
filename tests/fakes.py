"""Shared helpers for the fake agents.

Every fake satisfies a step by reading the `produces` block out of its prompt and
writing what it names. Since `produces` entries may be globs, "what it names" is
no longer a filename, and one definition of the expansion beats three.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from dsagent.runner import is_pattern

PRODUCES_PREFIX = "- `"


def produces_of(prompt: str) -> list[str]:
    """The `produces` entries a step prompt declares, patterns included.

    The runner annotates a pattern as ``- `a/*.png` (one or more matching
    files)``, so the entry is what sits between the first pair of backticks.
    """
    out = []
    for line in prompt.splitlines():
        if line.startswith(PRODUCES_PREFIX) and "`" in line[len(PRODUCES_PREFIX):]:
            out.append(line[len(PRODUCES_PREFIX):].split("`")[0])
    return out


def paths_for(entry: str, count: int = 2) -> list[str]:
    """Concrete workspace-relative paths that satisfy one `produces` entry.

    A literal is itself. A pattern becomes `count` files that match it — the
    point of a pattern is that the step decides how many, so a fake has to
    decide too. `count=0` is how a test makes a pattern go unsatisfied.
    """
    if not is_pattern(entry):
        return [entry]
    stem, _, suffix = entry.rpartition("*")
    return [f"{stem}{i:02d}{suffix}" for i in range(1, count + 1)]


def expand(prompt: str, count: int = 2) -> list[str]:
    """Every path a fake should write to satisfy a step prompt."""
    return [p for entry in produces_of(prompt) for p in paths_for(entry, count)]


class ScriptedChatModel(BaseChatModel):
    """A chat model that replays canned messages, streaming path included.

    `GenericFakeChatModel` implements `_stream` by chunking message *content*,
    so a tool-call message with empty content yields nothing and the async path
    dies with "No generations found in stream" — which is exactly the path the
    AG-UI bridge takes. Implementing only `_generate` leaves `astream` to fall
    back to it, so the same script works sync and async.
    """

    # Pydantic fields on a LangChain model, so declared rather than assigned in
    # `__init__`; `default_factory` keeps the list from being shared.
    replies: list[Any] = Field(default_factory=list)
    cursor: int = 0

    def __init__(self, replies: list[Any], **kwargs: Any) -> None:
        super().__init__(replies=list(replies), **kwargs)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        i = self.cursor
        self.cursor = i + 1
        reply = self.replies[i] if i < len(self.replies) else AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def tiny_cartridge(root, steps: list[dict[str, Any]], *, name: str = "tiny"):
    """A one-persona cartridge on disk, loaded — a fixture for harness behaviour.

    A harness test that needs a particular `produces` shape used to reach for
    whichever step of `eda-to-report` happened to have it, which made a test of
    the *runner* fail when the *cartridge* changed its mind about writing PNGs.
    Declaring the shape the test is about is both clearer and stable.

    `steps` are step dicts as `workflow.yaml` writes them, minus `instructions`,
    which is generated.
    """
    from dsagent.cartridge import load_cartridge

    root = Path(root)
    (root / "agents").mkdir(parents=True, exist_ok=True)
    (root / "skills" / "only").mkdir(parents=True, exist_ok=True)
    wdir = root / "workflows" / "w"
    (wdir / "steps").mkdir(parents=True, exist_ok=True)

    (root / "agents" / "ana.md").write_text(
        "---\nname: ana\ndescription: does the work\nskills: [only]\n---\nYou are Ana.\n"
    )
    (root / "skills" / "only" / "SKILL.md").write_text(
        "---\nname: only\ndescription: the only skill\n---\n# Only\n"
    )
    (root / "cartridge.yaml").write_text(yaml.safe_dump({
        "name": name, "version": "0.0.1", "description": "a cartridge for one test",
        "personas": {"ana": {"role": "worker", "workflows": ["w"]}},
        "skills": {"only": {"scope": "all"}},
        "workflows": ["workflows/w"],
        "envs": {"default": {"kind": "kernel", "requirements": []}},
    }, sort_keys=False))

    declared = []
    for i, step in enumerate(steps):
        rel = f"steps/{i:02d}-{step['id']}.md"
        (wdir / rel).write_text(f"Do the work of `{step['id']}`.\n")
        declared.append({**step, "persona": "ana", "instructions": rel})
    (wdir / "workflow.yaml").write_text(yaml.safe_dump({
        "name": "w", "description": "one workflow", "inputs": {"data_path": {"type": "path"}},
        "env": "default", "steps": declared,
    }, sort_keys=False))
    return load_cartridge(root)
