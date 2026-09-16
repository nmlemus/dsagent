"""`dsagent` CLI.

    dsagent cartridge validate ./cartridges/ds
    dsagent cartridge list ./cartridges/ds
    dsagent run eda-to-report --cartridge ./cartridges/ds --input data_path=data/sales.csv
    dsagent chat --cartridge ./cartridges/ds
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dsagent import __version__, requirements
from dsagent.cartridge import CartridgeError, load_cartridge, load_cartridges
from dsagent.cartridge.models import Cartridge, EnvSpec
from dsagent.runner import (
    FILE_EVENT,
    STEP_EVENT,
    TOOL_EVENT,
    GateDecision,
    RunnerEvent,
    WorkflowRunner,
    dispatch_runner_event,
    visible_input_names,
)

app = typer.Typer(help="DSAgent v2 — cartridge-driven Deep Agents harness.", no_args_is_help=True)
cartridge_app = typer.Typer(help="Inspect and validate cartridges.")
app.add_typer(cartridge_app, name="cartridge")
console = Console()

DEFAULT_CARTRIDGE = Path("cartridges/ds")
RUNS_DIR = Path(".dsagent/runs")


def workflow_run_id(workflow: str, tool_call_id: str = "") -> str:
    """The run directory name for one `run_workflow` invocation.

    Derived from the tool call, never from the clock. Under `dsagent serve` the
    gate is a LangGraph `interrupt()`, and resuming re-executes `run_workflow`
    from the top: a timestamped id would mint a fresh, empty run directory on
    every re-entry, so `resume` would find no `run.json`, every step would read
    as `pending`, and the run would redo — and re-pay for — work it had already
    done. `tool_call_id` is stable across the original call and the re-entry,
    because it belongs to the `AIMessage` the checkpoint replays (verified
    against a `create_deep_agent` graph).

    The timestamp remains the fallback for a caller that reaches this without a
    tool call — a direct programmatic call, where there is nothing to re-enter.
    `docs/ui-slice.md` §2 names `thread_id` plus a counter in graph state as the
    other option; it is strictly more machinery for the same guarantee, so it
    stays unbuilt until something needs it.
    """
    suffix = tool_call_id or time.strftime("%Y%m%d-%H%M%S")
    return f"{workflow}-{suffix}"


def _parse_inputs(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise typer.BadParameter(f"--input expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def _ask(prompt: str) -> GateDecision:
    return GateDecision.APPROVE if typer.confirm(prompt, default=False) else GateDecision.REJECT


@app.callback()
def _version(version: bool = typer.Option(False, "--version", is_eager=True)):
    if version:
        console.print(f"dsagent {__version__}")
        raise typer.Exit()


@cartridge_app.command("validate")
def cartridge_validate(path: Path = typer.Argument(DEFAULT_CARTRIDGE)):
    """Load a cartridge, check the persona/skill matrix, regenerate command skills."""
    try:
        c = load_cartridge(path)
    except CartridgeError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] cartridge [bold]{c.name}[/bold] v{c.version} is consistent")
    console.print(f"  personas: {len(c.personas)} · skills: {len(c.skills)} · workflows: {len(c.workflows)} · envs: {len(c.envs)}")
    console.print(f"  command skills regenerated: {', '.join(f'/{c.name}-{w}' for w in c.workflows)}")
    for wf_name, blind in _blind_steps(c).items():
        console.print(
            f"  [yellow]![/yellow] workflow '{wf_name}': step(s) {', '.join(blind)} are shown no "
            f"inputs — their instructions interpolate none and they declare no `sees:`"
        )
    for env, missing in _unsatisfied_envs(c).items():
        console.print(f"  [yellow]![/yellow] env '{env}' missing: {', '.join(missing)}")
    if _unsatisfied_envs(c):
        console.print(f"  run [bold]dsagent cartridge install {path}[/bold] to install them")


def _blind_steps(c: Cartridge) -> dict[str, list[str]]:
    """Steps of an input-taking workflow that end up seeing no input at all.

    Almost always a step whose instructions name an input in prose but never
    interpolate it, so the value silently stops reaching the persona.
    """
    out: dict[str, list[str]] = {}
    for wf in c.workflows.values():
        if not wf.inputs:
            continue
        blind = [
            s.id
            for s in wf.steps
            if s.sees is None
            and not visible_input_names(s, (wf.path / s.instructions).read_text(encoding="utf-8"))
        ]
        if blind:
            out[wf.name] = blind
    return out


def _kernel_envs(c: Cartridge) -> list[EnvSpec]:
    """Docker envs install their requirements in the Dockerfile, not here."""
    return [e for e in c.envs.values() if e.kind == "kernel" and e.requirements]


def _unsatisfied_envs(c: Cartridge) -> dict[str, list[str]]:
    found = {e.name: requirements.missing(e.requirements) for e in _kernel_envs(c)}
    return {name: missing for name, missing in found.items() if missing}


@cartridge_app.command("install")
def cartridge_install(
    path: Path = typer.Argument(DEFAULT_CARTRIDGE),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the command, install nothing"),
):
    """Pip-install the requirements this cartridge's kernel envs declare.

    Installs into the interpreter running dsagent, which is the one the kernel
    env uses. Docker envs are skipped: their dependencies belong in the image.
    """
    c = load_cartridge(path)
    kernel_envs = _kernel_envs(c)
    skipped = [e.name for e in c.envs.values() if e.kind != "kernel" and e.requirements]
    if skipped:
        console.print(f"[dim]skipping docker env(s) {', '.join(skipped)} — deps live in the Dockerfile[/dim]")
    if not kernel_envs:
        console.print(f"cartridge [bold]{c.name}[/bold] declares no kernel-env requirements")
        return

    wanted: list[str] = []
    for e in kernel_envs:
        console.print(f"  env '{e.name}': {', '.join(e.requirements)}")
        wanted.extend(r for r in e.requirements if r not in wanted)

    argv = [sys.executable, "-m", "pip", "install", *wanted]
    # printed quoted so it survives a copy-paste: a bare >= is a shell redirect
    console.print(f"[dim]{requirements.pip_install_command(wanted)}[/dim]")
    if dry_run:
        console.print("[dim]dry run — nothing installed[/dim]")
        return
    if subprocess.run(argv, check=False).returncode != 0:
        console.print("[red]pip install failed[/red]")
        raise typer.Exit(1)

    still_missing = _unsatisfied_envs(c)
    if still_missing:
        for env, missing in still_missing.items():
            console.print(f"[red]✗[/red] env '{env}' still missing: {', '.join(missing)}")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] {len(wanted)} requirement(s) satisfied for {c.name}")


@cartridge_app.command("list")
def cartridge_list(path: Path = typer.Argument(DEFAULT_CARTRIDGE)):
    """Show the persona ↔ skill ↔ workflow matrix."""
    c = load_cartridge(path)
    t = Table(title=f"{c.name} v{c.version}")
    t.add_column("Persona", style="bold")
    t.add_column("Role")
    for s in c.skills:
        t.add_column(s, justify="center")
    t.add_column("Workflows")
    for p in c.personas.values():
        cells = ["●" if s in p.skills else "" for s in c.skills]
        t.add_row(p.name, p.role, *cells, ", ".join(p.workflows))
    console.print(t)
    for w in c.workflows.values():
        console.print(f"[bold]{w.name}[/bold] (env={w.env}): " + " → ".join(
            f"{s.id}[{s.persona}]" + ("⏸" if s.gate and s.gate.kind == "human" else "✓" if s.gate else "")
            for s in w.ordered_steps()))


def _human_size(n: int) -> str:
    return f"{n} B" if n < 1024 else (f"{n / 1024:.1f} kB" if n < 1024 * 1024 else f"{n / 1048576:.1f} MB")


def _print_event(e: RunnerEvent) -> None:
    """Render the runner's stream the way `docs/runs/*.md` draws a timeline."""
    v = e.value
    if e.name == STEP_EVENT:
        status = v["status"]
        if status == "started":
            console.print(f"[bold]▶ {v['step']}[/bold] ({v['persona']}) {v['index'] + 1}/{v['total']}")
        elif status == "done":
            console.print(f"[green]◀ {v['step']} done[/green]")
        elif status == "failed":
            console.print(f"[red]◀ {v['step']} failed[/red] {v.get('error') or ''}", markup=False)
        else:
            console.print(f"[yellow]⏸ {v['step']} {status}[/yellow]")
    elif e.name == TOOL_EVENT and v["phase"] == "started":
        args = ", ".join(f"{k}={val}" for k, val in (v.get("args_preview") or {}).items())
        console.print(f"  · {v['tool']}({args})", style="dim", markup=False)
    elif e.name == FILE_EVENT:
        mark = "[cyan]●[/cyan]" if v["kind"] == "deliverable" else "○"
        console.print(f"  {mark} {v['path']}  [dim]({v['kind']}, {_human_size(v['size'])})[/dim]")


@app.command()
def run(
    workflow: str,
    cartridge: Path = typer.Option(DEFAULT_CARTRIDGE, "--cartridge", "-c"),
    input: list[str] = typer.Option([], "--input", "-i", help="key=value, repeatable"),
    run_id: str | None = typer.Option(None, "--run-id", help="reuse a run directory"),
    resume: bool = typer.Option(False, "--resume", help="continue a paused/failed run"),
    dry_run: bool = typer.Option(False, "--dry-run", help="walk the DAG without calling any model"),
    yes: bool = typer.Option(False, "--yes", "-y", help="auto-approve human gates"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="only print the final result"),
):
    """Run a workflow from a cartridge."""
    c = load_cartridge(cartridge)
    if workflow not in c.workflows:
        console.print(f"[red]unknown workflow[/red] {workflow}. Available: {', '.join(c.workflows)}")
        raise typer.Exit(1)
    run_id = run_id or f"{workflow}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    runner = WorkflowRunner(
        c, run_dir,
        ask_human=(lambda p: GateDecision.APPROVE) if yes else _ask,
        log=lambda m: console.print(m, style="dim", markup=False),
        on_event=_print_event if not quiet else None,
    )
    state = runner.run(workflow, _parse_inputs(input), resume=resume, dry_run=dry_run)
    color = {"done": "green", "failed": "red"}.get(state.status, "yellow")
    console.print(f"[{color}]run {run_id}: {state.status}[/{color}]  → {run_dir}")
    if state.status == "failed":
        for s in state.steps.values():
            if s.error:
                console.print(f"  [red]{s.id}[/red]: {s.error}")
        raise typer.Exit(1)


@app.command()
def chat(
    cartridge: list[Path] = typer.Option([DEFAULT_CARTRIDGE], "--cartridge", "-c", help="repeatable"),
    workspace: Path = typer.Option(Path(".dsagent/chat"), "--workspace"),
    model: str | None = typer.Option(None, "--model"),
):
    """Interactive session with the orchestrator (personas as subagents)."""
    from typing import Annotated

    from langchain_core.tools import InjectedToolCallId, tool

    from dsagent.envs import make_env
    from dsagent.host import build_orchestrator

    carts = load_cartridges(cartridge)
    by_wf = {w: c for c in carts for w in c.workflows}
    env = make_env(carts[0].envs["default"], workspace)

    @tool
    def list_workflows() -> str:
        """List workflows available in the loaded cartridges."""
        return json.dumps({w: c.workflows[w].description for w, c in by_wf.items()}, indent=2)

    @tool
    def run_workflow(
        name: str,
        inputs: dict | None = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> str:
        """Run a cartridge workflow end to end. `inputs` is a dict matching the workflow's declared inputs."""
        if name not in by_wf:
            return f"unknown workflow {name}; use list_workflows"
        run_dir = RUNS_DIR / workflow_run_id(name, tool_call_id)
        resume = (run_dir / "run.json").exists()
        def on_event(e: RunnerEvent) -> None:
            # Two consumers: the terminal running `dsagent chat`, and whatever is
            # attached to the graph's event stream — a browser, once PR 4 lands.
            _print_event(e)
            dispatch_runner_event(e)

        state = WorkflowRunner(by_wf[name], run_dir, ask_human=_ask,
                               log=lambda m: console.print(m, style="dim", markup=False),
                               on_event=on_event).run(name, inputs or {}, resume=resume)
        # `status` is the work and no longer says "paused", so report the gate
        # decisions alongside it — otherwise a paused run reads as all-done.
        return json.dumps({
            "run_dir": str(run_dir), "status": state.status,
            "steps": {k: v.status for k, v in state.steps.items()},
            "gates": {k: v.gate.decision for k, v in state.steps.items() if v.gate},
        })

    agent = build_orchestrator(carts, env, workspace, model=model, workflow_tools=[list_workflows, run_workflow])
    console.print(f"[bold]dsagent[/bold] {__version__} · cartridges: {', '.join(c.name for c in carts)} · Ctrl-C to quit")
    history: list[dict] = []
    try:
        while True:
            q = console.input("[bold cyan]you ›[/bold cyan] ").strip()
            if not q:
                continue
            history.append({"role": "user", "content": q})
            result = agent.invoke({"messages": history})
            history = result["messages"]
            console.print(f"[bold magenta]dsagent ›[/bold magenta] {history[-1].content}")
    except (KeyboardInterrupt, EOFError):
        env.close()
        console.print("\nbye")


if __name__ == "__main__":
    app()
