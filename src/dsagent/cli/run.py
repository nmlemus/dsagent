"""One-shot task execution CLI for DSAgent.

This module provides the `dsagent run` command for executing
single tasks without interactive mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console

from dsagent import PlannerAgent, EventType, HITLMode
from dsagent.core.context import RunContext
from dsagent.utils.validation import validate_configuration, ConfigurationError


def get_hitl_input(prompt: str, options: dict[str, str]) -> str:
    """Get user input for HITL decisions."""
    options_str = " / ".join(f"[{k}]{v[1:]}" for k, v in options.items())
    while True:
        try:
            choice = input(f"{prompt}\n{options_str}? > ").strip().lower()
            if choice in options:
                return choice
            print(f"Invalid choice. Please enter one of: {', '.join(options.keys())}")
        except EOFError:
            return "a"


def get_multiline_input(prompt: str) -> str:
    """Get multiline input from user."""
    print(f"{prompt} (end with empty line):")
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)


def run_task(args) -> int:
    """Run a one-shot task.

    Args:
        args: Namespace with task, data, model, workspace, etc.

    Returns:
        Exit code (0 for success)
    """
    console = Console()

    # Resolve model (CLI flag, env, or fallback) before validation
    from dsagent.config import get_default_model
    effective_model = get_default_model(explicit=args.model)

    try:
        validate_configuration(effective_model)
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        return 1

    # Create run context
    workspace = Path(args.workspace).resolve()
    context = RunContext(workspace=workspace)

    # Copy data if provided
    data_info = None
    if args.data:
        try:
            data_info = context.copy_data(args.data)
        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            return 1

    # Parse HITL mode
    hitl_mode = HITLMode(args.hitl)

    # Validate MCP config if provided
    mcp_config_path = None
    if args.mcp_config:
        mcp_config_path = Path(args.mcp_config).expanduser().resolve()
        if not mcp_config_path.exists():
            console.print(f"[red]Error: MCP config file not found: {mcp_config_path}[/red]")
            return 1

    # Print run info
    console.print(f"[cyan]Run ID:[/cyan] {context.run_id}")
    if data_info:
        console.print(f"[cyan]Data:[/cyan] {data_info}")
    console.print(f"[cyan]Run Path:[/cyan] {context.run_path}")
    console.print(f"[cyan]Model:[/cyan] {effective_model}")
    if hitl_mode != HITLMode.NONE:
        console.print(f"[cyan]HITL Mode:[/cyan] {hitl_mode.value}")
    if mcp_config_path:
        console.print(f"[cyan]MCP Config:[/cyan] {mcp_config_path}")
    console.print("-" * 60)

    # Build task with data context
    if data_info:
        task_with_context = f"""
{args.task}

The data is available in the 'data/' subdirectory of the current working directory.
List files in 'data/' first to see what's available.
"""
    else:
        task_with_context = args.task

    # Create and run agent
    agent = PlannerAgent(
        model=effective_model,
        workspace=context.run_path,
        max_rounds=args.max_rounds,
        verbose=not args.quiet,
        context=context,
        hitl=hitl_mode,
        mcp_config=mcp_config_path,
    )

    try:
        agent.start()

        # Streaming mode with HITL handling
        for event in agent.run_stream(task_with_context):
            if event.type == EventType.HITL_AWAITING_PLAN_APPROVAL:
                console.print("\n" + "=" * 60)
                console.print("[bold]PLAN APPROVAL REQUIRED[/bold]")
                console.print("=" * 60)
                if event.plan:
                    console.print(event.plan.raw_text)
                console.print("-" * 60)
                choice = get_hitl_input(
                    "Review the plan above",
                    {"a": "approve", "r": "reject", "m": "modify"}
                )
                if choice == "a":
                    agent.approve()
                elif choice == "r":
                    reason = input("Rejection reason (optional): ").strip()
                    agent.reject(reason or None)
                elif choice == "m":
                    new_plan = get_multiline_input("Enter modified plan")
                    agent.modify_plan(new_plan)

            elif event.type == EventType.HITL_AWAITING_CODE_APPROVAL:
                console.print("\n" + "=" * 60)
                console.print("[bold]CODE APPROVAL REQUIRED[/bold]")
                console.print("=" * 60)
                if event.code:
                    console.print(event.code)
                console.print("-" * 60)
                choice = get_hitl_input(
                    "Review the code above",
                    {"a": "approve", "r": "reject", "m": "modify", "s": "skip"}
                )
                if choice == "a":
                    agent.approve()
                elif choice == "r":
                    reason = input("Rejection reason (optional): ").strip()
                    agent.reject(reason or None)
                elif choice == "m":
                    new_code = get_multiline_input("Enter modified code")
                    agent.modify_code(new_code)
                elif choice == "s":
                    agent.skip()

            elif event.type == EventType.HITL_AWAITING_ERROR_GUIDANCE:
                console.print("\n" + "=" * 60)
                console.print("[bold red]ERROR - GUIDANCE REQUIRED[/bold red]")
                console.print("=" * 60)
                if event.code:
                    console.print("Code:")
                    console.print(event.code)
                if event.error:
                    console.print("\nError:")
                    console.print(event.error)
                console.print("-" * 60)
                choice = get_hitl_input(
                    "How should I proceed?",
                    {"r": "retry", "m": "modify", "s": "skip", "a": "abort", "f": "feedback"}
                )
                if choice == "r":
                    agent.approve()
                elif choice == "m":
                    new_code = get_multiline_input("Enter modified code")
                    agent.modify_code(new_code)
                elif choice == "s":
                    agent.skip()
                elif choice == "a":
                    agent.reject("User aborted")
                elif choice == "f":
                    feedback = input("Enter feedback for the agent: ").strip()
                    agent.send_feedback(feedback)

            elif event.type == EventType.HITL_EXECUTION_ABORTED:
                console.print("\n[yellow][ABORTED] Execution was aborted by user[/yellow]")

            elif args.quiet:
                if event.type == EventType.ROUND_STARTED:
                    console.print(f"Round {event.message}")
                elif event.type == EventType.CODE_SUCCESS:
                    console.print("  [green][OK][/green] Code executed")
                elif event.type == EventType.CODE_FAILED:
                    console.print("  [red][FAIL][/red] Code failed")
                elif event.type == EventType.ANSWER_ACCEPTED:
                    console.print(f"\nAnswer:\n{event.message}")

        # Get final result
        result = agent.get_result()

        console.print("\n" + "=" * 60)
        console.print(f"[cyan]Run ID:[/cyan] {context.run_id}")
        console.print(f"[cyan]Notebook:[/cyan] {result.notebook_path}")
        console.print(f"[cyan]Artifacts:[/cyan] {context.artifacts_path}")
        console.print(f"[cyan]Logs:[/cyan] {context.logs_path}")
        console.print(f"[cyan]Rounds:[/cyan] {result.rounds}")
        console.print("=" * 60)

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    finally:
        agent.shutdown()
