#!/usr/bin/env python3
"""Command-line interface for the Aiuda Planner Agent."""

import argparse
import os
import sys
from pathlib import Path

# Load .env file if exists (before importing agent)
from dotenv import load_dotenv

# Try multiple locations for .env (first found wins)
_env_locations = [
    Path.cwd() / ".env",                          # Current directory
    Path(__file__).parent.parent.parent.parent / ".env",  # Project root
    Path.home() / ".dsagent" / ".env",              # User config
]
for _env_path in _env_locations:
    if _env_path.exists():
        load_dotenv(_env_path)
        break

from dsagent import PlannerAgent, EventType, HITLMode
from dsagent.core.context import RunContext
from dsagent.utils.validation import validate_configuration, ConfigurationError


def get_hitl_input(prompt: str, options: dict[str, str]) -> str:
    """Get user input for HITL decisions.

    Args:
        prompt: The prompt to display
        options: Dict mapping single char to action name

    Returns:
        The selected action key
    """
    options_str = " / ".join(f"[{k}]{v[1:]}" for k, v in options.items())
    while True:
        try:
            choice = input(f"{prompt}\n{options_str}? > ").strip().lower()
            if choice in options:
                return choice
            print(f"Invalid choice. Please enter one of: {', '.join(options.keys())}")
        except EOFError:
            return "a"  # Default to approve if no input available


def get_multiline_input(prompt: str) -> str:
    """Get multiline input from user (end with empty line).

    Args:
        prompt: The prompt to display

    Returns:
        The multiline input as a string
    """
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


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="AI Planner Agent - Analyze data with dynamic planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dsagent "Analyze sales data" --data ./data/sales.csv
  dsagent "Build a predictive model" --data ./dataset --model gpt-4o
  dsagent "Create visualizations" --data ./data --workspace ./output
  dsagent "Analyze with approval" --data ./data --hitl plan_only
  dsagent "Search and analyze" --mcp-config ~/.dsagent/mcp.yaml
  dsagent "Write code to calculate fibonacci" --model claude-3-5-sonnet-20241022
        """,
    )

    parser.add_argument(
        "task",
        type=str,
        help="The task to perform (e.g., 'Analyze this dataset and find trends')",
    )

    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to data file or directory to analyze (optional)",
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.getenv("LLM_MODEL", "gpt-4o"),
        help="LLM model to use (default: gpt-4o)",
    )

    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default="./workspace",
        help="Workspace directory for outputs (default: ./workspace)",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID (auto-generated if not provided)",
    )

    parser.add_argument(
        "--max-rounds", "-r",
        type=int,
        default=30,
        help="Maximum agent iterations (default: 30)",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output",
    )

    parser.add_argument(
        "--hitl",
        type=str,
        choices=["none", "plan_only", "on_error", "plan_and_answer", "full"],
        default="none",
        help="Human-in-the-Loop mode (default: none)",
    )

    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP servers YAML config file (e.g., ~/.dsagent/mcp.yaml)",
    )

    args = parser.parse_args()

    # Resolve model (CLI flag, env, or fallback) before validation
    from dsagent.config import get_default_model
    effective_model = get_default_model(explicit=args.model)

    try:
        validate_configuration(effective_model)
    except ConfigurationError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Create run context with isolated workspace
    workspace = Path(args.workspace).resolve()
    context = RunContext(workspace=workspace, run_id=args.run_id)

    # Copy data to run-specific data directory (if provided)
    data_info = None
    if args.data:
        try:
            data_info = context.copy_data(args.data)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Parse HITL mode
    hitl_mode = HITLMode(args.hitl)

    # Validate MCP config if provided
    mcp_config_path = None
    if args.mcp_config:
        mcp_config_path = Path(args.mcp_config).expanduser().resolve()
        if not mcp_config_path.exists():
            print(f"Error: MCP config file not found: {mcp_config_path}", file=sys.stderr)
            sys.exit(1)

    print(f"Run ID: {context.run_id}")
    if data_info:
        print(f"Data: {data_info}")
    print(f"Run Path: {context.run_path}")
    print(f"Model: {effective_model}")
    if hitl_mode != HITLMode.NONE:
        print(f"HITL Mode: {hitl_mode.value}")
    if mcp_config_path:
        print(f"MCP Config: {mcp_config_path}")
    print("-" * 60)

    # Build task with data context (only mention data folder if data was provided)
    if data_info:
        task_with_context = f"""
{args.task}

The data is available in the 'data/' subdirectory of the current working directory.
List files in 'data/' first to see what's available.
"""
    else:
        task_with_context = args.task

    # Create and run agent with context
    agent = PlannerAgent(
        model=effective_model,
        workspace=context.run_path,  # Use run-specific path
        max_rounds=args.max_rounds,
        verbose=not args.quiet,
        context=context,
        hitl=hitl_mode,
        mcp_config=mcp_config_path,
    )

    try:
        agent.start()

        if args.no_stream:
            # Synchronous mode
            result = agent.run(task_with_context)
            print("\n" + "=" * 60)
            print("RESULT")
            print("=" * 60)
            print(result.answer)
        else:
            # Streaming mode
            for event in agent.run_stream(task_with_context):
                # Handle HITL events
                if event.type == EventType.HITL_AWAITING_PLAN_APPROVAL:
                    print("\n" + "=" * 60)
                    print("PLAN APPROVAL REQUIRED")
                    print("=" * 60)
                    if event.plan:
                        print(event.plan.raw_text)
                    print("-" * 60)
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
                    print("\n" + "=" * 60)
                    print("CODE APPROVAL REQUIRED")
                    print("=" * 60)
                    if event.code:
                        print(event.code)
                    print("-" * 60)
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
                    print("\n" + "=" * 60)
                    print("ERROR - GUIDANCE REQUIRED")
                    print("=" * 60)
                    if event.code:
                        print("Code:")
                        print(event.code)
                    if event.error:
                        print("\nError:")
                        print(event.error)
                    print("-" * 60)
                    choice = get_hitl_input(
                        "How should I proceed?",
                        {"r": "retry", "m": "modify", "s": "skip", "a": "abort", "f": "feedback"}
                    )
                    if choice == "r":
                        agent.approve()  # Retry
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

                elif event.type == EventType.HITL_AWAITING_ANSWER_APPROVAL:
                    print("\n" + "=" * 60)
                    print("ANSWER APPROVAL REQUIRED")
                    print("=" * 60)
                    if event.message:
                        print(event.message)
                    print("-" * 60)
                    choice = get_hitl_input(
                        "Accept this answer?",
                        {"a": "approve", "r": "reject", "f": "feedback"}
                    )
                    if choice == "a":
                        agent.approve()
                    elif choice == "r":
                        reason = input("Why is this answer not acceptable? ").strip()
                        agent.reject(reason or None)
                    elif choice == "f":
                        feedback = input("Enter feedback to improve the answer: ").strip()
                        agent.send_feedback(feedback)

                elif event.type == EventType.HITL_EXECUTION_ABORTED:
                    print("\n[ABORTED] Execution was aborted by user")

                # Standard event handling
                elif args.quiet:
                    # In quiet mode, only show key events
                    if event.type == EventType.ROUND_STARTED:
                        print(f"Round {event.message}")
                    elif event.type == EventType.CODE_SUCCESS:
                        print("  [OK] Code executed")
                    elif event.type == EventType.CODE_FAILED:
                        print("  [FAIL] Code failed")
                    elif event.type == EventType.ANSWER_ACCEPTED:
                        print(f"\nAnswer:\n{event.message}")

            # Get final result with notebook
            result = agent.get_result()

        print("\n" + "=" * 60)
        print(f"Run ID: {context.run_id}")
        print(f"Notebook: {result.notebook_path}")
        print(f"Artifacts: {context.artifacts_path}")
        print(f"Logs: {context.logs_path}")
        print(f"Rounds: {result.rounds}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        agent.shutdown()


if __name__ == "__main__":
    main()
