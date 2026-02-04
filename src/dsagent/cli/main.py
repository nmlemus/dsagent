"""Main CLI entry point for DSAgent.

Provides subcommands:
- dsagent chat     : Interactive conversational mode (default)
- dsagent run      : One-shot task execution
- dsagent init     : Setup wizard
- dsagent mcp      : MCP server management
- dsagent skills   : Agent Skills management
- dsagent serve    : Run API server (REST + WebSocket)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env files: global first, then local (local overrides global)
_global_env = Path.home() / ".dsagent" / ".env"
_local_env = Path.cwd() / ".env"

# Load global config first (defaults)
if _global_env.exists():
    load_dotenv(_global_env)

# Load local config second (overrides global)
if _local_env.exists():
    load_dotenv(_local_env, override=True)


def cmd_chat(args: argparse.Namespace) -> int:
    """Run the interactive chat REPL."""
    from dsagent.cli.repl import run_chat
    return run_chat(args)


def cmd_run(args: argparse.Namespace) -> int:
    """Run a one-shot task."""
    from dsagent.cli.run import run_task
    return run_task(args)


def cmd_init(args: argparse.Namespace) -> int:
    """Run the setup wizard."""
    from dsagent.cli.init import run_init
    return run_init(args)


def cmd_mcp(args: argparse.Namespace) -> int:
    """Manage MCP servers."""
    from dsagent.cli.mcp_cmd import run_mcp
    return run_mcp(args)


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the API server."""
    try:
        from dsagent.server.app import run_server
    except ImportError:
        print("Error: API dependencies not installed.", file=sys.stderr)
        print("Install with: pip install datascience-agent[api]", file=sys.stderr)
        return 1

    print(f"Starting DSAgent API Server on {args.host}:{args.port}")
    if args.reload:
        print("  Auto-reload enabled (development mode)")

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    """Manage skills."""
    from dsagent.cli.skills_cmd import run_skills
    return run_skills(args)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="dsagent",
        description="DSAgent - AI-powered Data Science Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dsagent                          # Start interactive chat (default)
  dsagent chat                     # Same as above
  dsagent chat --model claude-sonnet-4-5
  dsagent run "Analyze sales.csv"  # One-shot task
  dsagent init                     # Setup wizard
  dsagent mcp add brave-search     # Add MCP server
  dsagent skills list              # List installed skills
  dsagent skills install github:dsagent-skills/eda-analysis
  dsagent serve --port 8000        # Start API server

For more info on a command:
  dsagent <command> --help
        """,
    )

    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========== chat subcommand ==========
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive conversational mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dsagent chat
  dsagent chat --model gpt-4o
  dsagent chat --session abc123
  dsagent chat --mcp-config ~/.dsagent/mcp.yaml
        """,
    )
    chat_parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,  # Resolved via get_default_model()
        help="LLM model to use (default: from DSAGENT_DEFAULT_MODEL or LLM_MODEL env var)",
    )
    chat_parser.add_argument(
        "--workspace", "-w",
        type=str,
        default=os.getenv("DSAGENT_WORKSPACE", "./workspace"),
        help="Workspace directory (default: ./workspace or $DSAGENT_WORKSPACE)",
    )
    chat_parser.add_argument(
        "--session", "-s",
        type=str,
        default=None,
        help="Session ID to resume",
    )
    chat_parser.add_argument(
        "--hitl",
        type=str,
        choices=["none", "plan", "full", "plan_answer", "on_error"],
        default="none",
        help="Human-in-the-loop mode (default: none)",
    )
    chat_parser.add_argument(
        "--live-notebook",
        action="store_true",
        help="Enable live notebook (saves after each execution)",
    )
    chat_parser.add_argument(
        "--notebook-sync",
        action="store_true",
        help="Enable bidirectional notebook sync with Jupyter",
    )
    chat_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP servers YAML config file",
    )
    chat_parser.set_defaults(func=cmd_chat)

    # ========== run subcommand ==========
    run_parser = subparsers.add_parser(
        "run",
        help="Run a one-shot task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dsagent run "Analyze this dataset" --data ./data/sales.csv
  dsagent run "Build a predictive model" --data ./dataset
  dsagent run "Create visualizations" --workspace ./output
        """,
    )
    run_parser.add_argument(
        "task",
        type=str,
        help="The task to perform",
    )
    run_parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to data file or directory",
    )
    run_parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,  # Resolved via get_default_model()
        help="LLM model to use (default: from DSAGENT_DEFAULT_MODEL or LLM_MODEL env var)",
    )
    run_parser.add_argument(
        "--workspace", "-w",
        type=str,
        default=os.getenv("DSAGENT_WORKSPACE", "./workspace"),
        help="Workspace directory (default: ./workspace or $DSAGENT_WORKSPACE)",
    )
    run_parser.add_argument(
        "--max-rounds", "-r",
        type=int,
        default=30,
        help="Maximum agent iterations (default: 30)",
    )
    run_parser.add_argument(
        "--hitl",
        type=str,
        choices=["none", "plan_only", "on_error", "plan_and_answer", "full"],
        default="none",
        help="Human-in-the-loop mode (default: none)",
    )
    run_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP servers YAML config file",
    )
    run_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    run_parser.set_defaults(func=cmd_run)

    # ========== init subcommand ==========
    init_parser = subparsers.add_parser(
        "init",
        help="Setup wizard for DSAgent configuration",
        epilog="""
Interactively configure:
  - LLM provider and API keys
  - MCP tools (web search, etc.)
  - Default settings
        """,
    )
    init_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing configuration",
    )
    init_parser.set_defaults(func=cmd_init)

    # ========== mcp subcommand ==========
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Manage MCP servers",
        epilog="""
Commands:
  dsagent mcp list              # List configured servers
  dsagent mcp add <template>    # Add from template
  dsagent mcp remove <name>     # Remove a server
        """,
    )
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")

    # mcp list
    mcp_list = mcp_subparsers.add_parser("list", help="List configured MCP servers")
    mcp_list.set_defaults(mcp_action="list")

    # mcp add
    mcp_add = mcp_subparsers.add_parser("add", help="Add MCP server from template")
    mcp_add.add_argument("template", help="Template name (e.g., brave-search, filesystem)")
    mcp_add.set_defaults(mcp_action="add")

    # mcp remove
    mcp_remove = mcp_subparsers.add_parser("remove", help="Remove MCP server")
    mcp_remove.add_argument("name", help="Server name to remove")
    mcp_remove.set_defaults(mcp_action="remove")

    mcp_parser.set_defaults(func=cmd_mcp)

    # ========== serve subcommand ==========
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the API server (REST + WebSocket)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dsagent serve                    # Start server on 0.0.0.0:8000
  dsagent serve --port 3000        # Custom port
  dsagent serve --reload           # Enable auto-reload (dev mode)
  dsagent serve --host 127.0.0.1   # Localhost only

Environment variables:
  DSAGENT_API_KEY     : Enable API key authentication
  DSAGENT_CORS_ORIGINS: Comma-separated allowed origins (default: *)

API Endpoints:
  GET  /health                    : Health check
  POST /api/sessions              : Create session
  GET  /api/sessions              : List sessions
  POST /api/sessions/{id}/chat    : Send message
  WS   /ws/chat/{session_id}      : WebSocket chat
        """,
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    serve_parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)",
    )
    serve_parser.set_defaults(func=cmd_serve)

    # ========== skills subcommand ==========
    skills_parser = subparsers.add_parser(
        "skills",
        help="Manage Agent Skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dsagent skills list                                    # List installed skills
  dsagent skills install github:dsagent-skills/eda      # Install from GitHub
  dsagent skills install ./my-skill                     # Install from local
  dsagent skills remove eda-analysis                    # Remove a skill
  dsagent skills info eda-analysis                      # Show skill details

Skills extend the agent's capabilities with specialized knowledge.
They are stored in ~/.dsagent/skills/
        """,
    )
    skills_subparsers = skills_parser.add_subparsers(dest="skills_command")

    # skills list
    skills_list = skills_subparsers.add_parser("list", help="List installed skills")
    skills_list.set_defaults(skills_action="list")

    # skills install
    skills_install = skills_subparsers.add_parser("install", help="Install a skill")
    skills_install.add_argument(
        "source",
        help="Skill source (github:owner/repo, github:owner/repo/path, or local path)",
    )
    skills_install.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing skill",
    )
    skills_install.set_defaults(skills_action="install")

    # skills remove
    skills_remove = skills_subparsers.add_parser("remove", help="Remove a skill")
    skills_remove.add_argument("name", help="Skill name to remove")
    skills_remove.set_defaults(skills_action="remove")

    # skills info
    skills_info = skills_subparsers.add_parser("info", help="Show skill details")
    skills_info.add_argument("name", help="Skill name")
    skills_info.set_defaults(skills_action="info")

    skills_parser.set_defaults(func=cmd_skills)

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()

    # Check if we need to default to 'chat' subcommand
    # This handles cases like: dsagent --model gpt-4o (no subcommand specified)
    valid_commands = {'chat', 'run', 'init', 'mcp', 'serve', 'skills'}
    argv = sys.argv[1:]

    # If no args, or first arg starts with '-', or first arg is not a valid command
    # then prepend 'chat' as the default subcommand
    if not argv or argv[0].startswith('-') or argv[0] not in valid_commands:
        # But skip if it's --version or --help
        if not argv or argv[0] not in ('--version', '-v', '--help', '-h'):
            argv = ['chat'] + argv

    args = parser.parse_args(argv)

    # Handle --version
    if args.version:
        from dsagent import __version__
        print(f"dsagent {__version__}")
        return 0

    # Run the command
    if hasattr(args, 'func'):
        try:
            return args.func(args)
        except KeyboardInterrupt:
            print("\nInterrupted")
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
