"""Conversational REPL for DSAgent."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from dsagent.cli.banner import print_welcome
from dsagent.cli.commands import CommandRegistry, CommandResult, create_default_registry
from dsagent.cli.renderer import CLIRenderer
from dsagent.session import Session, SessionManager
from dsagent.agents import ConversationalAgent, ConversationalAgentConfig
from dsagent.schema.models import HITLMode


# Load .env file
_env_locations = [
    Path.cwd() / ".env",
    Path(__file__).parent.parent.parent.parent / ".env",
    Path.home() / ".dsagent" / ".env",
]
for _env_path in _env_locations:
    if _env_path.exists():
        load_dotenv(_env_path)
        break


@dataclass
class CLIContext:
    """Context for CLI commands and operations."""

    manager: SessionManager
    registry: CommandRegistry
    console: Console
    session: Optional[Session] = None
    model: str = "gpt-4o"
    data_path: Optional[str] = None
    workspace: Path = field(default_factory=lambda: Path("./workspace"))

    # Agent components (initialized lazily)
    _agent: Optional[object] = field(default=None, repr=False)
    _kernel_running: bool = False

    def set_session(self, session: Session) -> None:
        """Set the active session."""
        self.session = session

    @property
    def agent(self):
        """Get or create the agent instance."""
        return self._agent

    def has_active_session(self) -> bool:
        """Check if there's an active session."""
        return self.session is not None


class SlashCommandCompleter(Completer):
    """Completer for slash commands."""

    def __init__(self, registry: CommandRegistry):
        self.registry = registry

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only complete if starting with /
        if text.startswith("/"):
            partial = text[1:]  # Remove the /
            for name in self.registry._commands.keys():
                if name.startswith(partial):
                    yield Completion(
                        f"/{name}",
                        start_position=-len(text),
                        display=f"/{name}",
                        display_meta=self.registry._commands[name].description
                    )
            for alias in self.registry._aliases.keys():
                if alias.startswith(partial):
                    cmd_name = self.registry._aliases[alias]
                    yield Completion(
                        f"/{alias}",
                        start_position=-len(text),
                        display=f"/{alias}",
                        display_meta=f"(alias for /{cmd_name})"
                    )


class ConversationalCLI:
    """Main conversational CLI for DSAgent."""

    def __init__(
        self,
        workspace: Path = Path("./workspace"),
        model: str = "gpt-4o",
        session_id: Optional[str] = None,
        hitl_mode: HITLMode = HITLMode.NONE,
        enable_live_notebook: bool = False,
        enable_notebook_sync: bool = False,
        mcp_config: Optional[Path] = None,
    ):
        self.workspace = Path(workspace).resolve()
        self.model = model
        self.initial_session_id = session_id
        self.hitl_mode = hitl_mode
        self.enable_live_notebook = enable_live_notebook
        self.enable_notebook_sync = enable_notebook_sync
        self.mcp_config = mcp_config

        # Initialize components
        self.console = Console()
        self.renderer = CLIRenderer(self.console)
        self.registry = create_default_registry()
        self.manager = SessionManager(self.workspace)

        # Agent (initialized lazily)
        self._agent: Optional[ConversationalAgent] = None

        # Create CLI context
        self.ctx = CLIContext(
            manager=self.manager,
            registry=self.registry,
            console=self.console,
            model=model,
            workspace=self.workspace,
        )

        # Prompt session with history
        history_path = self.workspace / ".dsagent" / "history"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        self.prompt_session: PromptSession = PromptSession(
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=SlashCommandCompleter(self.registry),
            style=Style.from_dict({
                "prompt": "bold cyan",
            }),
        )

    def _get_or_create_agent(self) -> ConversationalAgent:
        """Get or create the conversational agent."""
        if self._agent is None or not self._agent.is_running:
            config = ConversationalAgentConfig(
                model=self.ctx.model,
                workspace=self.workspace,
                hitl_mode=self.hitl_mode,
                enable_live_notebook=self.enable_live_notebook,
                enable_notebook_sync=self.enable_notebook_sync,
                mcp_config=self.mcp_config,
            )
            self._agent = ConversationalAgent(
                config=config,
                session=self.ctx.session,
                session_manager=self.manager,
            )
            self._agent.start(self.ctx.session)
            # Make agent available to commands through context
            self.ctx._agent = self._agent

            # Show live notebook path if enabled
            if self.enable_live_notebook or self.enable_notebook_sync:
                self.console.print(
                    f"[cyan]Live notebook will be created on first message[/cyan]"
                )

            # Set up callback for notebook changes (if sync enabled)
            if self.enable_notebook_sync:
                def on_notebook_change(changes):
                    for change in changes:
                        if change.change_type == "cell_added":
                            self.console.print(
                                f"\n[yellow]📝 Jupyter: New cell detected at index {change.cell_index}[/yellow]"
                            )
                            if change.new_content:
                                preview = change.new_content[:100]
                                if len(change.new_content) > 100:
                                    preview += "..."
                                self.console.print(f"[dim]{preview}[/dim]")
                        elif change.change_type == "cell_modified":
                            self.console.print(
                                f"\n[yellow]📝 Jupyter: Cell {change.cell_index} modified[/yellow]"
                            )
                        elif change.change_type == "cell_deleted":
                            self.console.print(
                                f"\n[yellow]📝 Jupyter: Cell {change.cell_index} deleted[/yellow]"
                            )
                self._agent.set_callbacks(on_notebook_change=on_notebook_change)
        return self._agent

    def _shutdown_agent(self) -> None:
        """Shutdown the agent if running."""
        if self._agent and self._agent.is_running:
            self._agent.shutdown()
            self._agent = None

    def _print_welcome(self) -> None:
        """Print welcome message with ASCII banner."""
        # Show ASCII art banner
        session_id = self.ctx.session.id if self.ctx.session else ""
        print_welcome(self.console, model=self.model, session_id=session_id)

        # Show additional mode info
        if self.hitl_mode != HITLMode.NONE:
            self.console.print(f"  [yellow]HITL Mode: {self.hitl_mode.value}[/yellow] - You'll be asked to approve actions.")

        if self.enable_notebook_sync:
            self.console.print("  [cyan]Live Notebook Sync: ON[/cyan] - Notebook updates in real-time, syncs with Jupyter.")
        elif self.enable_live_notebook:
            self.console.print("  [cyan]Live Notebook: ON[/cyan] - Notebook saves after each execution.")

        if self.hitl_mode != HITLMode.NONE or self.enable_notebook_sync or self.enable_live_notebook:
            self.console.print()

    def _print_status_bar(self) -> None:
        """Print current status."""
        parts = []

        if self.ctx.session:
            parts.append(f"[cyan]{self.ctx.session.name[:20]}[/cyan]")
            parts.append(f"msgs:{len(self.ctx.session.history)}")
        else:
            parts.append("[dim]No session[/dim]")

        parts.append(f"model:{self.ctx.model}")

        if self.hitl_mode != HITLMode.NONE:
            parts.append(f"[yellow]hitl:{self.hitl_mode.value}[/yellow]")

        if self.enable_notebook_sync:
            parts.append("[cyan]sync[/cyan]")
        elif self.enable_live_notebook:
            parts.append("[cyan]live[/cyan]")

        status = " | ".join(parts)
        self.console.print(f"[dim]{status}[/dim]")

    def _get_prompt(self) -> str:
        """Get the prompt string."""
        if self.ctx.session:
            return "You> "
        return "(no session) You> "

    def _handle_command(self, input_text: str) -> bool:
        """Handle a slash command.

        Returns:
            True if should continue, False if should exit
        """
        parts = input_text[1:].split(maxsplit=1)
        cmd_name = parts[0] if parts else ""
        args = parts[1].split() if len(parts) > 1 else []

        cmd = self.registry.get(cmd_name)
        if not cmd:
            self.console.print(f"[red]Unknown command: /{cmd_name}[/red]")
            self.console.print("Type /help for available commands")
            return True

        result = cmd.execute(self.ctx, args)

        if result.clear_screen:
            self.console.clear()

        if result.message:
            if result.success:
                self.console.print(result.message)
            else:
                self.console.print(f"[red]{result.message}[/red]")

        return not result.should_exit

    def _prompt_hitl_approval(self, item_type: str, content: str) -> bool:
        """Prompt user for HITL approval.

        Args:
            item_type: Type of item (plan, code)
            content: Content to show

        Returns:
            True if approved, False if rejected
        """
        self.console.print()
        self.console.print(
            Panel(
                content,
                title=f"[yellow]HITL: Approve {item_type}?[/yellow]",
                border_style="yellow",
            )
        )

        while True:
            try:
                # Print prompt with Rich, then get input
                self.console.print("[yellow](a)pprove / (r)eject / (s)kip:[/yellow] ", end="")
                choice = input().strip().lower()

                if choice in ("a", "approve", "y", "yes"):
                    self.console.print("[green]✓ Approved[/green]")
                    return True
                elif choice in ("r", "reject", "n", "no"):
                    self.console.print("[red]✗ Rejected - stopping execution[/red]")
                    return False
                elif choice in ("s", "skip"):
                    self.console.print("[yellow]→ Skipping this step[/yellow]")
                    return True  # Continue but skip
                else:
                    self.console.print("[dim]Enter 'a' to approve, 'r' to reject, or 's' to skip[/dim]")
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[red]✗ Cancelled[/red]")
                return False

    def _handle_chat(self, message: str) -> None:
        """Handle a chat message to the agent."""
        if not self.ctx.session:
            self.console.print(
                "[yellow]No active session. Creating a new one...[/yellow]"
            )
            session = self.manager.create_session()
            self.ctx.set_session(session)

        self.console.print()

        try:
            # Get or create agent
            agent = self._get_or_create_agent()

            # Define callback for HITL code approval
            def on_code_execute(code: str) -> None:
                if self.hitl_mode in (HITLMode.FULL,):
                    if not self._prompt_hitl_approval("Code", code):
                        raise KeyboardInterrupt("User rejected code execution")

            # Use streaming for progress updates
            round_num = 0
            plan_approved = False
            notebook_path_shown = False

            # Show thinking indicator while waiting for first response
            with self.console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots") as status:
                response_gen = agent.chat_stream(message, on_code_execute=on_code_execute)
                # Get first response (this is where LLM call happens)
                try:
                    first_response = next(response_gen)
                except StopIteration:
                    return

            # Process first response
            round_num = 1

            # Show live notebook path on first response (if enabled)
            if not notebook_path_shown and (self.enable_live_notebook or self.enable_notebook_sync):
                notebook_path = agent.get_live_notebook_path()
                if notebook_path:
                    self.console.print(f"[dim]Live notebook: {notebook_path}[/dim]")
                    notebook_path_shown = True

            # Show plan if present and ask for HITL approval
            if first_response.plan:
                self._display_plan(first_response.plan, round_num)

                # HITL: Ask for plan approval on first plan
                if not plan_approved and self.hitl_mode in (HITLMode.PLAN_ONLY, HITLMode.PLAN_AND_ANSWER, HITLMode.FULL):
                    plan_text = "\n".join(
                        f"{s.number}. {'[x]' if s.completed else '[ ]'} {s.description}"
                        for s in first_response.plan.steps
                    )
                    if not self._prompt_hitl_approval("Plan", plan_text):
                        self.console.print("[yellow]Execution stopped by user[/yellow]")
                        return
                    plan_approved = True

            # Display the first response
            self._display_response(first_response, round_num)

            # Continue with remaining responses (autonomous execution)
            while True:
                # Show thinking indicator for next round
                with self.console.status(f"[bold cyan]Thinking... (round {round_num + 1})[/bold cyan]", spinner="dots"):
                    try:
                        response = next(response_gen)
                    except StopIteration:
                        break

                round_num += 1

                # Show live notebook path on first response (if enabled)
                if not notebook_path_shown and (self.enable_live_notebook or self.enable_notebook_sync):
                    notebook_path = agent.get_live_notebook_path()
                    if notebook_path:
                        self.console.print(f"[dim]Live notebook: {notebook_path}[/dim]")
                        notebook_path_shown = True

                # Show plan if present and ask for HITL approval
                if response.plan:
                    self._display_plan(response.plan, round_num)

                    # HITL: Ask for plan approval on first plan
                    if not plan_approved and self.hitl_mode in (HITLMode.PLAN_ONLY, HITLMode.PLAN_AND_ANSWER, HITLMode.FULL):
                        plan_text = "\n".join(
                            f"{s.number}. {'[x]' if s.completed else '[ ]'} {s.description}"
                            for s in response.plan.steps
                        )
                        if not self._prompt_hitl_approval("Plan", plan_text):
                            self.console.print("[yellow]Execution stopped by user[/yellow]")
                            break
                        plan_approved = True

                # Display the response
                self._display_response(response, round_num)

            # Update session in context
            if agent.session:
                self.ctx.session = agent.session

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Execution interrupted[/yellow]")
        except Exception as e:
            self.renderer.render_error(f"Error: {str(e)}")

    def _display_plan(self, plan, round_num: int = 1) -> None:
        """Display the current plan state."""
        # Build plan display
        lines = []
        for step in plan.steps:
            marker = "[green]✓[/green]" if step.completed else "[yellow]○[/yellow]"
            lines.append(f"  {step.number}. {marker} {step.description}")

        completed = sum(1 for s in plan.steps if s.completed)
        total = len(plan.steps)

        self.console.print()
        self.console.print(
            Panel(
                "\n".join(lines),
                title=f"[cyan]Plan[/cyan] ({completed}/{total} steps)",
                border_style="cyan",
            )
        )

    def _display_response(self, response, round_num: int = 1) -> None:
        """Display a chat response with proper formatting."""
        # Show round indicator for multi-step execution
        if round_num > 1:
            self.console.print(f"\n[dim]─── Round {round_num} ───[/dim]")

        # Show thinking if present
        if response.thinking:
            self.renderer.render_thinking(response.thinking)

        # Show code if present
        if response.code:
            self.renderer.render_code(response.code, title="Code")

            # Show execution result
            if response.execution_result:
                result = response.execution_result
                if result.success:
                    self.renderer.render_output(
                        result.output or "(no output)",
                        success=True,
                        title="Output"
                    )
                else:
                    self.renderer.render_output(
                        result.output or result.error or "Unknown error",
                        success=False,
                        title="Error"
                    )

                # Mention images if generated
                if result.images:
                    self.console.print(
                        f"[dim]Generated {len(result.images)} image(s)[/dim]"
                    )

        # Show answer if present (final response)
        if response.has_answer and response.answer:
            self.console.print()
            self.renderer.render_answer(response.answer)
        elif not response.code and not response.plan:
            # Just show the text response (simple conversational reply)
            self.renderer.render_assistant_message(response.content)

    def run(self) -> None:
        """Run the main REPL loop."""
        self._print_welcome()

        # Load or create initial session
        if self.initial_session_id:
            session = self.manager.load_session(self.initial_session_id)
            if session:
                self.ctx.set_session(session)
                self.console.print(f"Resumed session: {session.name}")
            else:
                self.console.print(
                    f"[yellow]Session not found: {self.initial_session_id}[/yellow]"
                )

        running = True
        while running:
            try:
                self._print_status_bar()

                # Get user input
                user_input = self.prompt_session.prompt(
                    self._get_prompt(),
                ).strip()

                if not user_input:
                    continue

                # Handle commands vs chat
                if user_input.startswith("/"):
                    running = self._handle_command(user_input)
                else:
                    self._handle_chat(user_input)

                self.console.print()

            except KeyboardInterrupt:
                self.console.print("\n[dim]Use /quit to exit[/dim]")
                continue

            except EOFError:
                running = False

        # Cleanup
        self.manager.close()
        self.console.print("[dim]Session saved. Goodbye![/dim]")


def main():
    """Main entry point for the conversational CLI."""
    parser = argparse.ArgumentParser(
        description="DSAgent - Conversational Data Science Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dsagent-chat                      # Start interactive session
  dsagent-chat --model gpt-4o       # Use specific model
  dsagent-chat --session abc123     # Resume a session
  dsagent-chat --workspace ./out    # Use specific workspace
  dsagent-chat --hitl plan          # Require approval for plans
  dsagent-chat --hitl full          # Require approval for everything
  dsagent-chat --live-notebook      # Enable live notebook sync
  dsagent-chat --mcp-config ~/.dsagent/mcp.yaml  # Use MCP tools

HITL Modes:
  none        - No approval required (default)
  plan        - Approve plans before execution
  full        - Approve both plans and code
  plan_answer - Approve plans and final answers
  on_error    - Intervene only on errors

Live Notebook:
  --live-notebook   - Save notebook after each code execution (can open in Jupyter)
  --notebook-sync   - Bidirectional sync (also detects user edits in Jupyter)
        """,
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
        default=os.getenv("DSAGENT_WORKSPACE", "./workspace"),
        help="Workspace directory (default: ./workspace or $DSAGENT_WORKSPACE)",
    )

    parser.add_argument(
        "--session", "-s",
        type=str,
        default=None,
        help="Session ID to resume",
    )

    parser.add_argument(
        "--hitl",
        type=str,
        choices=["none", "plan", "full", "plan_answer", "on_error"],
        default="none",
        help="Human-in-the-loop mode (default: none)",
    )

    parser.add_argument(
        "--live-notebook",
        action="store_true",
        help="Enable live notebook (saves after each execution)",
    )

    parser.add_argument(
        "--notebook-sync",
        action="store_true",
        help="Enable bidirectional notebook sync with Jupyter",
    )

    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP servers YAML config file (e.g., ~/.dsagent/mcp.yaml)",
    )

    args = parser.parse_args()

    # Validate configuration
    try:
        from dsagent.utils.validation import validate_configuration
        validate_configuration(args.model)
    except Exception as e:
        console = Console()
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)

    # Convert HITL mode string to enum
    hitl_mode_map = {
        "none": HITLMode.NONE,
        "plan": HITLMode.PLAN_ONLY,
        "full": HITLMode.FULL,
        "plan_answer": HITLMode.PLAN_AND_ANSWER,
        "on_error": HITLMode.ON_ERROR,
    }
    hitl_mode = hitl_mode_map.get(args.hitl, HITLMode.NONE)

    # Validate MCP config if provided
    mcp_config_path = None
    if args.mcp_config:
        mcp_config_path = Path(args.mcp_config).expanduser().resolve()
        if not mcp_config_path.exists():
            console = Console()
            console.print(f"[red]Error: MCP config file not found: {mcp_config_path}[/red]")
            sys.exit(1)

    # Run the CLI
    cli = ConversationalCLI(
        workspace=Path(args.workspace),
        model=args.model,
        session_id=args.session,
        hitl_mode=hitl_mode,
        enable_live_notebook=args.live_notebook,
        enable_notebook_sync=args.notebook_sync,
        mcp_config=mcp_config_path,
    )

    try:
        cli.run()
    except Exception as e:
        console = Console()
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def run_chat(args) -> int:
    """Run chat with args from main CLI.

    Args:
        args: Namespace with model, workspace, session, hitl, etc.

    Returns:
        Exit code (0 for success)
    """
    from dsagent.utils.validation import validate_configuration

    console = Console()

    # Validate configuration
    try:
        validate_configuration(args.model)
    except Exception as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        return 1

    # Convert HITL mode string to enum
    hitl_mode_map = {
        "none": HITLMode.NONE,
        "plan": HITLMode.PLAN_ONLY,
        "full": HITLMode.FULL,
        "plan_answer": HITLMode.PLAN_AND_ANSWER,
        "on_error": HITLMode.ON_ERROR,
    }
    hitl_mode = hitl_mode_map.get(args.hitl, HITLMode.NONE)

    # Validate MCP config if provided
    mcp_config_path = None
    if args.mcp_config:
        mcp_config_path = Path(args.mcp_config).expanduser().resolve()
        if not mcp_config_path.exists():
            console.print(f"[red]Error: MCP config file not found: {mcp_config_path}[/red]")
            return 1

    # Run the CLI
    cli = ConversationalCLI(
        workspace=Path(args.workspace),
        model=args.model,
        session_id=getattr(args, 'session', None),
        hitl_mode=hitl_mode,
        enable_live_notebook=getattr(args, 'live_notebook', False),
        enable_notebook_sync=getattr(args, 'notebook_sync', False),
        mcp_config=mcp_config_path,
    )

    try:
        cli.run()
        return 0
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


if __name__ == "__main__":
    main()
