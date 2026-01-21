"""Command registry and slash commands for conversational CLI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

if TYPE_CHECKING:
    from dsagent.cli.repl import CLIContext


@dataclass
class CommandResult:
    """Result of a command execution."""

    success: bool = True
    message: Optional[str] = None
    data: Any = None
    should_exit: bool = False
    clear_screen: bool = False


class Command(ABC):
    """Base class for slash commands."""

    name: str = ""
    aliases: List[str] = []
    description: str = ""
    usage: str = ""

    @abstractmethod
    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        """Execute the command.

        Args:
            ctx: CLI context with session, manager, etc.
            args: Command arguments (split by whitespace)

        Returns:
            CommandResult indicating success/failure and any output
        """
        pass

    def get_help(self) -> str:
        """Get detailed help for this command."""
        help_text = f"/{self.name}"
        if self.aliases:
            help_text += f" (aliases: {', '.join('/' + a for a in self.aliases)})"
        help_text += f"\n  {self.description}"
        if self.usage:
            help_text += f"\n  Usage: {self.usage}"
        return help_text


class CommandRegistry:
    """Registry for slash commands."""

    def __init__(self):
        self._commands: Dict[str, Command] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, command: Command) -> None:
        """Register a command."""
        self._commands[command.name] = command
        for alias in command.aliases:
            self._aliases[alias] = command.name

    def get(self, name: str) -> Optional[Command]:
        """Get a command by name or alias."""
        # Check direct name
        if name in self._commands:
            return self._commands[name]
        # Check aliases
        if name in self._aliases:
            return self._commands[self._aliases[name]]
        return None

    def list_commands(self) -> List[Command]:
        """List all registered commands."""
        return list(self._commands.values())

    def get_completions(self, partial: str) -> List[str]:
        """Get command completions for partial input."""
        all_names = list(self._commands.keys()) + list(self._aliases.keys())
        return [f"/{n}" for n in all_names if n.startswith(partial)]


# =============================================================================
# Built-in Commands
# =============================================================================


class HelpCommand(Command):
    """Show help information."""

    name = "help"
    aliases = ["h", "?"]
    description = "Show help for commands"
    usage = "/help [command]"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        if args:
            # Help for specific command
            cmd = ctx.registry.get(args[0].lstrip("/"))
            if cmd:
                return CommandResult(message=cmd.get_help())
            return CommandResult(
                success=False,
                message=f"Unknown command: {args[0]}"
            )

        # General help
        lines = ["Available commands:\n"]
        for cmd in sorted(ctx.registry.list_commands(), key=lambda c: c.name):
            alias_str = f" ({', '.join('/' + a for a in cmd.aliases)})" if cmd.aliases else ""
            lines.append(f"  /{cmd.name}{alias_str} - {cmd.description}")

        lines.append("\nType /help <command> for detailed help")
        lines.append("Type any message to chat with the agent")

        return CommandResult(message="\n".join(lines))


class NewCommand(Command):
    """Start a new session."""

    name = "new"
    aliases = ["n"]
    description = "Start a new conversation session"
    usage = "/new [name]"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        name = " ".join(args) if args else None
        session = ctx.manager.create_session(name=name)
        ctx.set_session(session)

        msg = f"New session created: {session.name}"
        if session.workspace_path:
            msg += f"\nWorkspace: {session.workspace_path}"

        return CommandResult(message=msg)


class SessionsCommand(Command):
    """List available sessions."""

    name = "sessions"
    aliases = ["ls", "list"]
    description = "List all saved sessions"
    usage = "/sessions [--all]"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        from dsagent.session import SessionStatus

        include_archived = "--all" in args
        status_filter = None if include_archived else SessionStatus.ACTIVE

        sessions = ctx.manager.list_sessions(status=status_filter, limit=20)

        if not sessions:
            return CommandResult(message="No sessions found")

        lines = ["Sessions:\n"]
        for i, s in enumerate(sessions, 1):
            status_icon = {
                "active": "*",
                "paused": "||",
                "completed": "+",
                "archived": "-",
                "error": "!"
            }.get(s["status"], "?")

            current = " (current)" if ctx.session and s["id"] == ctx.session.id else ""
            # Show enough of the ID to be unique (format: YYYYMMDD_HHMMSS_XXXXXX)
            short_id = s['id'][:22] if len(s['id']) > 22 else s['id']
            lines.append(
                f"  {status_icon} {s['name'][:25]:<25} [{short_id}] "
                f"msgs:{s['message_count']}{current}"
            )

        lines.append(f"\nTotal: {len(sessions)} sessions")
        if not include_archived:
            lines.append("Use /sessions --all to include archived")

        return CommandResult(message="\n".join(lines))


class LoadCommand(Command):
    """Load a previous session."""

    name = "load"
    aliases = ["open", "resume"]
    description = "Load a saved session by ID"
    usage = "/load <session_id>"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        if not args:
            return CommandResult(
                success=False,
                message="Session ID required. Use /sessions to list available sessions."
            )

        session_id = args[0]

        # Try to find session by partial ID match
        sessions = ctx.manager.list_sessions(limit=100)
        matches = [s for s in sessions if s["id"].startswith(session_id)]

        if not matches:
            return CommandResult(
                success=False,
                message=f"No session found matching: {session_id}"
            )

        if len(matches) > 1:
            lines = ["Multiple sessions match:\n"]
            for s in matches[:5]:
                lines.append(f"  {s['id']} - {s['name']}")
            return CommandResult(
                success=False,
                message="\n".join(lines)
            )

        # Load the session
        session = ctx.manager.load_session(matches[0]["id"])
        if session:
            ctx.set_session(session)
            return CommandResult(
                message=f"Loaded session: {session.name}\n"
                f"Messages: {len(session.history)}"
            )

        return CommandResult(
            success=False,
            message=f"Failed to load session: {session_id}"
        )


class ContextCommand(Command):
    """Show current kernel context."""

    name = "context"
    aliases = ["ctx"]
    description = "Show current kernel state and variables"
    usage = "/context"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        if not ctx.session:
            return CommandResult(message="No active session")

        snapshot = ctx.session.kernel_snapshot
        if not snapshot:
            return CommandResult(message="Kernel state not available (no code executed yet)")

        return CommandResult(message=snapshot.get_context_summary())


class VarsCommand(Command):
    """List variables in kernel."""

    name = "vars"
    aliases = ["v", "variables"]
    description = "List all variables in the kernel"
    usage = "/vars"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        if not ctx.session or not ctx.session.kernel_snapshot:
            return CommandResult(message="No kernel state available")

        snapshot = ctx.session.kernel_snapshot
        if not snapshot.variables:
            return CommandResult(message="No variables defined")

        lines = ["Variables:\n"]
        for name, type_name in sorted(snapshot.variables.items()):
            if name in snapshot.dataframes:
                df_info = snapshot.dataframes[name]
                shape = df_info.get("shape", [0, 0])
                lines.append(f"  {name}: DataFrame ({shape[0]}x{shape[1]})")
            else:
                lines.append(f"  {name}: {type_name}")

        return CommandResult(message="\n".join(lines))


class HistoryCommand(Command):
    """Show conversation history."""

    name = "history"
    aliases = ["hist"]
    description = "Show conversation history"
    usage = "/history [n]"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        if not ctx.session:
            return CommandResult(message="No active session")

        n = 10
        if args:
            try:
                n = int(args[0])
            except ValueError:
                pass

        messages = ctx.session.history.get_recent(n)
        if not messages:
            return CommandResult(message="No messages in history")

        lines = [f"Last {len(messages)} messages:\n"]
        for msg in messages:
            role_icon = {
                "user": ">",
                "assistant": "<",
                "system": "#",
                "execution": "$"
            }.get(msg.role.value, "?")

            # Truncate long messages
            content = msg.content[:100]
            if len(msg.content) > 100:
                content += "..."

            lines.append(f"  {role_icon} [{msg.role.value}] {content}")

        return CommandResult(message="\n".join(lines))


class ExportCommand(Command):
    """Export session to notebook."""

    name = "export"
    aliases = ["save"]
    description = "Export current session to Jupyter notebook"
    usage = "/export [filename]"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        if not ctx.session:
            return CommandResult(
                success=False,
                message="No active session"
            )

        # Check if we have an agent with notebook data
        agent = ctx.agent
        if not agent:
            return CommandResult(
                success=False,
                message="No agent running. Execute some code first."
            )

        # Get filename from args or auto-generate
        filename = args[0] if args else None
        if filename and not filename.endswith('.ipynb'):
            filename += '.ipynb'

        # Export the notebook
        try:
            notebook_path = agent.export_notebook(filename)
            if notebook_path:
                return CommandResult(
                    message=f"Notebook exported to:\n{notebook_path}"
                )
            else:
                return CommandResult(
                    success=False,
                    message="No code executions to export. Run some analysis first."
                )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Export failed: {str(e)}"
            )


class ClearCommand(Command):
    """Clear the screen."""

    name = "clear"
    aliases = ["cls"]
    description = "Clear the terminal screen"
    usage = "/clear"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        return CommandResult(clear_screen=True)


class QuitCommand(Command):
    """Exit the CLI."""

    name = "quit"
    aliases = ["q", "exit", "bye"]
    description = "Exit the chat session"
    usage = "/quit"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        # Save session before exit
        if ctx.session:
            ctx.manager.save_session(ctx.session)

        return CommandResult(
            message="Goodbye!",
            should_exit=True
        )


class ModelCommand(Command):
    """Show or change the LLM model."""

    name = "model"
    aliases = ["m"]
    description = "Show or change the current LLM model"
    usage = "/model [model_name]"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        if not args:
            return CommandResult(message=f"Current model: {ctx.model}")

        new_model = args[0]
        ctx.model = new_model
        return CommandResult(message=f"Model changed to: {new_model}")


class StatusCommand(Command):
    """Show session status."""

    name = "status"
    aliases = ["st"]
    description = "Show current session status"
    usage = "/status"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        if not ctx.session:
            return CommandResult(message="No active session. Use /new to start one.")

        lines = [
            "Session Status:",
            f"  ID: {ctx.session.id}",
            f"  Name: {ctx.session.name}",
            f"  Status: {ctx.session.status.value}",
            f"  Messages: {len(ctx.session.history)}",
            f"  Model: {ctx.model}",
        ]

        if ctx.session.workspace_path:
            lines.append(f"  Workspace: {ctx.session.workspace_path}")

        if ctx.session.kernel_snapshot:
            lines.append(f"  Kernel: Active ({len(ctx.session.kernel_snapshot.variables)} vars)")
        else:
            lines.append("  Kernel: Not initialized")

        return CommandResult(message="\n".join(lines))


class DataCommand(Command):
    """Load data into the session workspace."""

    name = "data"
    aliases = ["d"]
    description = "Copy data file to workspace or list files"
    usage = "/data [path]"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        import shutil
        from pathlib import Path

        # Create a session if one doesn't exist (ensures data goes to correct location)
        session_created = False
        if not ctx.session:
            session = ctx.manager.create_session()
            ctx.set_session(session)
            session_created = True

        # Use session's data directory
        data_dir = Path(ctx.session.data_path)
        data_dir.mkdir(parents=True, exist_ok=True)

        # If no args, list files in data directory
        if not args:
            files = list(data_dir.glob("*"))
            if not files:
                return CommandResult(
                    message=f"No files in data/\n"
                    f"Usage: /data <path> to copy a file to workspace"
                )

            file_list = "\n".join(f"  - {f.name}" for f in files if f.is_file())
            return CommandResult(
                message=f"Files in data/:\n{file_list}"
            )

        # Copy file to data directory
        source = Path(args[0]).expanduser().resolve()

        if not source.exists():
            return CommandResult(
                success=False,
                message=f"Path not found: {source}"
            )

        if source.is_file():
            dest = data_dir / source.name
            shutil.copy2(source, dest)
            ctx.data_path = str(dest)
            session_msg = "New session created.\n" if session_created else ""
            return CommandResult(
                message=f"{session_msg}Copied: {source.name} → data/\n"
                f"Use 'data/{source.name}' in your code."
            )
        elif source.is_dir():
            # Copy all files from directory
            copied = []
            for item in source.iterdir():
                if item.is_file():
                    shutil.copy2(item, data_dir / item.name)
                    copied.append(item.name)

            if copied:
                ctx.data_path = str(data_dir)
                session_msg = "New session created.\n" if session_created else ""
                return CommandResult(
                    message=f"{session_msg}Copied {len(copied)} files from {source.name}/:\n"
                    + "\n".join(f"  - {f}" for f in copied[:10])
                    + (f"\n  ... and {len(copied) - 10} more" if len(copied) > 10 else "")
                )
            else:
                return CommandResult(
                    success=False,
                    message=f"No files found in {source}"
                )

        return CommandResult(success=False, message="Unknown error")


class LogsCommand(Command):
    """Show logs information and paths."""

    name = "logs"
    aliases = ["log"]
    description = "Show logs path and recent events"
    usage = "/logs [tail N]"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        from pathlib import Path

        if not ctx.session:
            return CommandResult(message="No active session")

        if not ctx.session.logs_path:
            return CommandResult(message="Logs path not configured for this session")

        logs_path = Path(ctx.session.logs_path)

        if not logs_path.exists():
            return CommandResult(
                message=f"Logs directory: {logs_path}\n(not created yet - will be created on first chat)"
            )

        lines = [
            f"Logs directory: {logs_path}",
            "",
        ]

        # Check for log files
        run_log = logs_path / "run.log"
        events_log = logs_path / "events.jsonl"

        if run_log.exists():
            size = run_log.stat().st_size
            lines.append(f"  run.log: {size:,} bytes")
        else:
            lines.append("  run.log: (not created)")

        if events_log.exists():
            size = events_log.stat().st_size
            # Count events
            with open(events_log, "r") as f:
                event_count = sum(1 for _ in f)
            lines.append(f"  events.jsonl: {event_count} events ({size:,} bytes)")
        else:
            lines.append("  events.jsonl: (not created)")

        # Show tail of run.log if requested
        if args and args[0] == "tail":
            n = 20
            if len(args) > 1:
                try:
                    n = int(args[1])
                except ValueError:
                    pass

            if run_log.exists():
                lines.append("")
                lines.append(f"Last {n} lines of run.log:")
                lines.append("─" * 50)
                with open(run_log, "r") as f:
                    all_lines = f.readlines()
                    for line in all_lines[-n:]:
                        lines.append(line.rstrip())

        lines.append("")
        lines.append("Use '/logs tail [N]' to see recent log entries")

        return CommandResult(message="\n".join(lines))


class SummaryCommand(Command):
    """Show conversation summary."""

    name = "summary"
    aliases = ["sum"]
    description = "Show the conversation summary (if any)"
    usage = "/summary"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        if not ctx.session:
            return CommandResult(message="No active session")

        history = ctx.session.history

        if not history.summary:
            return CommandResult(
                message="No summary yet.\n"
                f"Current messages: {len(history.messages)}\n"
                f"Summarization threshold: 30 messages\n"
                "Summary will be created automatically when threshold is exceeded."
            )

        lines = [
            "═" * 60,
            "CONVERSATION SUMMARY",
            "═" * 60,
            "",
            history.summary,
            "",
            "─" * 60,
            f"Messages summarized: {history.summary_messages_count}",
            f"Total truncated: {history.truncated_count}",
            f"Current messages: {len(history.messages)}",
        ]

        if history.summary_created_at:
            lines.append(f"Created at: {history.summary_created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        lines.append("─" * 60)

        return CommandResult(message="\n".join(lines))


class WorkspaceCommand(Command):
    """Show workspace information."""

    name = "workspace"
    aliases = ["ws"]
    description = "Show workspace directory info"
    usage = "/workspace"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        from pathlib import Path

        # Use session's workspace if session exists, otherwise fallback to ctx.workspace
        if ctx.session and ctx.session.workspace_path:
            workspace = Path(ctx.session.workspace_path)
            data_dir = Path(ctx.session.data_path) if ctx.session.data_path else workspace / "data"
            artifacts_dir = Path(ctx.session.artifacts_path) if ctx.session.artifacts_path else workspace / "artifacts"
        else:
            workspace = ctx.workspace
            data_dir = workspace / "data"
            artifacts_dir = workspace / "artifacts"

        lines = [f"Workspace: {workspace}"]

        # Count files in data/
        if data_dir.exists():
            data_files = list(data_dir.glob("*"))
            data_count = len([f for f in data_files if f.is_file()])
            lines.append(f"  data/: {data_count} file(s)")
        else:
            lines.append("  data/: (not created)")

        # Count files in artifacts/
        if artifacts_dir.exists():
            artifact_files = list(artifacts_dir.glob("*"))
            artifact_count = len([f for f in artifact_files if f.is_file()])
            lines.append(f"  artifacts/: {artifact_count} file(s)")
        else:
            lines.append("  artifacts/: (not created)")

        return CommandResult(message="\n".join(lines))


class SkillsCommand(Command):
    """List installed skills."""

    name = "skills"
    aliases = ["sk"]
    description = "List installed agent skills"
    usage = "/skills"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        from dsagent.skills import SkillLoader, SkillRegistry

        try:
            loader = SkillLoader()
            registry = SkillRegistry(loader)
            count = registry.discover()
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error loading skills: {e}"
            )

        if count == 0:
            return CommandResult(
                message="No skills installed.\n\n"
                "Install skills with:\n"
                "  dsagent skills install github:dsagent-skills/eda-analysis\n\n"
                "Or see: /help skill"
            )

        lines = ["Installed Skills:\n"]
        for name, meta in registry.skills.items():
            skill = registry.get_skill(name)
            script_count = len(skill.scripts) if skill else 0
            desc = meta.description[:45] + "..." if len(meta.description) > 45 else meta.description
            lines.append(f"  {name} (v{meta.version}) - {desc}")
            if script_count:
                lines.append(f"    Scripts: {script_count}")

        lines.append(f"\nSkills directory: {loader.skills_dir}")
        lines.append("Use /skill <name> for details")

        return CommandResult(message="\n".join(lines))


class SkillInfoCommand(Command):
    """Show details about a skill."""

    name = "skill"
    aliases = []
    description = "Show details about an installed skill"
    usage = "/skill <name>"

    def execute(self, ctx: "CLIContext", args: List[str]) -> CommandResult:
        from dsagent.skills import SkillLoader

        if not args:
            return CommandResult(
                success=False,
                message="Skill name required. Use /skills to list installed skills."
            )

        name = args[0]
        loader = SkillLoader()

        if not loader.skill_exists(name):
            return CommandResult(
                success=False,
                message=f"Skill not found: {name}\n"
                "Use /skills to see installed skills."
            )

        try:
            skill = loader.load_skill(name)
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error loading skill: {e}"
            )

        lines = [
            f"{skill.metadata.name} (v{skill.metadata.version})",
            "",
        ]

        if skill.metadata.description:
            lines.append(skill.metadata.description)
            lines.append("")

        if skill.metadata.author:
            lines.append(f"Author: {skill.metadata.author}")

        if skill.metadata.tags:
            lines.append(f"Tags: {', '.join(skill.metadata.tags)}")

        if skill.metadata.compatibility.python:
            lines.append(f"Dependencies: {', '.join(skill.metadata.compatibility.python)}")

        if skill.scripts:
            lines.append("")
            lines.append("Scripts:")
            for script in skill.scripts:
                desc = f" - {script.description}" if script.description else ""
                lines.append(f"  - {script.name}{desc}")

        lines.append("")
        lines.append(f"Path: {skill.path}")

        return CommandResult(message="\n".join(lines))


def create_default_registry() -> CommandRegistry:
    """Create a registry with all default commands."""
    registry = CommandRegistry()

    # Register all built-in commands
    commands = [
        HelpCommand(),
        NewCommand(),
        SessionsCommand(),
        LoadCommand(),
        ContextCommand(),
        VarsCommand(),
        HistoryCommand(),
        SummaryCommand(),
        LogsCommand(),
        ExportCommand(),
        ClearCommand(),
        QuitCommand(),
        ModelCommand(),
        StatusCommand(),
        DataCommand(),
        WorkspaceCommand(),
        SkillsCommand(),
        SkillInfoCommand(),
    ]

    for cmd in commands:
        registry.register(cmd)

    return registry
