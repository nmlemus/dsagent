"""Rich rendering utilities for CLI output."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Pattern to match internal tags that shouldn't be shown to users
INTERNAL_TAGS_PATTERN = re.compile(
    r"<(intent|think)>.*?</\1>\s*",
    re.DOTALL | re.IGNORECASE
)


class CLIRenderer:
    """Renderer for CLI output using Rich."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def render_code(
        self,
        code: str,
        language: str = "python",
        title: Optional[str] = None,
        line_numbers: bool = True,
    ) -> None:
        """Render code with syntax highlighting."""
        syntax = Syntax(
            code,
            language,
            theme="monokai",
            line_numbers=line_numbers,
            word_wrap=True,
        )

        if title:
            self.console.print(Panel(syntax, title=title, border_style="cyan"))
        else:
            self.console.print(syntax)

    def render_output(
        self,
        output: str,
        success: bool = True,
        title: Optional[str] = None,
    ) -> None:
        """Render execution output."""
        style = "green" if success else "red"
        title = title or ("Output" if success else "Error")

        self.console.print(
            Panel(
                Text(output, style="white" if success else "red"),
                title=f"[{style}]{title}[/{style}]",
                border_style=style,
            )
        )

    def render_dataframe_info(
        self,
        name: str,
        shape: tuple,
        columns: List[str],
        dtypes: Optional[Dict[str, str]] = None,
        sample_data: Optional[List[List[Any]]] = None,
    ) -> None:
        """Render DataFrame information."""
        table = Table(
            title=f"DataFrame: {name}",
            show_header=True,
            header_style="bold cyan",
        )

        # Add columns
        for col in columns[:10]:  # Limit to 10 columns
            dtype_str = f" ({dtypes.get(col, 'unknown')})" if dtypes else ""
            table.add_column(f"{col}{dtype_str}")

        if len(columns) > 10:
            table.add_column(f"... +{len(columns) - 10} more")

        # Add sample data if provided
        if sample_data:
            for row in sample_data[:5]:  # Limit to 5 rows
                str_row = [str(v)[:20] for v in row[:10]]
                if len(row) > 10:
                    str_row.append("...")
                table.add_row(*str_row)

        self.console.print(table)
        self.console.print(f"[dim]Shape: {shape[0]} rows x {shape[1]} columns[/dim]")

    def render_variables(
        self,
        variables: Dict[str, str],
        dataframes: Optional[Dict[str, Dict]] = None,
    ) -> None:
        """Render variable list."""
        if not variables:
            self.console.print("[dim]No variables defined[/dim]")
            return

        tree = Tree("[bold]Kernel Variables[/bold]")

        # Group by type
        dataframe_vars = []
        other_vars = []

        for name, type_name in variables.items():
            if dataframes and name in dataframes:
                dataframe_vars.append((name, dataframes[name]))
            else:
                other_vars.append((name, type_name))

        # DataFrames section
        if dataframe_vars:
            df_branch = tree.add("[cyan]DataFrames[/cyan]")
            for name, info in dataframe_vars:
                shape = info.get("shape", [0, 0])
                cols = info.get("columns", [])[:3]
                cols_str = ", ".join(cols)
                if len(info.get("columns", [])) > 3:
                    cols_str += "..."
                df_branch.add(
                    f"[green]{name}[/green]: {shape[0]}x{shape[1]} [{cols_str}]"
                )

        # Other variables section
        if other_vars:
            var_branch = tree.add("[yellow]Variables[/yellow]")
            for name, type_name in other_vars:
                var_branch.add(f"[white]{name}[/white]: {type_name}")

        self.console.print(tree)

    def render_plan(
        self,
        steps: List[Dict[str, Any]],
        current_step: Optional[int] = None,
    ) -> None:
        """Render execution plan."""
        table = Table(
            title="Execution Plan",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("#", style="dim", width=3)
        table.add_column("Status", width=6)
        table.add_column("Step")

        for step in steps:
            num = step.get("number", 0)
            completed = step.get("completed", False)
            description = step.get("description", "")

            if completed:
                status = "[green]Done[/green]"
            elif current_step and num == current_step:
                status = "[yellow]>>>[/yellow]"
            else:
                status = "[dim]...[/dim]"

            table.add_row(str(num), status, description)

        self.console.print(table)

    def render_session_list(
        self,
        sessions: List[Dict[str, Any]],
        current_id: Optional[str] = None,
    ) -> None:
        """Render session list."""
        if not sessions:
            self.console.print("[dim]No sessions found[/dim]")
            return

        table = Table(
            title="Sessions",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("", width=2)
        table.add_column("Name", min_width=20)
        table.add_column("ID", width=10)
        table.add_column("Messages", width=8)
        table.add_column("Updated", width=12)
        table.add_column("Status", width=10)

        for s in sessions:
            is_current = current_id and s["id"] == current_id
            marker = "[cyan]*[/cyan]" if is_current else ""

            status_colors = {
                "active": "green",
                "paused": "yellow",
                "completed": "blue",
                "archived": "dim",
                "error": "red",
            }
            status_color = status_colors.get(s["status"], "white")

            # Parse and format date
            updated = s.get("updated_at", "")[:10]

            table.add_row(
                marker,
                s.get("name", "Unnamed")[:25],
                s["id"][:8],
                str(s.get("message_count", 0)),
                updated,
                f"[{status_color}]{s['status']}[/{status_color}]",
            )

        self.console.print(table)

    def render_thinking(self, content: str) -> None:
        """Render agent thinking/reasoning."""
        self.console.print(
            Panel(
                Text(content, style="italic dim"),
                title="[dim]Thinking[/dim]",
                border_style="dim",
            )
        )

    def render_assistant_message(
        self,
        content: str,
        code: Optional[str] = None,
    ) -> None:
        """Render assistant response."""
        # Strip internal tags (intent, think) that shouldn't be shown to users
        display_content = INTERNAL_TAGS_PATTERN.sub("", content).strip()

        # Skip if nothing to display after stripping
        if not display_content:
            return

        # Always use Markdown rendering for assistant messages
        # Rich's Markdown handles headers, bold, italic, lists, code, etc.
        self.console.print(
            Panel(
                Markdown(display_content),
                title="[cyan]Assistant[/cyan]",
                border_style="cyan",
            )
        )

        if code:
            self.render_code(code, title="Code to execute")

    def render_user_message(self, content: str) -> None:
        """Render user message (for history display)."""
        self.console.print(
            Panel(
                Text(content),
                title="[green]You[/green]",
                border_style="green",
            )
        )

    def render_execution_result(
        self,
        code: str,
        output: str,
        success: bool,
        execution_time: Optional[float] = None,
        images: Optional[List[Dict]] = None,
    ) -> None:
        """Render code execution result."""
        # Show code
        self.render_code(code, title="Executed")

        # Show output
        title = "Output"
        if execution_time:
            title += f" ({execution_time:.2f}s)"

        self.render_output(output, success=success, title=title)

        # Show images indicator
        if images:
            self.console.print(
                f"[dim]Generated {len(images)} image(s) - see artifacts folder[/dim]"
            )

    def render_error(self, message: str, title: str = "Error") -> None:
        """Render error message."""
        self.console.print(
            Panel(
                Text(message, style="red"),
                title=f"[red]{title}[/red]",
                border_style="red",
            )
        )

    def render_success(self, message: str, title: str = "Success") -> None:
        """Render success message."""
        self.console.print(
            Panel(
                Text(message, style="green"),
                title=f"[green]{title}[/green]",
                border_style="green",
            )
        )

    def render_info(self, message: str, title: str = "Info") -> None:
        """Render info message."""
        self.console.print(
            Panel(
                Text(message),
                title=f"[cyan]{title}[/cyan]",
                border_style="cyan",
            )
        )

    def render_answer(self, content: str, title: str = "Final Answer") -> None:
        """Render final answer with Markdown formatting."""
        # Strip internal tags
        display_content = INTERNAL_TAGS_PATTERN.sub("", content).strip()

        if not display_content:
            return

        self.console.print(
            Panel(
                Markdown(display_content),
                title=f"[bold green]{title}[/bold green]",
                border_style="green",
            )
        )

    def render_warning(self, message: str, title: str = "Warning") -> None:
        """Render warning message."""
        self.console.print(
            Panel(
                Text(message, style="yellow"),
                title=f"[yellow]{title}[/yellow]",
                border_style="yellow",
            )
        )

    def render_tool_calling(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> None:
        """Render tool calling indicator.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments (should be pre-sanitized)
        """
        # Format arguments for display
        if arguments:
            args_preview = ", ".join(
                f"{k}={repr(v)[:50]}" for k, v in list(arguments.items())[:3]
            )
            if len(arguments) > 3:
                args_preview += ", ..."
        else:
            args_preview = ""

        content = f"Calling: [bold]{tool_name}[/bold]({args_preview})"

        self.console.print(
            Panel(
                Text.from_markup(content),
                title="[cyan]Tool[/cyan]",
                border_style="cyan",
            )
        )

    def render_tool_result(
        self,
        tool_name: str,
        success: bool,
        result: Optional[str] = None,
        error: Optional[str] = None,
        execution_time: Optional[float] = None,
    ) -> None:
        """Render tool execution result.

        Args:
            tool_name: Name of the tool that was called
            success: Whether the tool execution succeeded
            result: Result from the tool (if success)
            error: Error message (if failed)
            execution_time: Execution time in milliseconds
        """
        if success:
            style = "green"
            title = f"[green]Tool Result: {tool_name}[/green]"
            # Truncate long results
            content = result or "(no output)"
            if len(content) > 500:
                content = content[:500] + f"\n... (truncated, {len(content)} chars)"
        else:
            style = "red"
            title = f"[red]Tool Failed: {tool_name}[/red]"
            content = error or "Unknown error"

        # Add execution time if available
        if execution_time is not None:
            title += f" ({execution_time:.0f}ms)"

        self.console.print(
            Panel(
                Text(content, style="white" if success else "red"),
                title=title,
                border_style=style,
            )
        )
