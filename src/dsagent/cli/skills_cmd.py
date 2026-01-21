"""CLI commands for skill management."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def run_skills(args: argparse.Namespace) -> int:
    """Handle skills subcommands."""
    action = getattr(args, "skills_action", None)

    if action == "list":
        return _cmd_list(args)
    elif action == "install":
        return _cmd_install(args)
    elif action == "remove":
        return _cmd_remove(args)
    elif action == "info":
        return _cmd_info(args)
    else:
        console.print("[yellow]Usage: dsagent skills <command>[/yellow]")
        console.print()
        console.print("Commands:")
        console.print("  list              List installed skills")
        console.print("  install <source>  Install a skill")
        console.print("  remove <name>     Remove a skill")
        console.print("  info <name>       Show skill details")
        console.print()
        console.print("Examples:")
        console.print("  dsagent skills list")
        console.print("  dsagent skills install github:dsagent-skills/eda-analysis")
        console.print("  dsagent skills install ./my-local-skill")
        console.print("  dsagent skills remove eda-analysis")
        return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """List installed skills."""
    from dsagent.skills import SkillLoader, SkillRegistry

    loader = SkillLoader()
    registry = SkillRegistry(loader)
    count = registry.discover()

    if count == 0:
        console.print("[dim]No skills installed.[/dim]")
        console.print()
        console.print("Install skills with:")
        console.print("  dsagent skills install github:dsagent-skills/eda-analysis")
        return 0

    table = Table(title="Installed Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Description")
    table.add_column("Scripts", style="yellow")

    for name, meta in registry.skills.items():
        # Get script count
        skill = registry.get_skill(name)
        script_count = len(skill.scripts) if skill else 0

        table.add_row(
            name,
            meta.version,
            meta.description[:50] + "..." if len(meta.description) > 50 else meta.description,
            str(script_count),
        )

    console.print(table)
    console.print(f"\n[dim]Skills directory: {loader.skills_dir}[/dim]")
    return 0


def _cmd_install(args: argparse.Namespace) -> int:
    """Install a skill."""
    from dsagent.skills import SkillInstaller
    from dsagent.skills.installer import (
        SkillExistsError,
        SkillInstallError,
        SkillValidationError,
    )

    source = args.source
    force = getattr(args, "force", False)

    console.print(f"[cyan]Installing skill from:[/cyan] {source}")

    installer = SkillInstaller()

    try:
        result = installer.install(source, force=force)

        if result.success:
            console.print()
            console.print(f"[green]Successfully installed:[/green] {result.skill_name}")
            console.print()

            if result.metadata:
                console.print(f"  Description: {result.metadata.description}")
                console.print(f"  Version: {result.metadata.version}")

                if result.metadata.compatibility.python:
                    deps = ", ".join(result.metadata.compatibility.python)
                    console.print(f"  Dependencies: {deps}")

            if result.path:
                # Count scripts
                skill_dir = result.path
                scripts_dir = skill_dir / "scripts"
                scripts = list(scripts_dir.glob("*.py")) if scripts_dir.exists() else []
                scripts += [f for f in skill_dir.glob("*.py") if not f.name.startswith("_")]
                if scripts:
                    console.print(f"  Scripts: {len(scripts)}")

            console.print()
            console.print("[dim]The agent will automatically use this skill when relevant.[/dim]")
            return 0
        else:
            console.print(f"[red]Installation failed:[/red] {result.message}")
            return 1

    except SkillExistsError as e:
        console.print(f"[yellow]{e}[/yellow]")
        return 1
    except SkillValidationError as e:
        console.print(f"[red]Validation failed:[/red] {e}")
        return 1
    except SkillInstallError as e:
        console.print(f"[red]Installation error:[/red] {e}")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        return 1


def _cmd_remove(args: argparse.Namespace) -> int:
    """Remove a skill."""
    from dsagent.skills import SkillInstaller

    name = args.name
    installer = SkillInstaller()

    console.print(f"[cyan]Removing skill:[/cyan] {name}")

    if installer.uninstall(name):
        console.print(f"[green]Successfully removed:[/green] {name}")
        return 0
    else:
        console.print(f"[yellow]Skill not found:[/yellow] {name}")
        return 1


def _cmd_info(args: argparse.Namespace) -> int:
    """Show skill details."""
    from dsagent.skills import SkillLoader

    name = args.name
    loader = SkillLoader()

    if not loader.skill_exists(name):
        console.print(f"[red]Skill not found:[/red] {name}")
        console.print()
        console.print("[dim]Use 'dsagent skills list' to see installed skills.[/dim]")
        return 1

    try:
        skill = loader.load_skill(name)
    except Exception as e:
        console.print(f"[red]Error loading skill:[/red] {e}")
        return 1

    # Header
    console.print()
    console.print(f"[bold cyan]{skill.metadata.name}[/bold cyan]")
    console.print(f"[dim]Version: {skill.metadata.version}[/dim]")
    console.print()

    # Description
    if skill.metadata.description:
        console.print(f"[bold]Description:[/bold]")
        console.print(f"  {skill.metadata.description}")
        console.print()

    # Author
    if skill.metadata.author:
        console.print(f"[bold]Author:[/bold] {skill.metadata.author}")
        console.print()

    # Tags
    if skill.metadata.tags:
        console.print(f"[bold]Tags:[/bold] {', '.join(skill.metadata.tags)}")
        console.print()

    # Dependencies
    if skill.metadata.compatibility.python:
        console.print(f"[bold]Python Dependencies:[/bold]")
        for dep in skill.metadata.compatibility.python:
            console.print(f"  - {dep}")
        console.print()

    # Scripts
    if skill.scripts:
        console.print(f"[bold]Scripts:[/bold]")
        for script in skill.scripts:
            desc = f" - {script.description}" if script.description else ""
            console.print(f"  - [yellow]{script.name}[/yellow]{desc}")
            console.print(f"    [dim]{script.path}[/dim]")
        console.print()

    # Location
    console.print(f"[bold]Location:[/bold]")
    console.print(f"  {skill.path}")
    console.print()

    # Instructions preview
    console.print(f"[bold]Instructions:[/bold]")
    # Show first 500 chars of instructions
    instructions = skill.instructions
    if len(instructions) > 500:
        instructions = instructions[:500] + "..."
    console.print(f"  {instructions}")
    console.print()

    return 0
