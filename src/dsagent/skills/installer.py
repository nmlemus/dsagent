"""Skill installer - installs skills from various sources."""

from __future__ import annotations

import io
import logging
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yaml

from dsagent.skills.loader import SkillLoader, SkillParseError
from dsagent.skills.models import (
    InstallResult,
    InstalledSkill,
    SkillMetadata,
    SkillsConfig,
)

logger = logging.getLogger(__name__)


class SkillInstallError(Exception):
    """Error during skill installation."""

    pass


class SkillExistsError(SkillInstallError):
    """Skill already exists."""

    pass


class SkillValidationError(SkillInstallError):
    """Skill validation failed."""

    pass


@dataclass
class SourceInfo:
    """Parsed source information."""

    type: str  # "github", "local", "url"
    owner: str = ""
    repo: str = ""
    path: str = ""
    branch: str = "main"
    local_path: Optional[Path] = None
    url: str = ""


class SkillInstaller:
    """Installs skills from various sources.

    Supports:
    - GitHub repositories: github:owner/repo or github:owner/repo/path/to/skill
    - Local directories: /path/to/skill or ./relative/path
    - URLs: https://github.com/owner/repo/tree/main/path/to/skill
    """

    def __init__(
        self,
        skills_dir: Optional[Path] = None,
        loader: Optional[SkillLoader] = None,
    ):
        """Initialize the installer.

        Args:
            skills_dir: Directory to install skills. Defaults to ~/.dsagent/skills/
            loader: SkillLoader for validation. Creates default if not provided.
        """
        self.skills_dir = skills_dir or (Path.home() / ".dsagent" / "skills")
        self.loader = loader or SkillLoader(self.skills_dir)
        self.config_path = self.skills_dir.parent / "skills.yaml"

    def install(
        self,
        source: str,
        force: bool = False,
    ) -> InstallResult:
        """Install a skill from a source.

        Args:
            source: Source string (github:owner/repo, local path, or URL)
            force: If True, overwrite existing skill

        Returns:
            InstallResult with installation details

        Raises:
            SkillInstallError: If installation fails
        """
        # Parse source
        source_info = self._parse_source(source)
        logger.info(f"Installing skill from {source_info.type}: {source}")

        # Download/copy to temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="dsagent-skill-"))
        try:
            skill_path = self._fetch_skill(source_info, temp_dir)

            # Validate skill
            try:
                skill = self.loader.load_skill_from_path(skill_path)
            except SkillParseError as e:
                raise SkillValidationError(f"Invalid skill: {e}")

            # Check for existing skill
            dest_path = self.skills_dir / skill.metadata.name
            if dest_path.exists():
                if not force:
                    raise SkillExistsError(
                        f"Skill '{skill.metadata.name}' already exists. "
                        f"Use --force to overwrite."
                    )
                shutil.rmtree(dest_path)

            # Create skills directory if needed
            self.skills_dir.mkdir(parents=True, exist_ok=True)

            # Move to final destination
            shutil.move(str(skill_path), str(dest_path))

            # Register in config
            self._register_skill(skill.metadata, source, dest_path)

            return InstallResult(
                success=True,
                skill_name=skill.metadata.name,
                message=f"Skill '{skill.metadata.name}' installed successfully",
                metadata=skill.metadata,
                path=dest_path,
            )

        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def uninstall(self, name: str) -> bool:
        """Uninstall a skill.

        Args:
            name: Skill name

        Returns:
            True if skill was uninstalled, False if not found
        """
        skill_path = self.skills_dir / name

        if not skill_path.exists():
            logger.warning(f"Skill '{name}' not found")
            return False

        # Remove directory
        shutil.rmtree(skill_path)

        # Remove from config
        self._unregister_skill(name)

        logger.info(f"Skill '{name}' uninstalled")
        return True

    def _parse_source(self, source: str) -> SourceInfo:
        """Parse a source string into SourceInfo.

        Handles:
        - github:owner/repo
        - github:owner/repo/path/to/skill
        - https://github.com/owner/repo/tree/branch/path
        - /absolute/path
        - ./relative/path

        Args:
            source: Source string

        Returns:
            SourceInfo with parsed components
        """
        # GitHub shorthand
        if source.startswith("github:"):
            return self._parse_github_shorthand(source[7:])

        # URL
        if source.startswith("https://") or source.startswith("http://"):
            return self._parse_url(source)

        # Local path
        path = Path(source).expanduser().resolve()
        return SourceInfo(type="local", local_path=path)

    def _parse_github_shorthand(self, ref: str) -> SourceInfo:
        """Parse GitHub shorthand: owner/repo or owner/repo/path/to/skill.

        Args:
            ref: GitHub reference (without github: prefix)

        Returns:
            SourceInfo
        """
        parts = ref.split("/")

        if len(parts) < 2:
            raise SkillInstallError(
                f"Invalid GitHub reference: {ref}. "
                f"Expected format: owner/repo or owner/repo/path/to/skill"
            )

        owner = parts[0]
        repo = parts[1]
        path = "/".join(parts[2:]) if len(parts) > 2 else ""

        return SourceInfo(
            type="github",
            owner=owner,
            repo=repo,
            path=path,
            branch="main",
        )

    def _parse_url(self, url: str) -> SourceInfo:
        """Parse a URL into SourceInfo.

        Handles GitHub URLs like:
        https://github.com/owner/repo/tree/branch/path/to/skill

        Args:
            url: URL string

        Returns:
            SourceInfo
        """
        parsed = urlparse(url)

        if "github.com" in parsed.netloc:
            # Parse GitHub URL
            path_parts = parsed.path.strip("/").split("/")

            if len(path_parts) < 2:
                raise SkillInstallError(f"Invalid GitHub URL: {url}")

            owner = path_parts[0]
            repo = path_parts[1]

            # Handle /tree/branch/path format
            branch = "main"
            skill_path = ""

            if len(path_parts) > 3 and path_parts[2] == "tree":
                branch = path_parts[3]
                skill_path = "/".join(path_parts[4:])
            elif len(path_parts) > 2:
                skill_path = "/".join(path_parts[2:])

            return SourceInfo(
                type="github",
                owner=owner,
                repo=repo,
                path=skill_path,
                branch=branch,
                url=url,
            )

        # Generic URL - not yet supported
        raise SkillInstallError(
            f"URL source not yet supported: {url}. "
            f"Use github: shorthand instead."
        )

    def _fetch_skill(self, source: SourceInfo, temp_dir: Path) -> Path:
        """Fetch skill from source to temp directory.

        Args:
            source: Parsed source info
            temp_dir: Temporary directory

        Returns:
            Path to skill directory in temp_dir
        """
        if source.type == "local":
            return self._fetch_local(source, temp_dir)
        elif source.type == "github":
            return self._fetch_github(source, temp_dir)
        else:
            raise SkillInstallError(f"Unknown source type: {source.type}")

    def _fetch_local(self, source: SourceInfo, temp_dir: Path) -> Path:
        """Copy skill from local path.

        Args:
            source: Source info with local_path
            temp_dir: Temporary directory

        Returns:
            Path to copied skill
        """
        if not source.local_path or not source.local_path.exists():
            raise SkillInstallError(f"Local path not found: {source.local_path}")

        skill_file = source.local_path / "SKILL.md"
        if not skill_file.exists():
            raise SkillInstallError(
                f"No SKILL.md found in {source.local_path}. "
                f"This doesn't appear to be a valid skill."
            )

        # Copy to temp
        dest = temp_dir / source.local_path.name
        shutil.copytree(source.local_path, dest)

        return dest

    def _fetch_github(self, source: SourceInfo, temp_dir: Path) -> Path:
        """Download skill from GitHub.

        Uses GitHub's zipball API to download only the needed files.

        Args:
            source: Source info with owner, repo, path, branch
            temp_dir: Temporary directory

        Returns:
            Path to extracted skill
        """
        try:
            import urllib.request
        except ImportError:
            raise SkillInstallError("urllib.request required for GitHub downloads")

        # Download zipball of the repo
        zip_url = f"https://github.com/{source.owner}/{source.repo}/archive/refs/heads/{source.branch}.zip"
        logger.info(f"Downloading from {zip_url}")

        try:
            with urllib.request.urlopen(zip_url, timeout=60) as response:
                zip_data = response.read()
        except Exception as e:
            raise SkillInstallError(f"Failed to download from GitHub: {e}")

        # Extract zip
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            zf.extractall(extract_dir)

        # Find the extracted repo directory (usually repo-branch/)
        extracted_dirs = list(extract_dir.iterdir())
        if not extracted_dirs:
            raise SkillInstallError("Empty zip file from GitHub")

        repo_dir = extracted_dirs[0]

        # Navigate to skill path if specified
        if source.path:
            skill_dir = repo_dir / source.path
            if not skill_dir.exists():
                raise SkillInstallError(
                    f"Path '{source.path}' not found in repository"
                )
        else:
            skill_dir = repo_dir

        # Verify SKILL.md exists
        if not (skill_dir / "SKILL.md").exists():
            raise SkillInstallError(
                f"No SKILL.md found at {source.path or 'repository root'}. "
                f"This doesn't appear to be a valid skill."
            )

        # Move skill to temp_dir root
        final_path = temp_dir / skill_dir.name
        shutil.move(str(skill_dir), str(final_path))

        return final_path

    def _register_skill(
        self,
        metadata: SkillMetadata,
        source: str,
        path: Path,
    ) -> None:
        """Register skill in skills.yaml.

        Args:
            metadata: Skill metadata
            source: Original source string
            path: Installation path
        """
        config = self._load_config()

        installed = InstalledSkill(
            name=metadata.name,
            source=source,
            version=metadata.version,
            installed_at=datetime.now(),
            path=str(path),
        )

        config.add_skill(installed)
        self._save_config(config)

    def _unregister_skill(self, name: str) -> None:
        """Remove skill from skills.yaml.

        Args:
            name: Skill name
        """
        config = self._load_config()
        config.remove_skill(name)
        self._save_config(config)

    def _load_config(self) -> SkillsConfig:
        """Load skills.yaml config.

        Returns:
            SkillsConfig (empty if file doesn't exist)
        """
        if not self.config_path.exists():
            return SkillsConfig()

        try:
            with open(self.config_path) as f:
                data = yaml.safe_load(f) or {}
            return SkillsConfig(**data)
        except Exception as e:
            logger.warning(f"Failed to load skills.yaml: {e}")
            return SkillsConfig()

    def _save_config(self, config: SkillsConfig) -> None:
        """Save skills.yaml config.

        Args:
            config: Config to save
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w") as f:
            yaml.dump(
                config.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def list_installed(self) -> list[InstalledSkill]:
        """List all installed skills from config.

        Returns:
            List of InstalledSkill records
        """
        return self._load_config().skills
