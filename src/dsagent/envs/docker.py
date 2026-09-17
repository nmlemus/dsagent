"""Docker env: one container per run, with the run's workspace bind-mounted.

Implements Deep Agents' `BaseSandbox`, which only needs `execute()`,
`upload_files()` and `download_files()`; ls/read/write/edit/glob/grep are
derived from those.

**The workspace is a bind-mount, never a copy** (`docs/architecture.md` §3.4).
Everything the harness does after a step reads the run directory from the host —
`produces` is verified there, the files endpoint serves from it, `show_chart`
names a file in it, the zip is built from it — so a container writing to its own
copy would produce a run whose evidence does not exist. Skills come along inside
that mount, because they are materialized under `<workspace>/.dsagent/skills/`;
they get a second, read-only bind of the same directory so that a persona cannot
rewrite the skill it was granted.

Every `docker` invocation is built by a function that returns a list of strings
and runs nothing, and the thing that runs them is injected. That is what makes
the mounts, the flags and the cleanup testable without a daemon: the unit tests
assert the argv, and the integration test — `DSAGENT_DOCKER=1` — runs it for
real.
"""

from __future__ import annotations

import base64
import os
import shlex
import subprocess
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from dsagent.cartridge.models import EnvSpec
from dsagent.envs.base import Env

WORKDIR = "/workspace"
"""Where the run's workspace is mounted, and the container's working directory.

Fixed rather than configurable: `run_skill_script` and the auto-gate run paths
that are relative to the workspace root, so "the workspace root" has to mean one
thing on both sides of the boundary."""

SKILLS = ".dsagent/skills"
"""Materialized skills, relative to the workspace. Inside the mount already;
bound a second time read-only so a step cannot edit what it was granted."""

DEFAULT_TIMEOUT = 3600
PYTHON = "python3"
"""The interpreter inside the image. Stated rather than inherited: `Env.python`
defaults to `python3` anyway, but the auto-gate and `run_skill_script` both run
`env.python`, and "whatever the default happens to be" is not a thing to leave
implicit across a machine boundary."""


class DockerError(RuntimeError):
    """Docker itself refused — no daemon, no image, a build that failed."""


Runner = Callable[..., "subprocess.CompletedProcess[str]"]
"""How a `docker` argv actually gets run. Injected so the tests can watch.

Takes the argv and an optional `timeout=`; anything that honours that shape will
do, which is the whole point — the unit tests hand it a recorder."""


def run_docker(argv: Sequence[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, check=False)


# ---- the argv, as values -----------------------------------------------------


def run_argv(image: str, container: str, workspace: Path, *, gpu: bool = False,
             user: str | None = None) -> list[str]:
    """`docker run` for a run's container: detached, mounted, idle.

    `sleep infinity` rather than a command, because the container is the *run's*
    machine and outlives any one step: a fit and the optimizer that reads its
    posterior are the same container, and `docker exec` is how each step lands
    in it.
    """
    argv = [
        "docker", "run", "--detach", "--name", container,
        "--volume", f"{workspace}:{WORKDIR}",
        # The same directory again, read-only, over the top: skills arrive with
        # the workspace, and a persona may run them but not rewrite them.
        "--volume", f"{workspace / SKILLS}:{WORKDIR}/{SKILLS}:ro",
        "--workdir", WORKDIR,
    ]
    if user:
        argv += ["--user", user]
    if gpu:
        argv += ["--gpus", "all"]
    return [*argv, image, "sleep", "infinity"]


def exec_argv(container: str, command: str) -> list[str]:
    """`docker exec` for one shell command. `-lc` so a login shell's PATH applies."""
    return ["docker", "exec", container, "bash", "-lc", command]


def build_argv(context: Path, tag: str) -> list[str]:
    return ["docker", "build", "--tag", tag, str(context)]


def remove_argv(container: str) -> list[str]:
    """`--force` because a running container is exactly the case this is for."""
    return ["docker", "rm", "--force", container]


def image_tag(spec: EnvSpec) -> str:
    """The image an env builds to, namespaced by the cartridge that declared it.

    Two cartridges may both declare `meridian`; without the cartridge in the tag
    the second build silently replaces the first.
    """
    if spec.image:
        return spec.image
    return f"dsagent/{spec.cartridge or 'cartridge'}-{spec.name}:latest"


def host_user() -> str:
    """`uid:gid` of whoever is running DSAgent, or "" where that has no meaning.

    A container writing as root leaves a workspace its owner cannot delete, and
    the workspace is the run's record. Windows has no uid; there the mount's
    ownership is the daemon's business and this stays empty.
    """
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:  # pragma: no cover — not POSIX
        return ""
    return f"{getuid()}:{getgid()}"


# ---- the backend -------------------------------------------------------------


class DockerBackend(BaseSandbox):
    def __init__(self, image: str, workspace: Path, *, gpu: bool = False,
                 timeout: int = DEFAULT_TIMEOUT, user: str | None = None,
                 runner: Runner | None = None) -> None:
        self._id = f"dsagent-{uuid.uuid4().hex[:12]}"
        self._runner: Runner = runner or run_docker
        self._closed = False
        self.image = image
        self.timeout = timeout
        # The read-only bind needs something to bind: Docker would otherwise
        # create it as a root-owned directory on the host.
        (workspace / SKILLS).mkdir(parents=True, exist_ok=True)
        started = self._runner(run_argv(image, self._id, workspace, gpu=gpu, user=user))
        if started.returncode != 0:
            raise DockerError(
                f"could not start a container from {image}: "
                f"{(started.stderr or started.stdout or '').strip()[-400:]}"
            )

    @property
    def id(self) -> str:
        return self._id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        try:
            r = self._runner(exec_argv(self._id, command), timeout=timeout or self.timeout)
        except subprocess.TimeoutExpired:
            seconds = timeout or self.timeout
            return ExecuteResponse(
                output=f"[timeout after {seconds}s]", exit_code=124, truncated=False
            )
        return ExecuteResponse(
            output=(r.stdout or "") + (r.stderr or ""), exit_code=r.returncode, truncated=False
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        out: list[FileUploadResponse] = []
        for path, data in files:
            b64 = base64.b64encode(data).decode()
            quoted = shlex.quote(path)
            r = self.execute(
                f"mkdir -p $(dirname {quoted}) && echo {b64} | base64 -d > {quoted}"
            )
            out.append(
                FileUploadResponse(path=path, error=None if r.exit_code == 0 else "permission_denied")
            )
        return out

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        out: list[FileDownloadResponse] = []
        for path in paths:
            r = self.execute(f"base64 {shlex.quote(path)} | tr -d '\\n'")
            if r.exit_code == 0:
                out.append(
                    FileDownloadResponse(
                        path=path, content=base64.b64decode(r.output.strip()), error=None
                    )
                )
            else:
                out.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
        return out

    def close(self) -> None:
        """Remove the container. Safe to call twice, and never raises.

        Called from the runner's `finally`, which may itself be unwinding an
        exception: a cleanup that raises would replace the reason the run failed
        with the reason the cleanup failed.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._runner(remove_argv(self._id))
        except Exception:  # noqa: BLE001, S110 — a dying container must not mask the run's error
            pass


def ensure_image(spec: EnvSpec, *, runner: Runner | None = None) -> str:
    """The image for this env, built from its context if it is not a pull."""
    tag = image_tag(spec)
    if spec.image:
        return tag
    if spec.build is None:
        raise DockerError(f"env '{spec.name}' is a docker env with neither `image` nor `build`")
    r = (runner or run_docker)(build_argv(spec.build, tag))
    if r.returncode != 0:
        raise DockerError(
            f"could not build {tag} from {spec.build}: "
            f"{(r.stderr or r.stdout or '').strip()[-400:]}"
        )
    return tag


def make_docker_env(spec: EnvSpec, workspace: Path, *, runner: Runner | None = None) -> Env:
    image = ensure_image(spec, runner=runner)
    backend = DockerBackend(
        image, workspace, gpu=spec.gpu == "required", user=host_user() or None, runner=runner,
    )
    # No `closers`: `Env.close` already calls the backend's own `close`, and the
    # container must be removed exactly as reliably either way.
    return Env(spec=spec, workspace=workspace, backend=backend, python=PYTHON)
