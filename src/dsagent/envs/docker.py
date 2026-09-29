"""Docker env: one container per run, with the run's workspace bind-mounted.

Implements Deep Agents' `BaseSandbox`, which only needs `execute()`,
`upload_files()` and `download_files()`; ls/read/write/edit/glob/grep are
derived from those.

**The workspace is a bind-mount, never a copy** (`docs/architecture.md` §3.4).
Everything the harness does after a step reads the run directory from the host —
`produces` is verified there, the files endpoint serves from it, `show_chart`
names a file in it, the zip is built from it — so a container writing to its own
copy would produce a run whose evidence does not exist. Harness-controlled
material comes along under `<workspace>/.dsagent/`; that entire directory gets a
second, read-only bind so generated code cannot rewrite its skills or checks.

Every `docker` invocation is built by a function that returns a list of strings
and runs nothing, and the thing that runs them is injected. That is what makes
the mounts, the flags and the cleanup testable without a daemon: the unit tests
assert the argv, and the integration test — `DSAGENT_DOCKER=1` — runs it for
real.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import tempfile
import textwrap
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox
from langchain_core.tools import tool

from dsagent.cartridge.models import EnvSpec
from dsagent.envs.base import SKILLS_DIR, Env
from dsagent.processes import PROCESS_DIR_ENV, resource_intent

WORKDIR = "/workspace"
"""Where the run's workspace is mounted, and the container's working directory.

Fixed rather than configurable: `run_skill_script` and the auto-gate run paths
that are relative to the workspace root, so "the workspace root" has to mean one
thing on both sides of the boundary."""

SKILLS = SKILLS_DIR.as_posix()
"""Materialized skills, relative to the workspace. Inside the mount already;
bound a second time read-only so a step cannot edit what it was granted."""
CONTROL_DIR = SKILLS_DIR.parent
"""All trusted material exposed inside the workspace, mounted read-only."""

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
    """Run a `docker` argv, collecting its output through files rather than pipes.

    Not `capture_output=True`, which is the obvious way to write this and hangs
    forever. A pipe ends when every writer closes it, and the Docker CLI leaves
    writers behind: `docker-credential-desktop get` is spawned with the CLI's own
    stderr, is orphaned when the CLI exits, and holds that end open. The child is
    reaped in seconds and `communicate()` then waits on an EOF that never comes —
    observed here as a `docker build` that sat for forty minutes having never
    pulled a layer. A file has no such rule: this waits on the process, and the
    process alone.

    `stdin` is `/dev/null` for the same family of reasons — nothing docker asks
    for interactively can be answered by a run.
    """
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as out, \
         tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err:
        done = subprocess.run(
            list(argv), stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            timeout=timeout, check=False,
        )
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(list(argv), done.returncode, out.read(), err.read())


# ---- the argv, as values -----------------------------------------------------


@dataclass(frozen=True)
class SandboxLimits:
    """Host policy. A cartridge cannot weaken these container restrictions."""

    memory: str = "8g"
    cpus: str = "4"
    pids: int = 256
    tmpfs: str = "512m"


DEFAULT_LIMITS = SandboxLimits()
SANDBOX_ENV = {
    "HOME": "/tmp/dsagent-home",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "IPYTHONDIR": "/tmp/ipython",
    "JUPYTER_CONFIG_DIR": "/tmp/jupyter",
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "XDG_CACHE_HOME": "/tmp/cache",
}


def run_argv(image: str, container: str, workspace: Path, *, gpu: bool = False,
             user: str | None = None, limits: SandboxLimits = DEFAULT_LIMITS) -> list[str]:
    """`docker run` for a run's container: detached, mounted, idle.

    `sleep infinity` rather than a command, because the container is the *run's*
    machine and outlives any one step: a fit and the optimizer that reads its
    posterior are the same container, and `docker exec` is how each step lands
    in it.
    """
    argv = [
        "docker", "run", "--detach", "--name", container,
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges=true",
        "--pids-limit", str(limits.pids),
        "--memory", limits.memory,
        "--memory-swap", limits.memory,
        "--cpus", limits.cpus,
        "--ulimit", "nofile=1024:1024",
        "--init",
        "--tmpfs", f"/tmp:rw,nosuid,nodev,noexec,size={limits.tmpfs},mode=1777",
        "--volume", f"{workspace}:{WORKDIR}",
        # Harness material (skills and auto-gate scripts) lives below this
        # directory. It may change from the trusted host after the mount starts,
        # but code inside the sandbox cannot rewrite it.
        "--volume", f"{workspace / CONTROL_DIR}:{WORKDIR}/{CONTROL_DIR.as_posix()}:ro",
        "--workdir", WORKDIR,
    ]
    for key, value in SANDBOX_ENV.items():
        argv += ["--env", f"{key}={value}"]
    if user:
        argv += ["--user", user]
    if gpu:
        argv += ["--gpus", "all"]
    return [*argv, image, "bash", "-lc", 'mkdir -p "$HOME" && exec sleep infinity']


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
        resolved_user = user or host_user()
        if not resolved_user or resolved_user.split(":", 1)[0] == "0":
            raise DockerError(
                "sandbox containers require a non-root POSIX host identity"
            )
        # The read-only bind needs something to bind: Docker would otherwise
        # create it as a root-owned directory on the host.
        (workspace / CONTROL_DIR).mkdir(parents=True, exist_ok=True)
        argv = run_argv(image, self._id, workspace, gpu=gpu, user=resolved_user)
        process_root = os.environ.get(PROCESS_DIR_ENV)
        if process_root:
            owner = Path(process_root).name
            resource_intent("docker", name=self._id, owner=owner)
            argv[2:2] = ["--label", f"dsagent.execution={owner}"]
        started = self._runner(argv)
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


def make_docker_env(
    spec: EnvSpec,
    workspace: Path,
    *,
    runner: Runner | None = None,
    user: str | None = None,
) -> Env:
    image = ensure_image(spec, runner=runner)
    backend = DockerBackend(
        image, workspace, gpu=spec.gpu == "required", user=user or host_user(), runner=runner,
    )
    try:
        inspected = (runner or run_docker)(
            ["docker", "inspect", "--format", "{{.Image}}", backend.id]
        )
        image_id = inspected.stdout.strip()
        if inspected.returncode != 0 or not image_id.startswith("sha256:"):
            raise DockerError(f"could not identify the runtime image for env '{spec.name}'")
    except BaseException:
        backend.close()
        raise
    kernel: DockerKernel | None = None

    @tool
    def run_python(code: str) -> str:
        """Run Python in the sandbox's persistent Jupyter kernel.

        Variables and imports persist for this environment. The kernel and its
        client both run inside the network-isolated container.
        """
        nonlocal kernel
        if kernel is None:
            kernel = DockerKernel(backend)
        try:
            return kernel.run(code)
        except TimeoutError:
            # DockerKernel stops a failed kernel. A later tool call must create
            # a fresh one rather than reusing the closed transport.
            kernel = None
            raise

    def close_kernel() -> None:
        if kernel is not None:
            kernel.close()

    return Env(spec=spec, workspace=workspace, backend=backend, python=PYTHON,
               tools=[run_python], closers=[close_kernel],
               runtime_identity={"docker_image_id": image_id})


_KERNEL_CLIENT = textwrap.dedent(r"""
    import base64, json, queue, sys, time
    from jupyter_client import BlockingKernelClient

    connection, encoded, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
    client = BlockingKernelClient(connection_file=connection)
    client.load_connection_file()
    client.start_channels()
    output = []
    try:
        client.wait_for_ready(timeout=min(timeout, 20))
        msg_id = client.execute(base64.b64decode(encoded).decode())
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("kernel execution deadline exceeded")
            try:
                msg = client.get_iopub_msg(timeout=remaining)
            except queue.Empty:
                raise TimeoutError("kernel execution deadline exceeded") from None
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            kind, content = msg["msg_type"], msg["content"]
            if kind == "stream":
                output.append(content.get("text", ""))
            elif kind in ("execute_result", "display_data"):
                data = content.get("data", {})
                output.append(data.get("text/plain", ""))
                if "image/png" in data:
                    output.append("[image/png output — save it to a file to keep it]")
            elif kind == "error":
                output.append("\n".join(content.get("traceback", [])))
            elif kind == "status" and content.get("execution_state") == "idle":
                break
        print("DSAGENT_KERNEL_RESULT=" + json.dumps({"output": "".join(output).strip()}))
    finally:
        client.stop_channels()
""")


class DockerKernel:
    """Persistent IPython kernel whose process and transport remain in Docker."""

    def __init__(self, backend: DockerBackend, timeout: int = 600) -> None:
        self.backend = backend
        self.timeout = timeout
        self.connection = "/tmp/dsagent-kernel.json"
        start = backend.execute(
            "rm -f /tmp/dsagent-kernel.json /tmp/dsagent-kernel.pid; "
            "nohup python3 -m ipykernel_launcher -f /tmp/dsagent-kernel.json "
            ">/tmp/dsagent-kernel.log 2>&1 & echo $! >/tmp/dsagent-kernel.pid; "
            "for i in $(seq 1 200); do test -s /tmp/dsagent-kernel.json && exit 0; sleep .05; done; "
            "cat /tmp/dsagent-kernel.log; exit 1",
            timeout=20,
        )
        if start.exit_code != 0:
            raise DockerError(f"sandbox kernel did not start: {start.output[-500:]}")

    def run(self, code: str) -> str:
        helper = base64.b64encode(_KERNEL_CLIENT.encode()).decode()
        payload = base64.b64encode(code.encode()).decode()
        command = " ".join(shlex.quote(value) for value in [
            PYTHON, "-c", f"import base64;exec(base64.b64decode('{helper}'))",
            self.connection, payload, str(self.timeout),
        ])
        result = self.backend.execute(command, timeout=self.timeout + 30)
        marker = "DSAGENT_KERNEL_RESULT="
        if result.exit_code != 0 or marker not in result.output:
            self.close()
            raise TimeoutError(
                f"sandbox kernel execution failed or exceeded {self.timeout}s: {result.output[-500:]}"
            )
        record = json.loads(result.output.rsplit(marker, 1)[1].splitlines()[0])
        return record["output"] or "(no output)"

    def close(self) -> None:
        self.backend.execute(
            "test ! -f /tmp/dsagent-kernel.pid || kill $(cat /tmp/dsagent-kernel.pid) 2>/dev/null || true",
            timeout=5,
        )
