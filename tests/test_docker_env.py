"""`DockerBackend`: the argv, the mounts and the cleanup, without a daemon.

Everything `DockerBackend` asks of Docker is a list of strings built by a
function that runs nothing, and the thing that runs them is injected. So the
mounts, the flags and the removal are ordinary unit tests, and the one test that
genuinely needs a daemon says so and skips.

The integration test at the bottom is the other half: it builds the cartridge's
own image and runs something in it. `DSAGENT_DOCKER=1` to opt in — it pulls a
base image and installs Meridian the first time, which is minutes, not seconds.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from dsagent.cartridge.models import EnvSpec
from dsagent.envs.docker import (
    SKILLS,
    WORKDIR,
    DockerBackend,
    DockerError,
    build_argv,
    ensure_image,
    exec_argv,
    host_user,
    image_tag,
    make_docker_env,
    remove_argv,
    run_argv,
    run_docker,
)

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"
BASE = "python:3.12-slim"
"""The image the cartridge's own Dockerfile starts from, for the tests that are
about the mount rather than about Meridian. Already local once the build below has
run once, so these cost seconds."""


class FakeDocker:
    """A docker client that records argv and answers from a script.

    Answers are matched on the subcommand (`run`, `exec`, `build`, `rm`), which
    is the only thing the backend branches on. Anything unscripted succeeds
    silently, because most calls in these tests are setup rather than subject.
    """

    def __init__(self, **answers: tuple[int, str, str]) -> None:
        self.calls: list[list[str]] = []
        self.timeouts: list[int | None] = []
        self.answers = answers
        self.raises: Exception | None = None

    def __call__(self, argv, *, timeout=None):
        self.calls.append(list(argv))
        self.timeouts.append(timeout)
        if self.raises is not None:
            raise self.raises
        code, out, err = self.answers.get(argv[1], (0, "", ""))
        return subprocess.CompletedProcess(list(argv), code, out, err)

    def of(self, subcommand: str) -> list[list[str]]:
        return [c for c in self.calls if len(c) > 1 and c[1] == subcommand]


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "workspace").mkdir()
    return tmp_path / "workspace"


# ---- the argv, as values -----------------------------------------------------


def test_a_run_mounts_the_workspace_and_works_in_it(workspace):
    argv = run_argv("img:1", "c1", workspace)

    assert argv[:5] == ["docker", "run", "--detach", "--name", "c1"]
    assert f"{workspace}:{WORKDIR}" in argv
    assert ["--workdir", WORKDIR] == argv[argv.index("--workdir"):argv.index("--workdir") + 2]
    # Idle, not a command: the container is the *run's* machine and outlives any
    # one step, which is what lets a later step read what an earlier one wrote.
    assert argv[-3:] == ["img:1", "sleep", "infinity"]


def test_skills_are_mounted_again_read_only(workspace):
    """They arrive inside the workspace mount; this is the containment, not the
    reachability — a persona may run the skill it was granted, not rewrite it."""
    argv = run_argv("img:1", "c1", workspace)

    assert f"{workspace / SKILLS}:{WORKDIR}/{SKILLS}:ro" in argv
    # And it lands *after* the workspace, or the writable mount would cover it.
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--volume"]
    assert mounts[0].endswith(f":{WORKDIR}")
    assert mounts[1].endswith(":ro")


def test_the_container_runs_as_the_person_who_started_the_run(workspace):
    """Root in the container leaves a workspace its owner cannot delete, and the
    workspace is the run's record."""
    argv = run_argv("img:1", "c1", workspace, user="501:20")

    assert argv[argv.index("--user") + 1] == "501:20"
    assert "--user" not in run_argv("img:1", "c1", workspace)


def test_a_gpu_env_asks_for_gpus_and_others_do_not(workspace):
    assert "--gpus" in run_argv("img:1", "c1", workspace, gpu=True)
    assert "--gpus" not in run_argv("img:1", "c1", workspace)


def test_exec_build_and_remove_are_what_they_say():
    assert exec_argv("c1", "ls -la") == ["docker", "exec", "c1", "bash", "-lc", "ls -la"]
    assert build_argv(Path("/ctx"), "t:1") == ["docker", "build", "--tag", "t:1", "/ctx"]
    # `--force`: a running container is exactly the case cleanup is for.
    assert remove_argv("c1") == ["docker", "rm", "--force", "c1"]


def test_an_image_is_named_after_the_cartridge_that_declared_it():
    """Two cartridges may both declare `meridian`; without this the second build
    silently replaces the first."""
    built = EnvSpec(name="meridian", cartridge="ds", kind="docker", build=Path("/ctx"))
    pulled = EnvSpec(name="meridian", cartridge="ds", kind="docker", image="ghcr.io/x/y:2")

    assert image_tag(built) == "dsagent/ds-meridian:latest"
    assert image_tag(pulled) == "ghcr.io/x/y:2"
    assert image_tag(EnvSpec(name="e", kind="docker")) == "dsagent/cartridge-e:latest"


def test_output_survives_a_process_that_leaves_a_writer_behind():
    """The real `run_docker`, against the thing that made it necessary.

    The Docker CLI spawns `docker-credential-desktop get` with its own stderr and
    is reaped seconds later; the helper is orphaned and holds that end open. Read
    through a pipe, the wait is for an EOF that never comes — seen here as a
    `docker build` that sat for forty minutes having never pulled a layer. So the
    stand-in: a command that prints, backgrounds a child holding both streams,
    and exits at once. The timeout is the assertion — this hangs without the fix.
    """
    done = run_docker(["bash", "-c", "echo out; echo err >&2; sleep 30 & exit 0"], timeout=15)

    assert (done.returncode, done.stdout.strip(), done.stderr.strip()) == (0, "out", "err")


def test_nothing_docker_asks_for_interactively_can_be_answered_by_a_run():
    done = run_docker(["bash", "-c", "read -r line; echo \"[$line]\""], timeout=15)

    assert done.stdout.strip() == "[]", "stdin should be /dev/null, not this terminal"


def test_the_host_user_is_a_uid_gid_pair_on_posix():
    user = host_user()
    assert user == "" or (user.count(":") == 1 and all(p.isdigit() for p in user.split(":")))


# ---- the backend -------------------------------------------------------------


def test_starting_a_backend_starts_one_container(workspace):
    docker = FakeDocker()

    backend = DockerBackend("img:1", workspace, runner=docker)

    assert len(docker.of("run")) == 1
    assert backend.id.startswith("dsagent-")
    assert backend.id in docker.of("run")[0]
    # The read-only bind needs a directory to bind, or Docker makes one as root.
    assert (workspace / SKILLS).is_dir()


def test_a_daemon_that_refuses_is_an_error_with_its_own_words(workspace):
    docker = FakeDocker(run=(125, "", "Cannot connect to the Docker daemon"))

    with pytest.raises(DockerError, match="Cannot connect to the Docker daemon"):
        DockerBackend("img:1", workspace, runner=docker)


def test_execute_reports_the_exit_code_and_both_streams(workspace):
    docker = FakeDocker(exec=(3, "out", "err"))
    backend = DockerBackend("img:1", workspace, runner=docker)

    answer = backend.execute("false")

    assert docker.of("exec")[0] == exec_argv(backend.id, "false")
    assert (answer.exit_code, answer.output) == (3, "outerr")


def test_a_command_that_never_returns_is_a_timeout_not_a_hang(workspace):
    docker = FakeDocker()
    backend = DockerBackend("img:1", workspace, runner=docker)
    docker.raises = subprocess.TimeoutExpired(cmd="docker", timeout=5)

    answer = backend.execute("sleep 99", timeout=5)

    assert answer.exit_code == 124
    assert "timeout after 5s" in answer.output


def test_files_go_in_and_come_back_out(workspace):
    """Upload and download are shell commands; this is that they are the right
    ones and that the round trip is base64-clean."""
    import base64

    payload = b"weather,precip\nrain,5.4\n"
    docker = FakeDocker(exec=(0, base64.b64encode(payload).decode(), ""))
    backend = DockerBackend("img:1", workspace, runner=docker)

    up = backend.upload_files([("artifacts/a.csv", payload)])
    down = backend.download_files(["artifacts/a.csv"])

    assert up[0].error is None
    assert down[0].content == payload
    assert "base64 -d" in docker.of("exec")[0][-1]


def test_a_file_that_is_not_there_is_not_found(workspace):
    docker = FakeDocker(exec=(1, "", "No such file"))
    backend = DockerBackend("img:1", workspace, runner=docker)

    assert backend.download_files(["nope.csv"])[0].error == "file_not_found"


def test_closing_removes_the_container_once(workspace):
    docker = FakeDocker()
    backend = DockerBackend("img:1", workspace, runner=docker)

    backend.close()
    backend.close()

    assert docker.of("rm") == [remove_argv(backend.id)]


def test_a_cleanup_that_fails_does_not_replace_the_reason_the_run_failed(workspace):
    """`close()` is called from a `finally` that may be unwinding an exception."""
    docker = FakeDocker()
    backend = DockerBackend("img:1", workspace, runner=docker)
    docker.raises = OSError("docker went away")

    backend.close()  # must not raise


def test_the_env_closes_its_container(workspace):
    docker = FakeDocker()
    spec = EnvSpec(name="meridian", cartridge="ds", kind="docker", image="img:1")

    env = make_docker_env(spec, workspace, runner=docker)
    env.close()

    assert env.python == "python3", "the auto-gate and run_skill_script run env.python"
    assert docker.of("build") == [], "an env with an image does not build"
    assert len(docker.of("rm")) == 1


def test_an_env_with_a_build_context_builds_before_it_runs(workspace, tmp_path):
    docker = FakeDocker()
    spec = EnvSpec(name="meridian", cartridge="ds", kind="docker", build=tmp_path / "ctx")

    make_docker_env(spec, workspace, runner=docker)

    assert docker.of("build")[0] == build_argv(tmp_path / "ctx", "dsagent/ds-meridian:latest")
    assert docker.calls.index(docker.of("build")[0]) < docker.calls.index(docker.of("run")[0])


def test_a_build_that_fails_says_so_rather_than_running_a_stale_image(tmp_path):
    docker = FakeDocker(build=(1, "", "failed to solve: no such file"))
    spec = EnvSpec(name="meridian", cartridge="ds", kind="docker", build=tmp_path / "ctx")

    with pytest.raises(DockerError, match="no such file"):
        ensure_image(spec, runner=docker)


def test_a_docker_env_declaring_neither_image_nor_build_is_refused():
    with pytest.raises(DockerError, match="neither"):
        ensure_image(EnvSpec(name="e", cartridge="ds", kind="docker"), runner=FakeDocker())


# ---- and once, for real ------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("DSAGENT_DOCKER") != "1",
    reason="needs a Docker daemon; set DSAGENT_DOCKER=1 (builds Meridian, takes minutes)",
)
def test_the_meridian_image_builds_and_runs_meridian(tmp_path):
    """The cartridge's own env, built and exercised.

    Everything above asserts what DSAgent *asks* Docker to do. This is the one
    that asserts Docker does it: the image the `ds` cartridge declares, built
    from its Dockerfile, with `import meridian` succeeding inside it.
    """
    from dsagent.cartridge import load_cartridge

    spec = load_cartridge(DS).envs["meridian"]
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    env = make_docker_env(spec, workspace)
    try:
        imported = env.backend.execute(f"{env.python} -c 'import meridian; print(meridian.__name__)'")
        assert imported.exit_code == 0, imported.output
        assert "meridian" in imported.output

        # The mount is a mount: a directory and a file made inside are on the host
        # at once, owned by whoever started the run rather than by root. The
        # `mkdir` is the container's, not the fixture's — a step that writes its
        # first artifact does exactly this, and root-owned output is the failure
        # `--user` exists to prevent.
        written = env.backend.execute("mkdir -p artifacts && echo hello > artifacts/from-container.txt")
        assert written.exit_code == 0, written.output
        landed = workspace / "artifacts" / "from-container.txt"
        assert landed.read_text().strip() == "hello"
        if hasattr(os, "getuid"):
            assert landed.stat().st_uid == os.getuid()

        # And the read-only bind holds.
        refused = env.backend.execute(f"touch {WORKDIR}/{SKILLS}/nope")
        assert refused.exit_code != 0
    finally:
        env.close()

    still_there = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name={env.backend.id}"],
        capture_output=True, text=True, check=False,
    )
    assert still_there.stdout.strip() == "", "the container outlived its env"


@pytest.mark.skipif(
    os.environ.get("DSAGENT_DOCKER") != "1",
    reason="needs a Docker daemon; set DSAGENT_DOCKER=1",
)
def test_skills_rematerialize_under_a_live_mount(tmp_path):
    """The container mounts `.dsagent/skills`, and then the host rebuilds it.

    That is the real order: `env_for` starts the container, and the persona's
    agent is built after it. `materialize_skills` used to replace the root
    directory, which a bind-mount cannot follow — and on macOS the mount makes
    the root undeletable, so it emptied the contents and raised `PermissionError`
    on the first step of every Docker run.
    """
    from dsagent.cartridge import load_cartridge
    from dsagent.host.build import materialize_skills

    cartridge = load_cartridge(DS)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = EnvSpec(name="e", cartridge="ds", kind="docker", image=BASE)

    env = make_docker_env(spec, workspace)
    try:
        materialize_skills([cartridge], workspace)  # the call that used to raise

        listed = env.backend.execute(f"ls {WORKDIR}/{SKILLS}")
        assert listed.exit_code == 0, listed.output
        assert "marie" in listed.output, "the container is reading a stale directory"
        refused = env.backend.execute(f"touch {WORKDIR}/{SKILLS}/nope")
        assert refused.exit_code != 0, "re-materializing must not make the bind writable"
    finally:
        env.close()


@pytest.mark.skipif(
    os.environ.get("DSAGENT_DOCKER") != "1",
    reason="needs a Docker daemon; set DSAGENT_DOCKER=1",
)
def test_a_step_writes_its_produces_in_the_container_and_the_host_verifies_it(tmp_path):
    """Invariant 5, across a machine boundary.

    `produces` is verified on the host by the runner, against the run directory,
    with one check for every env kind. That only holds because the workspace is a
    bind-mount and not a copy: a container writing to its own copy would leave a
    run whose evidence does not exist. Nothing here patches `make_env` — the step
    really runs in a container.
    """
    from dsagent.runner import WorkflowRunner
    from tests.fakes import produces_of, tiny_cartridge

    cartridge = tiny_cartridge(
        tmp_path / "cart",
        [{"id": "work", "produces": ["artifacts/out.txt", "figures/*.txt"]}],
        envs={"default": {"kind": "docker", "image": BASE}},
    )

    class ContainerAgent:
        """A persona that works where a real one does: inside the env."""

        def __init__(self, env):
            self.env = env

        def invoke(self, payload):
            for entry in produces_of(payload["messages"][0]["content"]):
                for rel in (entry.replace("*", "one"), entry.replace("*", "two")):
                    self.env.backend.execute(
                        f"mkdir -p $(dirname {rel}) && echo 'written in the container' > {rel}"
                    )
            return {"messages": [{"role": "assistant", "content": "done"}]}

    runner = WorkflowRunner(
        cartridge, tmp_path / "run",
        agent_factory=lambda cart, persona, env, ws: ContainerAgent(env),
        log=lambda m: None,
    )
    try:
        state = runner.run("w", {"data_path": "data/x.csv"})
    finally:
        runner.close()

    assert state.status == "done", state.steps["work"].error
    landed = tmp_path / "run" / "workspace" / "artifacts" / "out.txt"
    assert landed.read_text().strip() == "written in the container"
    assert sorted(p.name for p in (tmp_path / "run" / "workspace" / "figures").iterdir()) == [
        "one.txt", "two.txt",
    ], "a glob written inside the container expands on the host"
    if hasattr(os, "getuid"):
        assert landed.stat().st_uid == os.getuid(), "the run's own record is owned by root"


@pytest.mark.skipif(
    os.environ.get("DSAGENT_DOCKER") != "1",
    reason="needs a Docker daemon; set DSAGENT_DOCKER=1",
)
def test_a_skill_script_runs_inside_the_container(tmp_path):
    """#77 concluded skills need no mount of their own. This executes the claim.

    They are materialized *inside* the workspace, and `run_skill_script` resolves
    a workspace-relative path and runs it through the backend — so the one
    bind-mount carries them, and the tool works in Docker with no Docker-shaped
    code path anywhere near it.
    """
    from dsagent.cartridge import load_cartridge
    from dsagent.envs.base import SKILLS_DIR
    from dsagent.host.build import _skill_script_tool, materialize_skills

    cartridge = load_cartridge(DS)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env = make_docker_env(EnvSpec(name="e", cartridge="ds", kind="docker", image=BASE), workspace)
    try:
        materialize_skills([cartridge], workspace)
        skill = min(s.name for s in cartridge.skills_for("marie"))
        scripts = workspace / SKILLS_DIR / "marie" / skill / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "where.py").write_text(
            "import os, sys; print('cwd', os.getcwd()); print('argv', sys.argv[1])\n"
        )

        run_skill_script = _skill_script_tool(cartridge, "marie", env, workspace)
        out = run_skill_script.invoke({"skill": skill, "script": "where.py", "argv": ["hello"]})

        assert "exit code: 0" in out, out
        assert f"cwd {WORKDIR}" in out, "the script ran outside the mounted workspace"
        assert "argv hello" in out
    finally:
        env.close()
