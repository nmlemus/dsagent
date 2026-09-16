"""`run_skill_script`: the only way a persona runs a script its skill ships.

Skills are mounted at a virtual `/skills/<persona>/` path that the file tools
resolve and nothing running *inside* the env can see. Run 001's step
instructions told the persona to `execute` that path; it would have failed on
the first step of run 002.
"""

from pathlib import Path

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.cartridge.models import EnvSpec
from dsagent.envs.base import Env
from dsagent.envs.kernel import make_kernel_env
from dsagent.host.build import SKILLS_DIR, _skill_script_tool, materialize_skills

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


@pytest.fixture
def env_and_workspace(tmp_path):
    """A real kernel env: LocalShellBackend, no kernel started."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    env = make_kernel_env(EnvSpec(name="default", kind="kernel"), workspace)
    return env, workspace


def tool_for(persona, env, workspace, cartridge=None):
    cartridge = cartridge or load_cartridge(DS)
    materialize_skills([cartridge], workspace)
    return _skill_script_tool(cartridge, persona, env, workspace)


def write_script(workspace, persona, skill, name, body):
    d = workspace / SKILLS_DIR / persona / skill / "scripts"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body)


def test_the_virtual_skills_path_is_not_reachable_from_the_env(env_and_workspace):
    """The bug this tool exists for: /skills/... is a file-tool mount, not a path."""
    env, _ = env_and_workspace
    assert env.backend.execute("ls /skills/marie/eda/scripts").exit_code != 0


def test_it_runs_a_script_from_a_granted_skill(env_and_workspace):
    env, workspace = env_and_workspace
    run = tool_for("marie", env, workspace)
    write_script(workspace, "marie", "eda", "hello.py", "print('from the skill')")
    out = run.invoke({"skill": "eda", "script": "hello.py"})
    assert "exit code: 0" in out
    assert "from the skill" in out


def test_arguments_reach_the_script(env_and_workspace):
    env, workspace = env_and_workspace
    run = tool_for("marie", env, workspace)
    write_script(workspace, "marie", "eda", "argv.py", "import sys; print('|'.join(sys.argv[1:]))")
    out = run.invoke({"skill": "eda", "script": "argv.py",
                      "argv": ["data/a b.csv", "artifacts/", "--key-column", "date"]})
    assert "data/a b.csv|artifacts/|--key-column|date" in out  # quoting survives a space


def test_it_runs_from_the_workspace_root(env_and_workspace):
    """So a workspace-relative path in `args` means what the step instructions say."""
    env, workspace = env_and_workspace
    run = tool_for("marie", env, workspace)
    (workspace / "data").mkdir()
    (workspace / "data" / "raw.csv").write_text("a,b\n1,2\n")
    write_script(workspace, "marie", "eda", "read.py",
                 "import sys; print(open(sys.argv[1]).read().strip())")
    out = run.invoke({"skill": "eda", "script": "read.py", "argv": ["data/raw.csv"]})
    assert "a,b" in out


def test_it_uses_the_env_interpreter_not_whatever_is_on_path(env_and_workspace):
    """The cartridge's requirements are installed in DSAgent's interpreter."""
    env, workspace = env_and_workspace
    run = tool_for("marie", env, workspace)
    write_script(workspace, "marie", "eda", "which.py", "import sys; print(sys.executable)")
    assert env.python in run.invoke({"skill": "eda", "script": "which.py"})


def test_a_failing_script_reports_its_exit_code_and_stderr(env_and_workspace):
    env, workspace = env_and_workspace
    run = tool_for("marie", env, workspace)
    write_script(workspace, "marie", "eda", "boom.py", "import sys; sys.exit('bad key column')")
    out = run.invoke({"skill": "eda", "script": "boom.py"})
    assert "exit code: 1" in out
    assert "bad key column" in out


def test_a_skill_the_persona_was_not_granted_is_an_error(env_and_workspace):
    """Scope is enforced here, not left to a path that happens not to exist."""
    env, workspace = env_and_workspace
    run = tool_for("marie", env, workspace)          # marie has eda + reports, not mmm
    write_script(workspace, "marie", "mmm", "sneak.py", "print('should not run')")
    out = run.invoke({"skill": "mmm", "script": "sneak.py"})
    assert out.startswith("error: 'mmm' is not one of your skills")
    assert "eda, reports" in out
    assert "should not run" not in out


def test_a_missing_script_lists_what_the_skill_does_have(env_and_workspace):
    env, workspace = env_and_workspace
    run = tool_for("marie", env, workspace)
    out = run.invoke({"skill": "eda", "script": "nope.py"})
    assert out.startswith("error: skill 'eda' has no script 'nope.py'")
    assert "profile.py" in out


def test_the_real_eda_script_runs_through_the_tool(env_and_workspace):
    """End to end on the cartridge as shipped — the run-002 step-1 command."""
    pytest.importorskip("pandas")
    env, workspace = env_and_workspace
    run = tool_for("marie", env, workspace)
    (workspace / "data").mkdir()
    (workspace / "data" / "raw.csv").write_text("id,v\n1,10\n2,20\n")
    out = run.invoke({"skill": "eda", "script": "profile.py",
                      "argv": ["data/raw.csv", "artifacts/", "--key-column", "id"]})
    assert "exit code: 0" in out, out
    assert (workspace / "artifacts" / "data-profile.json").exists()


def test_every_persona_agent_is_given_the_tool():
    cartridge = load_cartridge(DS)
    for persona in cartridge.personas:
        env = Env(spec=EnvSpec(name="e", kind="kernel"), workspace=Path("/tmp"), backend=None)
        assert _skill_script_tool(cartridge, persona, env, Path("/tmp")).name == "run_skill_script"
