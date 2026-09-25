"""Generated code receives a minimal environment, including unknown secrets."""

import json
import os
import shlex
import sys

import pytest

from dsagent.cartridge.models import EnvSpec
from dsagent.envs.kernel import JupyterKernel, make_kernel_env
from dsagent.envs.variables import compute_environment


@pytest.fixture
def secret_environment(monkeypatch):
    keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NEW_VENDOR_SECRET",
            "AWS_SESSION_TOKEN", "DSAGENT_PROCESS_DIR", "PYTHONPATH", "BASH_ENV")
    for key in keys:
        monkeypatch.setenv(key, "test-only-sentinel")
    # DSAGENT_PROCESS_DIR has meaning to the host supervisor; use a separate
    # assertion on the pure environment builder, then remove for direct shell.
    return keys


def test_allowlist_excludes_credentials_and_startup_hooks(tmp_path, secret_environment):
    env = compute_environment(tmp_path)
    assert not set(secret_environment) & env.keys()
    assert all(os.environ[key] == "test-only-sentinel" for key in secret_environment)
    assert str(tmp_path) in env["IPYTHONDIR"]


def test_actual_shell_does_not_inherit_secrets(tmp_path, secret_environment, monkeypatch):
    monkeypatch.delenv("DSAGENT_PROCESS_DIR")
    env = make_kernel_env(EnvSpec(name="default"), tmp_path)
    code = "import os,json; print(json.dumps(dict(os.environ)))"
    try:
        result = env.backend.execute(f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}")
        assert result.exit_code == 0
        variables = json.loads(result.output)
        assert not set(secret_environment) & variables.keys()
        assert variables["PYTHONNOUSERSITE"] == "1"
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get("DSAGENT_KERNEL") != "1", reason="requires a real local kernel")
def test_actual_kernel_does_not_inherit_secrets(tmp_path, secret_environment, monkeypatch):
    monkeypatch.delenv("DSAGENT_PROCESS_DIR")
    kernel = JupyterKernel(tmp_path)
    try:
        variables = json.loads(kernel.run("import os,json; print(json.dumps(dict(os.environ)))"))
        assert not set(secret_environment) & variables.keys()
        assert kernel.run("a = 21; print(a * 2)") == "42"
        assert kernel.run("print(a)") == "21"
    finally:
        kernel.close()
