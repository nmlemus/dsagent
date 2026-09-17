"""Env requirements: parsing, cartridge validation, and the kernel precondition.

No model, no kernel, no Docker — `make_kernel_env` raises before it builds
anything, and the install command is exercised with `--dry-run`.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dsagent.cartridge import CartridgeError, load_cartridge
from dsagent.cartridge.loader import _load_envs
from dsagent.cartridge.models import Cartridge, EnvSpec
from dsagent.cli import _kernel_envs, app
from dsagent.envs.base import EnvRequirementsError
from dsagent.envs.kernel import make_kernel_env
from dsagent.requirements import distribution_name, missing, pip_install_command

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"
INSTALLED = "pytest"  # a dev dependency: present whenever this test runs
ABSENT = "dsagent-definitely-not-a-real-distribution"


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("pandas", "pandas"),
        ("pandas>=2", "pandas"),
        ("  matplotlib  ", "matplotlib"),
        ("scikit-learn[extra]>=1,<2", "scikit-learn"),
        ("markdown; python_version>'3.8'", "markdown"),
        ("pkg @ https://example.invalid/pkg.whl", "pkg"),
    ],
)
def test_distribution_name_ignores_everything_after_the_name(requirement, expected):
    assert distribution_name(requirement) == expected


@pytest.mark.parametrize("requirement", ["", "   ", ">=2", "!!!"])
def test_distribution_name_rejects_non_requirements(requirement):
    with pytest.raises(ValueError, match="not a pip requirement"):
        distribution_name(requirement)


def test_missing_reports_only_what_is_absent():
    assert missing([INSTALLED]) == []
    assert missing([ABSENT, INSTALLED]) == [ABSENT]


def test_pip_install_command_quotes_specifiers():
    cmd = pip_install_command(["pandas>=2", "markdown"])
    assert "-m pip install" in cmd
    assert "'pandas>=2'" in cmd  # a bare >= would be a shell redirect
    assert cmd.endswith(" markdown")


# --- cartridge validation ----------------------------------------------------


def test_loader_accepts_a_list_of_requirement_strings(tmp_path):
    envs = _load_envs(tmp_path, {"default": {"kind": "kernel", "requirements": ["pandas>=2"]}})
    assert envs["default"].requirements == ["pandas>=2"]


def test_loader_defaults_requirements_to_empty(tmp_path):
    assert _load_envs(tmp_path, {"default": {"kind": "kernel"}})["default"].requirements == []


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        ("pandas", "must be a list"),
        ({"pandas": ">=2"}, "must be a list"),
        ([42], "must be a string"),
        ([">=2"], "not a pip requirement"),
    ],
)
def test_loader_rejects_malformed_requirements(tmp_path, declared, message):
    with pytest.raises(CartridgeError, match=message):
        _load_envs(tmp_path, {"broken": {"kind": "kernel", "requirements": declared}})


def test_docker_env_requirements_are_validated_but_not_checked(tmp_path):
    """Deps live in the Dockerfile; the harness only checks the field is well formed."""
    envs = _load_envs(tmp_path, {"box": {"kind": "docker", "image": "x", "requirements": [ABSENT]}})
    assert envs["box"].requirements == [ABSENT]


def test_ds_cartridge_declares_what_its_skill_scripts_need():
    default = load_cartridge(DS).envs["default"]
    assert default.kind == "kernel"
    # The last two are the odd ones out: nothing in the cartridge draws with
    # them. They are how `show_chart` validates a spec — altair for the schema,
    # vl-convert to compile and draw it once — declared here so a CLI run, which
    # has no `[ui]` extra, still validates before recording a chart.
    assert default.requirements == [
        "pandas>=2", "matplotlib", "markdown", "altair>=5.5,<6", "vl-convert-python>=1.7",
    ]


# --- kernel precondition -----------------------------------------------------


def test_kernel_env_fails_fast_with_the_install_command(tmp_path):
    spec = EnvSpec(name="default", kind="kernel", requirements=[INSTALLED, ABSENT])
    with pytest.raises(EnvRequirementsError) as e:
        make_kernel_env(spec, tmp_path)
    message = str(e.value)
    assert ABSENT in message
    assert INSTALLED not in message.split("missing")[1]  # only the missing one is reported
    assert "-m pip install" in message
    assert "dsagent cartridge install" in message


def test_kernel_env_provisions_when_requirements_are_satisfied(tmp_path):
    env = make_kernel_env(EnvSpec(name="default", kind="kernel", requirements=[INSTALLED]), tmp_path)
    assert [t.name for t in env.tools] == ["run_python"]
    env.close()  # no kernel was started: run_python was never called


# --- install command ---------------------------------------------------------


def test_cartridge_install_dry_run_prints_a_pasteable_command():
    result = CliRunner().invoke(app, ["cartridge", "install", str(DS), "--dry-run"])
    assert result.exit_code == 0, result.output
    printed = result.output.replace("\n", " ")  # rich soft-wraps the line
    assert "-m pip install" in printed
    assert "'pandas>=2'" in printed  # unquoted, a paste would redirect into a file named 2
    # meridian declares no requirements of its own, so there is nothing to skip
    assert "meridian" not in printed


def _cartridge(**envs: EnvSpec) -> Cartridge:
    return Cartridge(
        name="t", root=Path("."), personas={}, skills={}, workflows={}, envs=envs
    )


def test_install_targets_kernel_envs_only():
    c = _cartridge(
        default=EnvSpec(name="default", kind="kernel", requirements=["pandas>=2"]),
        box=EnvSpec(name="box", kind="docker", image="x", requirements=[ABSENT]),
        bare=EnvSpec(name="bare", kind="kernel"),
    )
    # docker deps belong in the Dockerfile; an env with nothing declared is not a target
    assert [e.name for e in _kernel_envs(c)] == ["default"]
