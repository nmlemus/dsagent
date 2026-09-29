"""Server-mode trust boundary and the non-executing filesystem provider."""

import pytest

from dsagent.cartridge.models import EnvSpec
from dsagent.envs.filesystem import make_filesystem_env
from dsagent.serve import validate_serve_environments
from tests.fakes import tiny_cartridge


def test_filesystem_provider_is_workspace_scoped_and_cannot_execute(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    env = make_filesystem_env(EnvSpec(name="default", kind="filesystem"), workspace)

    assert env.backend.write("/allowed.txt", "inside").path == "/allowed.txt"
    assert (workspace / "allowed.txt").read_text() == "inside"
    assert not hasattr(env.backend, "execute")
    with pytest.raises(ValueError, match="traversal"):
        env.backend.read("/../outside.txt")


def test_server_rejects_every_host_kernel_by_default(tmp_path):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    with pytest.raises(ValueError, match="serve refuses host kernels"):
        validate_serve_environments([cart], allow_host_compute=False)


def test_insecure_local_is_an_explicit_development_escape_hatch(tmp_path):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    validate_serve_environments([cart], allow_host_compute=True)


def test_server_accepts_non_executing_and_container_environments(tmp_path):
    cart = tiny_cartridge(
        tmp_path / "cart",
        [{"id": "work"}],
        envs={
            "default": {"kind": "filesystem"},
            "compute": {"kind": "docker", "image": "example@sha256:abc"},
        },
    )
    validate_serve_environments([cart], allow_host_compute=False)


def test_server_refuses_building_third_party_images_during_a_request(tmp_path):
    build = tmp_path / "cart" / "image"
    build.mkdir(parents=True)
    (build / "Dockerfile").write_text("FROM scratch")
    cart = tiny_cartridge(
        tmp_path / "cart",
        [{"id": "work"}],
        envs={"default": {"kind": "docker", "build": "image"}},
    )
    with pytest.raises(ValueError, match="request-time image builds"):
        validate_serve_environments([cart], allow_host_compute=False)
