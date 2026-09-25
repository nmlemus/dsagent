"""Local process provider and out-of-process guardian. No graph dependencies."""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import psutil

from dsagent.execution_storage import durable_json

TERMINAL = {"exited", "cancelled", "timed_out", "interrupted", "cleanup_required"}
PROCESS_DIR_ENV = "DSAGENT_PROCESS_DIR"
DEFAULT_STEP_SECONDS = 3600.0


def seconds(value: str | None) -> float:
    if value is None:
        return DEFAULT_STEP_SECONDS
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", value)
    if not match:
        raise ValueError("timeout must be a positive duration, e.g. 30s, 5m or 1h")
    result = float(match[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[match[2]]
    if not math.isfinite(result) or result <= 0:
        raise ValueError("timeout must be positive and finite")
    return result


def identity(pid: int) -> dict:
    return {"pid": pid, "created": psutil.Process(pid).create_time(),
            "boot": psutil.boot_time(), "host": socket.gethostname()}


def matches(record: dict | None) -> bool:
    if (not record or record.get("boot") != psutil.boot_time()
            or record.get("host") != socket.gethostname()):
        return False
    try:
        process = psutil.Process(record["pid"])
        return process.create_time() == record["created"] and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def reference(factory) -> str:
    name = f"{factory.__module__}:{factory.__qualname__}"
    if "<" in name or resolve(name) is not factory:
        raise ValueError("process factories must be importable top-level functions")
    return name


def resolve(name: str):
    module, qualname = name.split(":", 1)
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def kill_group(record: dict, *, grace: float = .5) -> None:
    """Refuse a reused leader PID. Signal descendants even if the leader exited."""
    if record.get("host") != socket.gethostname():
        raise RuntimeError("process belongs to a different host; cannot verify cleanup")
    if record.get("boot") != psutil.boot_time():
        return
    pid = record["pid"]
    try:
        if psutil.Process(pid).create_time() != record["created"]:
            raise RuntimeError(f"process identity changed for pid {pid}; manual reconciliation required")
    except psutil.NoSuchProcess:
        pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not group_alive(pid):
                return
            time.sleep(.05)
    if group_alive(pid):
        raise RuntimeError(f"process group {pid} did not terminate")


def group_alive(group: int) -> bool:
    for p in psutil.process_iter(["pid", "status"]):
        try:
            if p.info["status"] != psutil.STATUS_ZOMBIE and os.getpgid(p.pid) == group:
                return True
        except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
            continue
    return False


def resource_intent(kind: str, **values) -> Path | None:
    root = os.environ.get(PROCESS_DIR_ENV)
    if not root:
        return None
    path = Path(root) / "resources" / f"{uuid4().hex}.json"
    durable_json(path, {"kind": kind, **values})
    return path


def cleanup_resources(root: Path) -> list[str]:
    errors = []
    for path in (root / "resources").glob("*.json"):
        try:
            rec = read(path)
            if rec["kind"] == "process":
                if not rec.get("identity"):
                    raise RuntimeError("resource provisioning interrupted before PID registration")
                kill_group(rec["identity"])
            elif rec["kind"] == "docker":
                result = subprocess.run(["docker", "inspect", rec["name"]],
                                        capture_output=True, text=True, timeout=10, check=False)
                if result.returncode:
                    if "No such object" in result.stderr or "No such container" in result.stderr:
                        continue
                    raise RuntimeError("Docker unavailable; container cleanup unverified")
                container = json.loads(result.stdout)[0]
                if (container["Config"].get("Labels") or {}).get("dsagent.execution") != rec["owner"]:
                    raise RuntimeError("container ownership mismatch")
                result = subprocess.run(["docker", "rm", "--force", container["Id"]],
                                        capture_output=True, text=True, timeout=10, check=False)
                if result.returncode:
                    raise RuntimeError("Docker refused container cleanup")
        except Exception as exc:  # noqa: BLE001 — attempt all registered resources
            errors.append(f"{path.name}: {exc}")
    return errors


class ProcessProvider(Protocol):
    def start(self, argv: list[str], root: Path, *, cwd: Path, payload: dict) -> Path: ...
    def poll(self, root: Path) -> dict: ...
    def cancel(self, root: Path, reason: str = "cancelled") -> None: ...
    def reconcile(self, root: Path) -> dict: ...


class LocalProcessProvider:
    def __init__(self):
        self._children: dict[str, subprocess.Popen] = {}

    def start(self, argv: list[str], root: Path, *, cwd: Path, payload: dict) -> Path:
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=False)
        durable_json(root / "request.json", {"argv": argv, "cwd": str(cwd.resolve()),
                                            "owner": identity(os.getpid()), "payload": payload})
        durable_json(root / "process.json", {"status": "starting", "id": root.name})
        # File output never waits for EOF from orphaned descendants.
        with (root / "process.log").open("ab") as output:
            child = subprocess.Popen([sys.executable, "-m", "dsagent.processes", "guard", str(root)],
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                     start_new_session=True)
        # Retain/reap the guardian without letting Popen's destructor own it.
        self._children[str(root)] = child
        # Reap even when the last poll observes the terminal record just before
        # the guardian exits. No growing collection of zombie guardians.
        def reap():
            child.wait()
            if read(root / "process.json")["status"] in TERMINAL:
                self._children.pop(str(root), None)
        threading.Thread(target=reap, daemon=True).start()
        return root

    def poll(self, root: Path) -> dict:
        rec = read(root / "process.json")
        child = getattr(self, "_children", {}).get(str(root))
        if child and child.poll() is not None and rec["status"] not in TERMINAL:
            result = self.reconcile(root)
            self._children.pop(str(root), None)
            return result
        if rec["status"] not in TERMINAL and rec.get("guardian") and not matches(rec["guardian"]):
            return self.reconcile(root)
        return rec

    def cancel(self, root: Path, reason: str = "cancelled") -> None:
        durable_json(root / "cancel.json", {"reason": reason})

    def reconcile(self, root: Path) -> dict:
        self.cancel(root, "interrupted")
        rec = read(root / "process.json")
        if matches(rec.get("guardian")):
            self.cancel(root, "interrupted")
            deadline = time.monotonic() + 30
            while matches(rec.get("guardian")) and time.monotonic() < deadline:
                time.sleep(.1)
            if matches(rec.get("guardian")):
                raise RuntimeError("process guardian has not stopped; retry is blocked")
            rec = read(root / "process.json")
        errors = []
        if rec.get("child"):
            try:
                kill_group(rec["child"])
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        errors.extend(cleanup_resources(root))
        rec.update(status="cleanup_required" if errors else (
            rec["status"] if rec["status"] in TERMINAL - {"cleanup_required"} else "interrupted"
        ), cleanup_errors=errors)
        durable_json(root / "process.json", rec)
        return rec


def guardian(root: Path) -> None:
    request = read(root / "request.json")
    rec = {"id": root.name, "status": "starting", "guardian": identity(os.getpid()),
           "started_at": time.time()}
    durable_json(root / "process.json", rec)
    if not matches(request["owner"]) or (root / "cancel.json").exists():
        rec["status"] = "interrupted"
        durable_json(root / "process.json", rec)
        return
    env = {**os.environ, PROCESS_DIR_ENV: str(root)}
    child = subprocess.Popen([sys.executable, "-m", "dsagent.processes", "child", str(root)],
                             env=env, start_new_session=True)
    rec.update(child=identity(child.pid), status="running")
    durable_json(root / "process.json", rec)
    durable_json(root / "go.json", {"ready": True})
    reason = "exited"
    try:
        while child.poll() is None:
            if not matches(request["owner"]):
                reason = "interrupted"
                break
            if (root / "cancel.json").exists():
                reason = read(root / "cancel.json")["reason"]
                break
            deadline = root / "deadline.json"
            if not deadline.exists() and time.time() - rec["started_at"] > 60:
                reason = "timed_out"
                break
            if deadline.exists():
                value = read(deadline)
                if value.get("expires") and time.time() >= value["expires"]:
                    reason = "timed_out"
                    rec["step"] = value.get("step")
                    break
            time.sleep(.1)
    finally:
        rec.update(status="cancelling", reason=reason)
        durable_json(root / "process.json", rec)
        errors = []
        try:
            kill_group(rec["child"])
            child.wait(timeout=3)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        errors.extend(cleanup_resources(root))
        rec.update(status="cleanup_required" if errors else reason,
                   cleanup_errors=errors, exit_code=child.returncode, finished_at=time.time())
        durable_json(root / "process.json", rec)


def child_entry(root: Path) -> None:
    # No domain code before the guardian durably records ownership. If it dies
    # in the spawn/record gap, this child exits without executing the payload.
    parent = identity(os.getppid())
    while not (root / "go.json").exists():
        if not matches(parent):
            return
        time.sleep(.05)
    if not matches(parent):
        return
    request = read(root / "request.json")
    os.chdir(request["cwd"])
    os.execvpe(request["argv"][0], request["argv"], os.environ)


if __name__ == "__main__":
    (guardian if sys.argv[1] == "guard" else child_entry)(Path(sys.argv[2]))
