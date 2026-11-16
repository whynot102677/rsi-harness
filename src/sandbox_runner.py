"""Execute Python source and MBPP checks only inside a restricted Docker container."""

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass

# The trusted controller reports test counts. Generated code runs in child
# interpreters with closed output streams and a fresh namespace per check.
WORKER = """
import ast, inspect, json, os, sys
job = json.load(sys.stdin)
try:
    source = compile(job["source"], "<candidate>", "exec")
except SyntaxError:
    os._exit(11)
scope = {"__name__": "candidate"}
try:
    exec(job["setup"], scope)
    exec(source, scope)
except BaseException:
    os._exit(12)
test = job["test"]
if name := job.get("diagnose_name"):
    function = scope.get(name)
    if not callable(function):
        os._exit(21)
    boundary = []
    def check_call(*args, **kwargs):
        try:
            inspect.signature(function).bind(*args, **kwargs)
        except TypeError:
            os._exit(21)
        except ValueError:
            pass
        boundary.append(any(
            (type(value) in (str, list, tuple, dict, set) and not value)
            or (type(value) in (int, float) and value == 0)
            for value in (*args, *kwargs.values())
        ))
        return function(*args, **kwargs)
    def check_equal(actual, expected):
        if actual == expected:
            return True
        if any(boundary):
            os._exit(22)
        os._exit(23)
    scope[name] = check_call
    scope["_rsi_equal"] = check_equal
    tree = ast.parse(test)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
                and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)):
            comparison = node.test
            node.test = ast.Call(ast.Name("_rsi_equal", ast.Load()),
                                 [comparison.left, comparison.comparators[0]], [])
    test = compile(ast.fix_missing_locations(tree), "<diagnostic>", "exec")
try:
    exec(test, scope)
except AssertionError:
    os._exit(13)
except BaseException:
    os._exit(12)
os._exit(10)
"""

CONTROLLER = f"""
import json, subprocess, sys
job = json.load(sys.stdin)
checks = []
for test in job["tests"]:
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", {WORKER!r}],
        input=json.dumps({{"source": job["source"], "setup": job["setup"], "test": test,
                           "diagnose_name": job.get("diagnose_name")}}),
        text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    checks.append(result.returncode)
    if result.returncode == 11:
        break
print(json.dumps(checks))
"""

RESTRICTIONS = [
    "--network", "none", "--user", "65534:65534", "--read-only",
    "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true",
    "--cpus", "1", "--memory", "128m", "--memory-swap", "128m",
    "--pids-limit", "32", "--ulimit", "nofile=64:64", "--ipc", "none", "--cgroupns", "private",
    "--log-driver", "none", "--pull", "never", "--entrypoint", "python",
]

PROBE = """
import os
import socket
from pathlib import Path
assert os.geteuid() == os.getegid() == 65534
assert os.statvfs("/").f_flag & os.ST_RDONLY
# The Linux kernel can expose inactive tunnel devices even with --network none.
interfaces_up = {
    name for _, name in socket.if_nameindex()
    if int((Path("/sys/class/net") / name / "flags").read_text(), 16) & 1
}
assert interfaces_up == {"lo"}
ipv4_routes = Path("/proc/net/route").read_text().splitlines()[1:]
ipv6_routes = Path("/proc/net/ipv6_route").read_text().splitlines()
assert all(route.split()[0] == "lo" for route in ipv4_routes)
assert all(route.split()[-1] == "lo" for route in ipv6_routes)
assert "DEEPSEEK_API_KEY" not in os.environ
assert not os.path.exists("/var/run/docker.sock")
with open("/proc/self/status") as stream:
    status = dict(line.split(":", 1) for line in stream if ":" in line)
assert int(status["CapEff"].strip(), 16) == 0
assert status["NoNewPrivs"].strip() == "1"
cgroups = Path("/sys/fs/cgroup")
quota, period = map(int, (cgroups / "cpu.max").read_text().split())
assert quota == period
assert (cgroups / "memory.max").read_text().strip() == str(128 * 1024 * 1024)
assert (cgroups / "memory.swap.max").read_text().strip() == "0"
assert (cgroups / "pids.max").read_text().strip() == "32"
"""


class SandboxProblem(Exception):
    pass


@dataclass
class Evaluation:
    execution_status: str
    tests_passed: int | None
    failure_type: str | None
    execution_time: float


class Sandbox:
    def __init__(self, image, timeout):
        self.image = image
        self.timeout = timeout
        self.image_id = None
        self.endpoint = None
        self.docker = shutil.which("docker")
        self.env = {k: v for k, v in os.environ.items() if k not in (
            "DEEPSEEK_API_KEY", "DOCKER_HOST", "DOCKER_CONTEXT"
        )}

    def _command(self, args, *, timeout=10, input_text=None):
        connection = ["--host", self.endpoint] if self.endpoint else []
        return subprocess.run(
            [self.docker, *connection, *args], input=input_text, text=True, capture_output=True,
            timeout=timeout, env=self.env,
        )

    def prepare(self):
        if not self.docker:
            raise SandboxProblem("Docker CLI is unavailable")
        try:
            endpoint = self._command([
                "context", "inspect", "--format", "{{.Endpoints.docker.Host}}",
            ])
            if endpoint.returncode or not endpoint.stdout.strip().startswith("unix://"):
                raise SandboxProblem("Select a local Docker context backed by a Unix socket")
            self.endpoint = endpoint.stdout.strip()
            engine = self._command(["info", "--format", "{{.OSType}}"])
            if engine.returncode or engine.stdout.strip() != "linux":
                raise SandboxProblem("A running Linux Docker Engine is required")
            image = self._command(["image", "inspect", "--format", "{{.Id}}", self.image])
            image_id = image.stdout.strip()
            if image.returncode or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
                raise SandboxProblem(f"Local Docker image unavailable. Pull {self.image} first")
        except (OSError, subprocess.TimeoutExpired):
            raise SandboxProblem("Docker preflight failed or timed out") from None

        self.image_id = image_id
        result = self.evaluate("def add(a, b): return a + b", "", [PROBE + "\nassert add(2, 3) == 5"])
        if result.execution_status != "passed":
            self.image_id = None
            raise SandboxProblem(f"Docker isolation probe failed ({result.failure_type})")
        return image_id

    def evaluate(self, source, setup, tests, *, diagnose_name=None):
        started = time.monotonic()

        def result(status, passed=None, failure=None):
            return Evaluation(status, passed, failure, round(time.monotonic() - started, 3))

        if not self.image_id:
            return result("blocked", failure="sandbox_initialization_failure")

        name = f"rsi-p0-{uuid.uuid4().hex}"
        payload = json.dumps({"source": source, "setup": setup, "tests": tests, "diagnose_name": diagnose_name})
        try:
            created = self._command([
                "create", "-i", "--name", name, *RESTRICTIONS, self.image_id,
                "-I", "-B", "-c", CONTROLLER,
            ])
        except subprocess.TimeoutExpired:
            self._remove(name, missing_ok=True)
            return result("blocked", failure="sandbox_initialization_failure")
        except OSError:
            return result("blocked", failure="sandbox_initialization_failure")
        if created.returncode:
            self._remove(name, missing_ok=True)
            return result("blocked", failure="sandbox_initialization_failure")

        try:
            completed = self._command(["start", "-ai", name], timeout=self.timeout, input_text=payload)
        except subprocess.TimeoutExpired:
            return result("failed", failure="timeout")
        except OSError:
            return result("blocked", failure="sandbox_initialization_failure")
        finally:
            self._remove(name)

        if completed.returncode:
            return result("blocked", failure="sandbox_initialization_failure")
        try:
            checks = json.loads(completed.stdout)
            if not isinstance(checks, list) or not all(type(code) is int for code in checks):
                raise ValueError
            if checks == [11]:
                return result("failed", 0, "syntax_error")
            if len(checks) != len(tests):
                raise ValueError
        except ValueError:
            return result("failed", failure="sandbox_protocol_error")
        passed = checks.count(10)
        if passed == len(tests):
            return result("passed", passed)
        if diagnose_name:
            diagnostic_categories = {21: "signature", 22: "boundary", 23: "output"}
            categories = {diagnostic_categories.get(code, "assertion_failure" if code == 13 else "runtime_error")
                          for code in checks if code != 10}
            if len(categories) == 1:
                return result("failed", passed, categories.pop())
            return result("failed", passed, "mixed")
        failure = "runtime_error" if any(code not in (10, 13) for code in checks) else "assertion_failure"
        return result("failed", passed, failure)

    def _remove(self, name, *, missing_ok=False):
        try:
            removed = self._command(["rm", "-f", name])
        except (OSError, subprocess.TimeoutExpired):
            raise SandboxProblem(f"Docker cleanup could not be verified for {name}") from None
        if removed.returncode and not (missing_ok and "No such container" in removed.stderr):
            raise SandboxProblem(f"Docker cleanup could not be verified for {name}")
