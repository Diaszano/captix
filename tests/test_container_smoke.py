"""Deployment checks; Docker tests use temporary synthetic ONNX, never real OCR."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def command(*args, timeout=60, check=True, **kwargs):
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, check=False, **kwargs
    )
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_runtime_imports_without_training_packages():
    result = command(
        sys.executable,
        "-c",
        """
import importlib.abc
import sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'captix', 'huggingface_hub'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockTraining())
import captix_runtime
import service.main
""",
        cwd=ROOT,
    )
    assert result.returncode == 0


def test_runtime_build_has_no_training_install_or_startup_download():
    dockerfile = ROOT / "deploy/Dockerfile"
    assert dockerfile.is_file(), "offline runtime Dockerfile is missing"
    text = dockerfile.read_text().lower()
    assert "--require-hashes" in text
    assert "pip install ." not in text
    assert "git clone" not in text
    assert "src/captix " not in text
    assert "src/captix/" not in text
    startup = "\n".join(
        line for line in text.splitlines() if line.startswith(("cmd", "entrypoint"))
    )
    assert startup
    assert not any(word in startup for word in ("pip", "curl", "wget", "download"))
    locked = (ROOT / "deploy/requirements-runtime.txt").read_text()
    assert "--hash=sha256:" in locked
    assert not re.search(
        r"^(captix|torch|torchvision|onnx|httpx)==", locked, re.MULTILINE
    )


@pytest.fixture(scope="module")
def docker():
    if not shutil.which("docker"):
        pytest.skip("Docker binary unavailable; container behavior is unverified")
    result = command("docker", "info", timeout=20, check=False)
    if result.returncode:
        pytest.skip(
            "Docker daemon unavailable; container behavior is unverified: "
            + result.stderr
        )
    return "docker"


@pytest.mark.integration
def test_compose_applies_resource_and_exposure_limits(docker):
    result = command(
        docker,
        "compose",
        "-f",
        str(ROOT / "deploy/compose.yaml"),
        "config",
        "--format",
        "json",
        env={**os.environ, "CAPTIX_API_TOKEN": "synthetic-smoke-token"},
    )
    service = json.loads(result.stdout)["services"]["captix"]
    assert service["platform"] == "linux/amd64"
    assert service["user"] == "10001:10001"
    assert service["read_only"] is True
    assert float(service["cpus"]) == 2.0
    assert int(service["mem_limit"]) == 1024**3
    assert service["environment"]["CAPTIX_METRICS_ENABLED"] == "false"
    assert service["ports"][0]["host_ip"] == "127.0.0.1"
    assert len(service["ports"]) == 1
    assert service["tmpfs"] == ["/tmp:rw,noexec,nosuid,size=16m,mode=1777"]
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]


def synthetic_artifact(directory):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    directory.mkdir()
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["images", "weights"], ["features"], strides=[32, 4]
            ),
            helper.make_node("Squeeze", ["features", "axes"], ["squeezed"]),
            helper.make_node("Transpose", ["squeezed"], ["transposed"], perm=[1, 0]),
            helper.make_node("Unsqueeze", ["transposed", "batch"], ["logits"]),
        ],
        "synthetic-container-smoke-only",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, "W"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["T", 1, 63])],
        [
            numpy_helper.from_array(np.zeros((63, 3, 32, 4), np.float32), "weights"),
            numpy_helper.from_array(np.array([0, 2], np.int64), "axes"),
            numpy_helper.from_array(np.array([1], np.int64), "batch"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")
    manifest = {
        "schema_version": 1,
        "model_version": "captix-v1-onnx-fp32",
        "exporter": "torchscript",
        "opset": 18,
        "model_sha256": hashlib.sha256(
            (directory / "model.onnx").read_bytes()
        ).hexdigest(),
        "source_commit": "a" * 40,
        "revision_sha": "b" * 40,
        "source_sha256": {"model.safetensors": "c" * 64, "config.json": "d" * 64},
        "versions": {
            name: "synthetic-test-only"
            for name in ("torch", "onnx", "onnxscript", "onnxruntime")
        },
        "height": 32,
        "input_name": "images",
        "output_name": "logits",
        "dtype": "float32",
        "vocab": list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "blank_index": 0,
        "width_range": [4, 1024],
        "input_scale": [0, 1],
        "input_layout": "NCHW",
        "output_layout": "TBC",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture(scope="module")
def image(docker, tmp_path_factory):
    context = tmp_path_factory.mktemp("container-context")
    for source in ("deploy", "service", "src/captix_runtime"):
        shutil.copytree(
            ROOT / source,
            context / source,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    shutil.copy2(ROOT / ".dockerignore", context / ".dockerignore")
    synthetic_artifact(context / "models")
    # These must never enter the image, even when present in the build context.
    (context / ".env").write_text("PRIVATE_SENTINEL=must-not-ship")
    (context / "models/private.pt").write_bytes(b"private checkpoint")
    tag = "captix:smoke-" + uuid.uuid4().hex
    try:
        command(
            docker,
            "build",
            "--platform",
            "linux/amd64",
            "-f",
            "deploy/Dockerfile",
            "-t",
            tag,
            ".",
            cwd=context,
            timeout=900,
        )
        yield tag, context
    finally:
        command(docker, "image", "rm", "-f", tag, check=False)


def start_container(docker, image, *extra):
    return command(
        docker,
        "run",
        "-d",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m,mode=1777",
        "--cpus",
        "2",
        "--memory",
        "1g",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        *extra,
        image,
    ).stdout.strip()


@pytest.mark.integration
def test_offline_container_health_uid_upload_and_sigterm(docker, image):
    tag, _ = image
    metadata = json.loads(command(docker, "image", "inspect", tag).stdout)[0]
    assert metadata["Architecture"] == "amd64"
    assert metadata["Os"] == "linux"
    assert metadata["Config"]["User"] == "10001:10001"
    assert "--workers" in metadata["Config"]["Cmd"]
    assert (
        metadata["Config"]["Cmd"][metadata["Config"]["Cmd"].index("--workers") + 1]
        == "1"
    )
    container = start_container(
        docker, tag, "-e", "CAPTIX_API_TOKEN=synthetic-smoke-token"
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = json.loads(command(docker, "inspect", container).stdout)[0]["State"]
            assert state["Running"], command(docker, "logs", container).stdout
            if state.get("Health", {}).get("Status") == "healthy":
                break
            time.sleep(0.5)
        else:
            pytest.fail(
                "container did not become healthy: "
                + command(docker, "logs", container).stdout
            )
        command(
            docker,
            "exec",
            container,
            "python",
            "-c",
            """
import importlib.util
import os
from pathlib import Path
assert os.getuid() == 10001
for name in ('captix', 'torch', 'torchvision', 'onnx', 'huggingface_hub'):
    assert importlib.util.find_spec(name) is None, name
import captix_runtime
import service.main
assert not Path('/app/.env').exists()
assert sorted(p.name for p in Path('/app/models').iterdir()) == ['manifest.json', 'model.onnx']
try:
    Path('/app/write-probe').write_text('blocked')
except OSError:
    pass
else:
    raise AssertionError('root filesystem is writable')
""",
        )
        command(
            docker,
            "exec",
            container,
            "python",
            "-c",
            """
import urllib.request
from urllib.error import HTTPError
assert urllib.request.urlopen('http://127.0.0.1:8000/health/ready').status == 200
try:
    urllib.request.urlopen('http://127.0.0.1:8000/metrics')
except HTTPError as exc:
    assert exc.code == 404
else:
    raise AssertionError('metrics exposed')
body = (b'--smoke\\r\\nContent-Disposition: form-data; name="image"; filename="x.png"\\r\\n'
        b'Content-Type: image/png\\r\\n\\r\\n' + b'x' * (5 * 1024 * 1024 + 1) + b'\\r\\n--smoke--\\r\\n')
request = urllib.request.Request('http://127.0.0.1:8000/v1/ocr', data=body, headers={
    'Authorization': 'Bearer synthetic-smoke-token', 'Content-Type': 'multipart/form-data; boundary=smoke'})
try:
    urllib.request.urlopen(request)
except HTTPError as exc:
    assert exc.code == 413, exc.code
else:
    raise AssertionError('oversized upload accepted')
""",
        )
        command(docker, "kill", "--signal", "TERM", container)
        assert command(docker, "wait", container, timeout=20).stdout.strip() == "0"
    finally:
        command(docker, "rm", "-f", container, check=False)


@pytest.mark.integration
@pytest.mark.parametrize("defect", ["tampered", "missing", "no-token"])
def test_container_fails_closed(docker, image, tmp_path, defect):
    tag, context = image
    extra = []
    if defect != "no-token":
        extra = ["-e", "CAPTIX_API_TOKEN=synthetic-smoke-token"]
        if defect == "tampered":
            shutil.copytree(context / "models", tmp_path / "models")
            (tmp_path / "models/model.onnx").write_bytes(b"tampered")
        else:
            (tmp_path / "models").mkdir()
        extra += [
            "--mount",
            f"type=bind,src={tmp_path / 'models'},dst=/app/models,readonly",
        ]
    container = start_container(docker, tag, *extra)
    try:
        assert command(docker, "wait", container, timeout=30).stdout.strip() != "0"
        logs = command(docker, "logs", container)
        expected = {
            "tampered": "SHA256 mismatch",
            "missing": "Cannot read manifest.json",
            "no-token": "CAPTIX_API_TOKEN must not be empty",
        }[defect]
        assert expected in logs.stdout + logs.stderr
    finally:
        command(docker, "rm", "-f", container, check=False)
