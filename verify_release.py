"""Verify a release's payload and actual Python default without importing Torch."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def verify_release(root, experimental=False, release=None):
    root = Path(root)
    if release is None:
        name = "release-pv-int8.json" if experimental else "release.json"
        release = json.loads((root / name).read_text())
    expected_pv = "int8" if experimental else "fp16"
    if release.get("default_pv") != expected_pv:
        raise ValueError("release precision differs from the selected channel")
    filename = release["wheel"]
    if Path(filename).name != filename:
        raise ValueError("unsafe wheel filename")
    wheel = root / filename
    if not wheel.is_file() or hashlib.sha256(wheel.read_bytes()).hexdigest() != release["wheel_sha256"]:
        raise ValueError("wheel missing or SHA256 mismatch")
    with zipfile.ZipFile(wheel) as archive:
        manifest = json.loads(archive.read("sageattention/_ppu_wheel_manifest.json"))
        native = manifest["native"]
        payload = archive.read("sageattention/" + native["artifact"])
        if (hashlib.sha256(payload).hexdigest() != native["artifact_sha256"]
                or native["artifact_sha256"] != release["native_sha256"]
                or manifest["package_source_sha"] != release["package_source_sha"]):
            raise ValueError("embedded native/source identity differs from release")
        tree = ast.parse(archive.read("sageattention/core.py"))
        entry = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "sageattn")
        branches = [node for node in entry.body if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name) and node.test.id == "PPU_ENABLED"]
        if len(branches) != 1 or len(branches[0].body) != 1:
            raise ValueError("unknown installed PPU dispatch structure")
        statement = branches[0].body[0]
        if not (isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == f"sageattn_qk_int8_pv_{expected_pv}_ppu"):
            raise ValueError("installed default does not match release precision")
    return release, wheel


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    for experimental in (False, True):
        release, wheel = verify_release(root, experimental)
        print(f"[wheel channel] {release['channel']} PV={release['default_pv']} {wheel.name} PASS")
        changed = dict(release, default_pv="fp16" if experimental else "int8")
        try:
            verify_release(root, experimental, changed)
        except ValueError:
            print("[wheel negative] swapped precision EXPECTED-RED/PASS")
        else:
            raise AssertionError("precision mutation survived")
    default = json.loads((root / "release.json").read_text())
    wrong_payload = dict(default)
    experiment = json.loads((root / "release-pv-int8.json").read_text())
    for key in ("wheel", "wheel_sha256", "native_sha256", "package_source_sha"):
        wrong_payload[key] = experiment[key]
    try:
        verify_release(root, False, wrong_payload)
    except ValueError as error:
        if "installed default" not in str(error):
            raise
        print("[wheel negative] valid INT8 wheel disguised as FP16 EXPECTED-RED/PASS")
    else:
        raise AssertionError("a valid wrong-precision wheel was admitted")
