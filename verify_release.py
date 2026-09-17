"""Verify a release's payload and actual Python default without importing Torch."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def verify_default(core_source):
    tree = ast.parse(core_source)
    entry = next(node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "sageattn")
    branches = [node for node in entry.body if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name) and node.test.id == "PPU_ENABLED"]
    if len(branches) != 1 or len(branches[0].body) != 1:
        raise ValueError("unknown installed PPU dispatch structure")
    statement = branches[0].body[0]
    if not (isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "sageattn_qk_int8_pv_fp16_ppu"):
        raise ValueError("installed default does not match FP16 PV")


def verify_release(root, release=None):
    root = Path(root)
    if release is None:
        release = json.loads((root / "release.json").read_text())
    if release.get("default_pv") != "fp16" or release.get("channel") != "main":
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
        verify_default(archive.read("sageattention/core.py"))
    return release, wheel


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    release, wheel = verify_release(root)
    print(f"[wheel channel] {release['channel']} PV={release['default_pv']} {wheel.name} PASS")
    for label, changed in (
        ("precision", dict(release, default_pv="int8")),
        ("channel", dict(release, channel="experimental")),
        ("payload-hash", dict(release, wheel_sha256="0" * 64)),
    ):
        try:
            verify_release(root, changed)
        except ValueError:
            print(f"[wheel negative] {label} EXPECTED-RED/PASS")
        else:
            raise AssertionError(f"{label} mutation survived")
    with zipfile.ZipFile(wheel) as archive:
        core = archive.read("sageattention/core.py").decode()
    wrong_route = core.replace("sageattn_qk_int8_pv_fp16_ppu", "sageattn_qk_int8_pv_int8_ppu")
    if wrong_route == core:
        raise AssertionError("route negative did not mutate the real wheel source")
    try:
        verify_default(wrong_route)
    except ValueError as error:
        if "installed default" not in str(error):
            raise
        print("[wheel negative] integer-SV default in installed source EXPECTED-RED/PASS")
    else:
        raise AssertionError("wrong installed precision was admitted")
