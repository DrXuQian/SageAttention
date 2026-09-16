"""Native-runtime link authority, independent of Torch and device execution."""

from pathlib import Path
import re
import subprocess


def runtime_soname(library):
    text = subprocess.check_output(["readelf", "-d", str(library)], text=True)
    names = re.findall(r"\(SONAME\).*?\[(.*?)\]", text)
    if len(names) != 1 or not re.fullmatch(r"libhggcrt\.\d+\.\d+\.so", names[0]):
        raise RuntimeError(f"not a versioned native HGGC runtime: {library}: {names}")
    return names[0]


def wrapped_symbols(source):
    names = re.findall(r"^SAGE_RUNTIME_FORWARD\((\w+),", Path(source).read_text(), re.M)
    if not names or len(names) != len(set(names)):
        raise RuntimeError("empty or duplicated native-runtime forwarding contract")
    return names


def check_native_linkage(extension, soname):
    text = subprocess.check_output(["readelf", "-d", str(extension)], text=True)
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", text)
    if soname not in needed or any("wrapper" in item for item in needed):
        raise RuntimeError(f"PPU extension must depend directly on {soname}, not SDK shims: {needed}")
    undefined = subprocess.check_output(["nm", "-D", "-u", str(extension)], text=True)
    unbound = re.findall(r"\b[Uw]\s+((?:__)?hggc\w*)(?:@\S+)?", undefined)
    if unbound:
        raise RuntimeError(f"native PPU entries remain open to global interposition: {unbound}")
    return {"soname": soname, "elf_needed": needed, "unbound_native_entries": unbound}
