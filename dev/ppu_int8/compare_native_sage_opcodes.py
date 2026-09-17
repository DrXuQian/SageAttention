#!/usr/bin/env python3
"""Compare measured warp instructions, not latency or equal-precision accuracy.

Inputs: NCU raw CSV with --print-metric-instances details, and the ACU
per-PC JSON exported with warp_executed_count. Counts must close before any
normalization. All integer/bit/select membership is explicit below.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re


NV_INTEGER = {
    "IADD", "IADD3", "IMAD", "LEA", "LOP3", "SHF", "PRMT", "ISETP", "ISET",
    "PLOP3", "UIADD3", "UIMAD", "ULEA", "ULOP3", "USHF", "UISETP", "SEL", "FSEL",
}
PPU_INTEGER = re.compile(
    r"^[vs]\.(add|sub|mul|madl|madw|shll|shrl|shra|and|or|xor|xnor|lop3|bfi|bfe|"
    r"min|max|cmp|ucmp|csel|pcnvt)(?:\.|$)"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instances(text):
    match = re.fullmatch(r"(\d+) \((.*)\)", text)
    if not match:
        raise ValueError("NCU correlation IDs required: use --print-metric-instances details")
    result = {}
    for item in match[2].split("; "):
        name, value = item.rsplit(": ", 1)
        if name in result:
            raise ValueError("duplicate metric correlation ID")
        result[name] = int(value)
    if sum(result.values()) != int(match[1]):
        raise ValueError("instance sum differs from aggregate")
    return result


def nvidia(path):
    with path.open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["ID"]]
    if len(rows) != 1 or "qk_int_sv_f8_attn_kernel" not in rows[0]["Kernel Name"]:
        raise ValueError("expected exactly one native FP8 attention launch")
    row = rows[0]
    counts = instances(row["sass__inst_executed_per_opcode_with_modifier_all"])
    pcs = instances(row["inst_executed"])
    if sum(pcs.values()) != sum(counts.values()):
        raise ValueError("NCU PC/opcode denominator differs")
    return counts, dict(kernel=row["Kernel Name"], block=row["Block Size"],
                       grid=row["Grid Size"], regs=row["launch__registers_per_thread"],
                       stack_attribute=row["launch__stack_size"], source_sha256=sha(path))


def ppu(path):
    rows = json.loads(path.read_text())
    if len(rows) != 1 or "qk_int8_pv_kernel<128, false, false" not in rows[0]["name"]:
        raise ValueError("expected exactly one PPU D128 full-attention kernel")
    counts = Counter()
    seen = set()
    for row in rows[0]["instructions"]:
        if row["pc"] in seen or row["executed"] < 0:
            raise ValueError("duplicate PC or negative execution count")
        seen.add(row["pc"])
        counts[row["sass"].split()[0]] += row["executed"]
    return dict(counts), dict(kernel=rows[0]["name"], source_sha256=sha(path),
                             denominator="ACU per-PC warp counts, not the separate aggregate counter")


def summary(counts, kind, visits):
    if kind == "nvidia":
        is_integer = lambda op: op.split(".")[0] in NV_INTEGER
        is_shuffle = lambda op: op.startswith("SHFL.")
        is_convert = lambda op: op.split(".")[0] in {"F2FP", "I2FP", "I2F", "F2I", "F2F", "I2I"}
        is_mma = lambda op: op.startswith(("IMMA.", "QMMA.", "HMMA."))
        is_nop = lambda op: op == "NOP"
    else:
        is_integer = lambda op: bool(PPU_INTEGER.match(op)) and not re.search(r"\.(f16|f32|bf16)(\.|$)", op)
        is_shuffle = lambda op: op.startswith("v.shuffle.")
        is_convert = lambda op: op.startswith("v.cnvt.")
        is_mma = lambda op: op.startswith("v.mma.")
        is_nop = lambda op: op == "s.nop"
    groups = {"integer_bit_select": is_integer, "shuffle": is_shuffle,
              "numeric_convert": is_convert, "mma": is_mma, "nop": is_nop}
    absolute = {name: sum(v for k, v in counts.items() if predicate(k))
                for name, predicate in groups.items()}
    absolute["total"] = sum(counts.values())
    absolute["total_without_nop"] = absolute["total"] - absolute["nop"]
    return dict(absolute=absolute, per_warp_k64={k: v / visits for k, v in absolute.items()},
                opcodes=dict(sorted(counts.items())),
                integer_opcodes=[op for op in sorted(counts) if is_integer(op)])


def require_mma(counts, kind, visits):
    if kind == "nvidia":
        qk = counts.get("IMMA.16832.S8.S8", 0)
        pv = sum(v for k, v in counts.items() if k.startswith("QMMA.16832."))
        expected = visits * 64
    else:
        qk = counts.get("v.mma.i32.i8.i8.m16n16k32", 0)
        pv = counts.get("v.mma.i32.u8.i8.m16n16k32", 0)
        expected = visits * 32
    if qk != expected or pv != expected:
        raise ValueError(f"MMA work mismatch {kind}: qk={qk} pv={pv} expected={expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvidia-raw", type=Path, nargs="+", required=True)
    parser.add_argument("--ppu-sass", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Registered common geometry, independently checked against both MMA counts.
    visits = 56 * ((73774 + 127) // 128) * ((73774 + 63) // 64) * 4
    result = dict(shape="B1,H56,S73774,D128,full", warp_k64_visits=visits,
                  scope="cross-precision/device instruction reference; not latency or model quality",
                  results={})
    for path, kind, reader in [(args.ppu_sass, "ppu", ppu)] + [
            (p, "nvidia", nvidia) for p in args.nvidia_raw]:
        counts, identity = reader(path)
        require_mma(counts, kind, visits)
        record = summary(counts, kind, visits)
        record["identity"] = identity
        result["results"][path.name] = record
        print(path.name, json.dumps(record["per_warp_k64"]))
        # A missing MMA family / wrong geometry must not get a plausible ratio.
        bad = dict(counts)
        name = "IMMA.16832.S8.S8" if kind == "nvidia" else "v.mma.i32.i8.i8.m16n16k32"
        bad[name] -= 1
        try:
            require_mma(bad, kind, visits)
        except ValueError:
            pass
        else:
            raise AssertionError("missing-work negative survived")
    for text in ("3 (a: 1; b: 1)", "2 (a: 1; a: 1)"):
        try:
            instances(text)
        except ValueError:
            pass
        else:
            raise AssertionError("correlation denominator negative survived")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("[instruction reference] common matrix work and denominator negatives PASS")


if __name__ == "__main__":
    main()
