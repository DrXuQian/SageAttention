"""PC-preserving parser for the supported hgobjdump kernel control flow.

Bound every kernel at the next ELF section, then follow actual branch targets.
An early exit does not discard a live target after it. Unreachable return
padding and text from another ELF can never become instructions of this kernel.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Instruction:
    pc: int
    opcode: str
    operands: str


@dataclass
class Kernel:
    name: str
    instructions: dict[int, Instruction]
    edges: dict[int, set[int]]
    entry: int
    discarded_instructions: int


def parse_native(text):
    headers = list(re.finditer(r"^Disassembly of section ([^:\n]+):", text, re.M))
    result = {}
    for index, header in enumerate(headers):
        if not header[1].startswith(".text.kernel."):
            continue
        name = header[1][len(".text.kernel."):]
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        body = text[header.end():end]
        labels = {m[2]: int(m[1], 16) for m in re.finditer(r"^([0-9a-f]+) <([^>]+)>:", body, re.M)}
        instructions = {}
        for line in body.splitlines():
            match = re.match(r"\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){8}\s*(\S+)(.*)", line)
            if not match:
                continue
            pc = int(match[1], 16)
            if pc in instructions:
                raise ValueError(f"duplicate PC {pc:x} in {name}")
            instructions[pc] = Instruction(pc, match[2], match[3].strip())
        if not instructions or name in result:
            raise ValueError(f"empty or duplicate kernel: {name}")
        pcs = list(instructions)
        if pcs != sorted(pcs):
            raise ValueError("kernel PCs must be ordered")
        edges = {pc: set() for pc in pcs}
        exits = {"s.exit", "simt.exit"}
        branches = {"s.cbr", "s.cbr.az", "s.cbr.nz"}
        for position, pc in enumerate(pcs):
            op, args = instructions[pc].opcode, instructions[pc].operands
            if op.startswith("s.cbr") and op not in branches:
                raise ValueError(f"unmodelled branch: {op}")
            if op.startswith("simt.") and op not in exits:
                raise ValueError(f"unmodelled SIMT control flow: {op}")
            if op in branches:
                target = re.search(r"<([^>]+)>", args)
                if target is None or target[1] not in labels or labels[target[1]] not in instructions:
                    raise ValueError(f"unresolved branch target: {name} {pc:x}")
                edges[pc].add(labels[target[1]])
            if op not in exits | {"s.cbr"} and position + 1 < len(pcs):
                edges[pc].add(pcs[position + 1])
        reachable, pending = set(), [pcs[0]]
        while pending:
            pc = pending.pop()
            if pc not in reachable:
                reachable.add(pc)
                pending.extend(edges[pc])
        if not any(instructions[pc].opcode in exits for pc in reachable):
            raise ValueError(f"kernel has no reachable exit: {name}")
        result[name] = Kernel(name, {pc: row for pc, row in instructions.items() if pc in reachable},
                              {pc: edges[pc] for pc in instructions if pc in reachable}, pcs[0],
                              len(instructions) - len(reachable))
    return result
