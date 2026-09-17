#!/usr/bin/env python3
"""Parser must reject padding pollution without truncating live early-exit CFGs."""
import unittest
from native_isa import parse_native


def line(pc, op):
    return f"{pc:x}: 00 00 00 00 00 00 00 00 {op}\n"


class Boundaries(unittest.TestCase):
    def fixture(self):
        return ("Disassembly of section .text.kernel.example:\n"
                "00000000 <example>:\n" + line(0, "s.cbr.az scc, 0x1 <work>") +
                line(8, "s.exit") + "00000010 <work>:\n" +
                line(16, "v.add.i32 vreg0, vreg0, 0x1") + line(24, "s.exit") +
                line(32, "s.nop") + line(40, "s.cbr 0x0 <pad>") +
                "00000030 <pad>:\n" + line(48, "s.nop") +
                "ELF FILE 2:\nDisassembly of section .text:\n" +
                line(0, "v.mul.i32 vreg0, vreg0, vreg0"))

    def test_both_exits_and_late_live_branch(self):
        k = parse_native(self.fixture())["example"]
        self.assertEqual(list(k.instructions), [0, 8, 16, 24])
        self.assertEqual(k.discarded_instructions, 3)

    def test_simt_exit(self):
        text = "Disassembly of section .text.kernel.p:\n" + line(0, "simt.exit") + line(8, "v.add.i32 a,b,c")
        self.assertEqual(len(parse_native(text)["p"].instructions), 1)

    def test_reachable_nop_is_not_hidden(self):
        text = "Disassembly of section .text.kernel.p:\n" + line(0, "s.nop") + line(8, "s.exit")
        self.assertEqual(len(parse_native(text)["p"].instructions), 2)

    def test_next_kernel_is_independent(self):
        text = self.fixture() + "Disassembly of section .text.kernel.other:\n" + line(0, "s.exit")
        self.assertEqual({k: len(v.instructions) for k, v in parse_native(text).items()}, {"example": 4, "other": 1})

    def test_missing_target_is_red(self):
        with self.assertRaisesRegex(ValueError, "unresolved branch"):
            parse_native(self.fixture().replace("<work>", "<absent>", 1))

    def test_duplicate_pc_is_red(self):
        with self.assertRaisesRegex(ValueError, "duplicate PC"):
            parse_native(self.fixture().replace(line(24, "s.exit"), line(16, "s.exit")))

    def test_wrong_exit_legacy_parser_is_a_real_negative(self):
        text = self.fixture()
        # The historical stop-at-simt.exit-only rule includes the late padding
        # and even the foreign text section. Its 8 records are NOT a valid 4.
        legacy_count = sum(": 00 00" in row for row in text.splitlines())
        self.assertEqual(legacy_count, 8)
        self.assertNotEqual(legacy_count, len(parse_native(text)["example"].instructions))


if __name__ == "__main__":
    unittest.main()
