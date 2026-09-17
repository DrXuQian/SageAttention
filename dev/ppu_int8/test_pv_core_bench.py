#!/usr/bin/env python3
"""Offline discrimination tests; no device timing asserted."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import benchmark_ppu_pv_core as bench


class CoreBench(unittest.TestCase):
    def test_alternate_same_two_arms(self):
        self.assertEqual(bench.roles_for_sample(0), ("fp16-pv", "int8-pv"))
        self.assertEqual(bench.roles_for_sample(1), ("int8-pv", "fp16-pv"))

    def test_faster_and_slower_are_both_reported(self):
        fast, slow = bench.summary([10, 11, 12]), bench.summary([15, 16, 17])
        self.assertEqual(bench.verdict(slow, fast), "INT8-FASTER")
        self.assertEqual(bench.verdict(fast, slow), "FP16-FASTER")

    def test_overlapping_or_touching_envelopes_unresolved(self):
        a, b = bench.summary([10, 12, 15]), bench.summary([11, 12, 14])
        self.assertEqual(bench.verdict(a, b), "UNRESOLVED")
        self.assertEqual(bench.verdict(a, bench.summary([15, 16, 17])), "UNRESOLVED")

    def test_invalid_measurement_is_not_skip_or_speedup(self):
        for values in ([], [1], [1, 2, 0], [1, 2, float("nan")], [1, 2, float("inf")]):
            with self.assertRaises(ValueError):
                bench.summary(values)

    def test_different_native_entrypoints_shared_qk_no_prepare(self):
        native = Mock()
        tensors = {name: object() for name in
                   ("qi", "ki", "vi", "vf", "qs", "ks", "vs", "int8-pv", "fp16-pv")}
        bench.launch_core("int8-pv", native, tensors, False, .125)
        bench.launch_core("fp16-pv", native, tensors, False, .125)
        calls = native.mock_calls
        self.assertEqual([call[0] for call in calls],
                         ["qk_int8_sv_int8_accum_f32_attn", "qk_int8_sv_f16_accum_f32_attn"])
        self.assertEqual(calls[0].args[:2], calls[1].args[:2])
        self.assertIs(calls[0].args[2], tensors["vi"])
        self.assertIs(calls[1].args[2], tensors["vf"])
        self.assertEqual(calls[0].args[-5:], (0, 0, 2, .125, 0))
        self.assertEqual(calls[1].args[-5:], (0, 0, 2, .125, 0))
        with self.assertRaises(ValueError):
            bench.launch_core("typo", native, tensors, False, .125)

    def test_plan_requires_no_torch_device(self):
        self.assertEqual(bench.main(["--describe"]), 0)

    def test_permuted_entrypoint_is_explicit(self):
        native = Mock()
        names = ("qi", "kp", "vi", "int8-pv-permuted-k", "qs", "kps", "vs")
        t = {name: object() for name in names}
        bench.launch_core("int8-pv-permuted-k", native, t, False, .125)
        call, = native.mock_calls
        self.assertEqual(call[0], "qk_int8_sv_int8_permuted_k_attn")
        self.assertIs(call.args[1], t["kp"])
        self.assertIs(call.args[5], t["kps"])

    def test_moved_prepare_cost_is_not_hidden(self):
        self.assertIn("K-quant-raw", bench.PERMUTED_ROLES)
        self.assertIn("K-quant-permuted", bench.PERMUTED_ROLES)
        self.assertIn("K-prepare+int8-pv", bench.PERMUTED_ROLES)
        self.assertIn("K-prepare+int8-pv-permuted-k", bench.PERMUTED_ROLES)
        for i in range(len(bench.PERMUTED_ROLES)):
            self.assertEqual(set(bench.roles_for_sample(i, bench.PERMUTED_ROLES)), set(bench.PERMUTED_ROLES))


if __name__ == "__main__":
    unittest.main()
