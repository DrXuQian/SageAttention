#!/usr/bin/env python3
"""CPU-only plan/timing arithmetic checks, not a PPU benchmark verdict."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bench", ROOT / "tools/benchmark_ppu_sage_bf16.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Contract(unittest.TestCase):
    def test_historical_shape(self):
        args = bench.arguments([])
        self.assertEqual((args.batch, args.heads, args.seq, args.head_dim), (1, 16, 4096, 128))

    def test_h3_shape(self):
        args = bench.arguments(["--heads", "56", "--seq", "73774", "--launches", "1"])
        self.assertEqual((args.heads, args.seq, args.launches), (56, 73774, 1))

    def test_every_round_has_every_role_once(self):
        for index in range(30):
            self.assertEqual(sorted(bench.roles_for_sample(index)), sorted(bench.ROLES))
        self.assertEqual(len({bench.roles_for_sample(i)[0] for i in range(3)}), 3)

    def test_timing_denominator(self):
        row = bench.summarize([2.0, 6.0, 4.0], 4_000_000, 500.0)
        self.assertEqual(row["median_us"], 4)
        self.assertEqual(row["logical_tflops"], 1)
        self.assertEqual(row["bf16_equivalent_mfu_percent"], 0.2)

    def test_invalid_samples_are_not_results(self):
        for values in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                bench.summarize(values, 1000, 500)

    def test_invalid_work_does_not_reach_device(self):
        for argv in (["--seq", "0"], ["--launches", "0"], ["--head-dim", "96"],
                     ["--samples", "-1"], ["--peak-bf16-tflops", "nan"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.arguments(argv)

    def test_describe_does_not_import_gpu_extensions(self):
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            self.assertEqual(bench.main(["--describe"]), 0)
        self.assertIn('"logical_flops": 137438953472', capture.getvalue())
        self.assertIn('"compile": false', capture.getvalue())


if __name__ == "__main__":
    unittest.main()
