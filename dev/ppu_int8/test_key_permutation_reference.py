#!/usr/bin/env python3
"""Independent whole-tensor reference covers all K64 tails and layout strides."""
import unittest
import torch

from key_permutation_reference import permute_key_reference


class KeyPermutation(unittest.TestCase):
    def test_every_tail_batch_head_and_dimension(self):
        cases = cells = 0
        for n in range(1, 193):
            for d in (64, 128):
                source = torch.arange(2 * 3 * n * d).reshape(2, 3, n, d)
                for layout in ("HND", "NHD"):
                    x = source if layout == "HND" else source.transpose(1, 2).contiguous()
                    y = permute_key_reference(x, layout)
                    self.assertTrue(torch.equal(permute_key_reference(y, layout), x))
                    axis = 2 if layout == "HND" else 1
                    full = n // 64 * 64
                    tail = slice(full, None)
                    sl = [slice(None)] * 4
                    sl[axis] = tail
                    self.assertTrue(torch.equal(y[tuple(sl)], x[tuple(sl)]))
                    cases += 1
                    cells += x.numel()
        self.assertEqual(cases, 768)
        self.assertEqual(cells, 42688512)
        print(f"[K permutation tensor reference] cases={cases} cells={cells} roundtrip/tail PASS")

    def test_not_identity_and_shape_fail_closed(self):
        x = torch.arange(64).reshape(1, 1, 64, 1)
        y = permute_key_reference(x, "HND")
        self.assertFalse(torch.equal(x, y))
        self.assertEqual(y.flatten()[:16].tolist(),
                         [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15])
        with self.assertRaises(ValueError):
            permute_key_reference(x, "unknown")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
