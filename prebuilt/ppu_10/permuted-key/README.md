# Isolated K-permuted INT8 candidate

Source `711ec9930b94aa4a3bcd878ee35ebda91e7eab89`; SDK2.1.1 / ppu_10.
CPython3.12, Torch2.9.0 public ABI, CXX11 ABI1, native runtime13.0.

Binary SHA256:
`42702fd43a5cce73acd0c38967ab568f519cca67ffc0d98bf847a6a8591beea2`.

Local: SDK72/72,no private stack,H3 regs244/104. Clean build reproduces all72
trial opcode+operand streams. Old48 streams (including FP16 PV and raw INT8)
remain unchanged. Host traits/basis/tails, source and generated-code negatives,
manifest negatives and real extension import pass. Real PPU device numerics,
latency and dynamic ACU counts are **NOT RUN**.

```bash
git switch experiment/ppu-pv-int8
git pull --ff-only
PROFILE=1 bash tools/run_ppu_int8_key_permutation_box.sh
```

No box build, pip installation or replacement of the installed package. A
failed numeric gate stops timing/profile. Output is under `/workspace`; the
final line prints the directory. `PROFILE=0` gives correctness+normal events
only. If the SDK is not `/usr/local/PPU_SDK`, set `PPU_SDK` to its root or
`PPU_RUNTIME_DIR` to its actual runtime directory.

Normal event rows: FP16 PV, raw INT8 PV, K-permuted INT8 PV, both K preparation
times, and both K-prepare+INT8-PV times. ACU separately records the new explicit
attention symbol; its instrumented duration is not normal-event latency.
See `docs/PPU_KEY_PERMUTATION.md` for the unchanged numerical admission and
the distinction between emitted CFG paths and measured instruction counts.
