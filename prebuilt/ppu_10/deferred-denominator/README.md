# Experimental deferred-denominator prebuilt

Not a replacement default wheel. Main retains INT8-QK / FP16-PV.

Source 567bba728393209c4ac13f38279be93f6e56e830, PPU SDK2.1.1, ppu_10,
CPython3.12 / Torch2.9 / C++11 ABI1, native runtime libhggcrt.13.0.so.
Binary SHA256 0c41e28d02c491b61992fd33de75eacafd76a967cc937b2d882f23f7029ad22d.

Local native build/placement/numerical-seam checks passed. Device correctness,
latency and ACU have NOT RUN; use the mandatory device gate before timing.
Denominator FP32 summation order changes; P/V quantization does not.

Run `bash tools/run_ppu_int8_denominator_ab_box.sh` from the experimental
branch. It preserves the old a49338f binary and stages both arms privately,
sequentially, using the same numerical and profile driver. No box compilation
or pip installation. See docs/PPU_DEFERRED_DENOMINATOR.md for the fixed gates.
