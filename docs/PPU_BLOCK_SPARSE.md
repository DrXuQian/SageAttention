# PPU block-sparse SageAttention

The first PPU sparse forward has one executor and two routing front ends.  It
does not turn either community algorithm into a dense boolean mask.

## Canonical plan

`SparseAttentionPlan` owns an ordered CSR list of exact KV64 tiles plus a
bitset of selected route blocks.  The device executor performs exact INT8-QK / 
FP16-PV attention over the CSR list.  When summary correction is enabled it
then visits the contiguous block means, masks means already represented by an
exact tile, adds `log2(valid_tokens)`, and continues the same online softmax.

The kernel does not know whether a plan came from H3 top-k or Sol:

| planner | Q route block | K route block | exact lowering |
|---|---:|---:|---|
| H3 top-k | 128 | 128 | one selected K128 becomes two KV64 tiles |
| Sol | 64 | 64 | one selected K64 becomes one KV64 tile |

The H3 planner reproduces block-mean top-k routing.  The Sol planner reproduces
the diagonal/exact threshold choice, forced adjacent blocks, and explicit KV
and query sink ranges.  Both canonicalize exact execution into increasing KV
order so repeated launches have a fixed floating-point order.

## Deliberate first-shipping boundary

The native sparse ABI is forward-only and fail-closes unless all of these are
true:

- BF16 Q/K/V and BF16 output;
- head dimension 128;
- non-causal execution described by an admitted plan;
- no LSE result;
- Q blocks of 64 (Sol) or 128 (H3), and KV compute blocks of 64.

The generic host planner can model other head dimensions, but successful host
algebra is not device support.  The SDK census therefore contains exactly four
sparse specializations, not a speculative cross product.

PPU code generation currently assigns 256 vector registers and an 8-byte
private frame to each summary specialization.  Exact-only specializations and
all pre-existing dense/quantization kernels remain stack-free.  This is an
explicit ACU postcondition: a frame in metadata is not yet evidence of actual
private-memory traffic, but it is not hidden as `spill=0` either.

## Admission

Host admission proves plan algebra independently of the device executor.  It
also runs three expected-red identity mutations and rejects any box runner
that contains a compiler, linker, or disassembler entry point:

```bash
bash dev/ppu_sparse/run_local_gates.sh
```

The checked-in PPU extension is compiled locally from the real `-arch=ppu_10`
source graph.  Its manifest binds the binary hash to the source hashes,
actlize revision, Python/Torch ABI, SDK release, and the locally measured
resource census.  The box runner is execution-only: it verifies that identity,
copies the prebuilt extension into the package, and never invokes `hgcc`,
`build_ext`, or a disassembler.  It then requires:

1. full-density Q128 CSR is raw-bit identical to dense PPU SageAttention;
2. full-density Q64 CSR is raw-bit identical to the same dense path;
3. H3 top-k summary agrees with an oracle made from the exact quantized
   operands consumed by the kernel;
4. Sol diagonal-summary agrees with that same independent oracle;
5. tail, GQA, HND/NHD and repeated-launch fingerprints pass;
6. a mutated, unadmitted plan is rejected before launch.

```bash
OUT=/workspace/sageattention-ppu-sparse-run \
  bash tools/run_ppu_sparse_box.sh
```

`PPU_RUNTIME_DIR` may override the default `/usr/local/PPU_SDK/lib`; it names
runtime libraries only and is not a compiler path.  A source, ABI, submodule,
or binary mismatch fails before any device launch.

Performance output keeps three costs separate: planner, prequantized sparse
core, and preplanned quantization-plus-core.  The current Python/Torch Sol
planner materializes its route score matrix; it is the semantic reference and
first device baseline, not the final long-sequence routing implementation.

## Sources of semantics

- H3 top-k and summary-row semantics:
  `Occipital-Labs/h3-sparse-attn` (MIT).
- Sol threshold, neighbour, sink and summary semantics:
  `Saganaki22/ComfyUI-sol-attn`.

This implementation reuses neither project's NVIDIA/Triton instruction code.
Only their forward algorithms are transcribed onto the existing actlize PPU
AIU primitives.
