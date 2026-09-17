# Remaining Sage integer-PV instruction account

Start `97ea065`, experimental worktree only. Keep clean-source `a49338f`
candidate binary and main FP16 PV immutable. Workload B1/H56/S73774/D128/full,
Q128/KV64, four warps. No GPU run or user deadline is specified this turn.

Scope: audit actual PPU and NVIDIA generated code, not a second mathematical
implementation. Old PPU ACU PC weights and native 5070 NCU exports anchor the
static-code interpretation. Separate QK, P packing/retiling, online softmax,
PV restoration/scaling/addition, address/control and once-only epilogue.
Treat MUL and FMA substitutions as one semantic account, and include native
FP16-buffer HADD2.F32 work. Do not attribute their common QK conversion to PV.

Bounded inventory:

1. Correct static parser section/exit/reachability semantics; an s.exit and
   a neighbouring ELF text section must not import padding into kernel counts.
   A branch to code after an early exit must remain counted. Recompute stored
   whole-body totals; do not change integer-category membership.
2. Derive actual native loop SCC, distinguish per-iteration from epilogue
   counts, and calibrate old code against its independent ACU PC evidence.
   Any new execution-count bound is a bound, not measured dynamic counts.
3. Investigate native deferred denominator reduction and QK scale/sub FMA.
   Construct positive/rounding-negative host witnesses. These reorder FP32
   operations: do not silently install them under the existing raw-equality
   contract. Record their instruction opportunity and numerical boundary.

No production kernel edits or replacement binary in this investigation.
Keep source/negative tests and conclusion together, then hand off the next
single experiment. Main and installed wheel routing must stay unchanged.
