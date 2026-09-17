# Resume

Parent4486f03. K-row map/direct P actual-traits basis and all negative controls
pass. r1/r2 rejected at248 registers. r3 row-local tail bridge has244/104,
stack0,72 real SDK types,old48 opcode+operand streams identical. Evidence:
`/workspace/sage-key-permutation-20260917/sdk-r3/key-codegen.json`.

Clean source711ec99 build replays72/72 native streams. Isolated binary under
`prebuilt/ppu_10/permuted-key`,hash42702fd43a5cce73acd0c38967ab568f519cca67ffc0d98bf847a6a8591beea2.
Manifest negatives and real Torch2.9 extension import (even with legacy shim)
pass. No device launched locally. Native SDK gate ran; legacy NVCC host-wrapper
gate SKIP(nvcc absent),actual host CuTe layout oracle passes with SDK headers.

Next: `PROFILE=1 bash tools/run_ppu_int8_key_permutation_box.sh` runs numeric
gate then normal INT8/FP16/new controls including moved K-prepare cost,then ACU.
Do not assign unbound user570ms to a candidate. No default/main update.
