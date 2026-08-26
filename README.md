# `workgroupDim()` reads the dispatch packet through the vector path

`AMDGPU.workgroupDim()` (and its alias `blockDim()`) reads the workgroup size out
of the AQL dispatch packet. The packet's fields are 16 bits wide and SMEM has no
sub-dword load, so putting that read on the scalar path requires LLVM to widen the
i16 access. It manages that for a lone `.x` and stops managing it the moment a
second field is read, at which point every wave issues a **vector** load of
uncached queue memory and the index arithmetic behind it waits on `vmcnt`.

CUDA never has this problem: `blockDim` is `%ntid` in PTX and a cached
constant-bank load in the final SASS — off the memory-dependency path either
way. The same reproducer measures 1.00x on an H200 where AMD measures 4.6x;
see [the same test on CUDA](#the-same-test-on-cuda).

`workgroupdim_bandwidth.jl` reproduces it, measures it, and prints the ISA of the
kernels it timed so the result cannot be confused with a stale build.

## What it costs, and why that number varies so much

Two conditions have to hold together, and missing the second is what makes this
defect easy to under-measure by a factor of ten.

**1. How many dimensions the kernel reads.** One field stays scalar; a second one
drops the whole read onto the vector path.

| dims read | without the fix | with the fix |
|---|---|---|
| 1 (x) | `s_load_dword` | `s_load_dword` |
| 2 (x,y) | `global_load_dword` | `s_load_dword` |
| 3 (x,y,z) | `global_load_dword` + `global_load_ushort` | `s_load_dwordx2` |

**2. How many loads the kernel has in flight.** `vmcnt` is a counter over *all*
outstanding vector memory operations, not just the one being waited on. So a
kernel that streams many arrays has its own memory-level parallelism drained at
exactly the point where it computes its indices. A two-array copy has almost none
to lose; a kernel touching a dozen arrays loses most of it.

That is the axis the reproducer sweeps: the same fused update over 2, 6 and 14
array accesses per thread, at one grid, one block shape and one precision.

## Measured

gfx950 (MI355X), ROCm 7.14, Julia 1.13.0-rc3, `N=256`, Float64, block 256.
AMDGPU.jl 2.8.0 from the registry versus the same version plus the fix. Ranges
are three run-pairs. The figure is `dim/arg` — what the packet read costs the
kernel that performs it, against an otherwise identical kernel receiving the
block size as an argument.

| kernel | accesses/thread | packet loads | unfixed | fixed | recovered |
|---|---|---|---|---|---|
| `stream2` | 2 | 2 → 1 | 1.8-2.5x | 1.4-1.7x | +26-71% |
| `stream6` | 6 | 2 → 1 | 3.0-3.2x | 1.05-1.07x | +185-203% |
| **`stream14`** | **14** | **2 → 1** | **4.6-4.7x** | **1.02-1.03x** | **+347-361%** |
| `stream2/mix` (control) | 2 | 1 → 1 | 1.29-1.30x | 1.31-1.38x | -1 to -6% |
| short kernels | — | — | 8.8-9.4x | 5.5-5.6x | +58-67% |

In absolute terms the 14-access kernel runs at **1050-1068 GB/s unfixed and
4657-4734 GB/s fixed** — a 4.6x difference in the kernel, landing on the
argument-passed baseline once the read is scalar.

The `stream2` row is the shape a reproducer reaches for first, and it understates
the defect by more than ten times. Measuring a copy kernel and concluding "about
20%" measures the one case with no memory-level parallelism to lose.

### The same test on CUDA

The same script, geometry and precision on an NVIDIA H200 NVL (CUDA.jl 6.3.0,
Julia 1.12.7, `N=256`, Float64, block 256). Ranges are three runs; the
short-kernel row, two. Same `dim/arg` figure as above, against the same `arg`
control within each run.

| kernel | accesses/thread | GB/s (`dim`) | GB/s (`arg`) | `dim/arg` | scatter |
|---|---|---|---|---|---|
| `stream2` | 2 | 3174-3186 | 3168-3190 | 1.00-1.01x | 1.9-2.8% |
| `stream6` | 6 | 4051-4066 | 4114-4133 | 1.01-1.02x | 0.9-1.4% |
| **`stream14`** | **14** | **4063-4067** | **4048-4055** | **1.00x** | 0.4-0.9% |
| `stream2/mix` (control) | 2 | 3181-3184 | — | 1.00x | 2.3-2.6% |
| short kernels | — | 2.61-2.72 us/launch | 2.64-2.73 us/launch | 0.99x | 0.6-0.7% |

The axis that costs 4.6x on AMD is flat: 1.00-1.02x from 2 to 14 accesses per
thread, with `dim` marginally *faster* than the argument-passed baseline in all
three runs. The effect is at or below this machine's noise.

The `stream2/mix` row is worth reading against its AMD counterpart. There it is
1.29-1.30x unfixed and 1.31-1.38x fixed — the residual cost of a *scalar* packet
read, which the fix does not claim to remove. Here the same control is 1.00x,
because there is no packet to read.

Why, in the generated code. PTX models the three dimensions as special registers:

```
mov.u32 %r2, %ntid.x;   mov.u32 %r6, %ntid.y;   mov.u32 %r11, %ntid.z;
```

and the H200 SASS materialises them as constant-bank loads:

```
LDC R7, c[0x0][RZ]    ; blockDim.x
LDC R0, c[0x0][0x4]   ; blockDim.y
LDC R4, c[0x0][0x8]   ; blockDim.z
```

`LDC` is a broadcast read of the cached constant bank. It is not a global-memory
access, and it does not enter the scoreboard that the kernel's 15 `LDG` data loads
use — so the index arithmetic has nothing to wait on, and there is no `vmcnt`
equivalent to drain. Reading a second and a third dimension costs one more `LDC`
each; the access path never changes. That is exactly the step that does not
survive on AMD.

`stream14/dim` and `stream14/arg` compile to the same instruction mix:

| | `LDG` (data) | `LDC` (constant bank) | total instructions |
|---|---|---|---|
| `stream14/dim` | 15 | 42 | 493 |
| `stream14/arg` | 15 | 42 | 500 |

On NVIDIA, reading `blockDim` and reading a kernel argument are the same class of
access — both `LDC` from constant bank 0. That is why `dim/arg` is 1.00x and could
not have been much else.

So the 4.6x is not the price of indexing a 3-D kernel by its block size. The
identical Julia source pays nothing for it on comparable hardware. The cost is
specific to the AMD lowering putting a 16-bit packet field on the vector path,
where `vmcnt` then serialises the kernel's other in-flight loads — which is what
the fix removes, landing at 1.02-1.03x, where CUDA already starts.

Two labels in the CUDA output are artifacts of an AMD-oriented script rather than
findings: the `path` column reads `other`, because `classify` only knows
`s_load`/`global_load` mnemonics, and `this build: FIXED` is trivially true since
there is no unfixed CUDA variant to compare against.

### Controls

- **`arg`** — same source, same arithmetic, same launch geometry, block size
  arriving as a kernel argument (a preloaded SGPR). Every ratio above is against
  this, within the same run.
- **`stream2/mix`** — reads exactly **one** packet field with everything else
  held identical, so it is scalar in both builds and the fix cannot change one
  instruction of it. It is a control for the fix, not a noise floor: a scalar
  packet read is not free either, and this row is what "as good as it gets" looks
  like on this machine.

### Scatter

Reported per row. The 6- and 14-access rows settle within 1-4% once fixed,
against 20-36% unfixed. The 2-access rows are unstable in both builds (5-50%) —
one more reason not to rest a conclusion on them.

## Running it

```bash
julia --project=. -e 'using Pkg; Pkg.add("AMDGPU")'
julia --project=. workgroupdim_bandwidth.jl

julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/wave-photonics/AMDGPU.jl", rev="widen-dispatch-packet-loads")'
julia --project=. workgroupdim_bandwidth.jl
```

Either order. Each run writes its numbers next to the script, keyed by device,
size and precision, and the second run prints the before/after table. The run
also states which lowering is in the binary, read from the ISA of the kernels it
timed rather than from what is believed to be installed.

CUDA control, on an NVIDIA machine — `cudacheck/` carries a resolved manifest, so
this is the exact stack the numbers above came from:

```bash
julia --project=cudacheck -e 'using Pkg; Pkg.instantiate()'
MWE_BACKEND=CUDA julia --project=cudacheck workgroupdim_bandwidth.jl
```

Optional ParallelStencil section — its `@parallel_indices` emits the
3-dimension index expression and offers no way to pass the block size in, so it
has no `arg` control of its own and is compared against `stream2/arg`:

```bash
julia --project=. -e 'using Pkg; Pkg.add("ParallelStencil")'
MWE_PS=1 julia --project=. workgroupdim_bandwidth.jl
```

Knobs: `MWE_BACKEND=CUDA|AMDGPU`, `MWE_N` (cube side, default 256), `MWE_BLOCK`
(threads in x, default 256), `MWE_REPS` (default 30), `MWE_PREC=F64|F32`
(default F64), `MWE_PS=0|1`.

### Two traps worth knowing

**Switching builds.** Pass the revision to `Pkg.add`, as above. Editing a
`[sources]` entry and calling `Pkg.resolve()` is *not* enough: Pkg keeps the
manifest's `git-tree-sha1` and reports the new revision while loading the old
tree, which silently turns an A/B into an A/A. The line each run prints about
which lowering is present is there to catch this.

**Sizing.** Ratios inflate at sizes that fit the last-level cache — the control's
included — because wave entry stops being amortised against DRAM traffic. Results
are only comparable at equal `N` and precision, which is why the saved result
files are keyed by both. Fourteen arrays at `N=256` is 1.75 GiB in Float64, and
one array (128 MiB) does not fit a 256 MB LLC alongside the others.

**Comparing runs.** Compare ratios, not GB/s. Absolute bandwidth drifts by ~9%
between two runs on a clock-throttled part, which is larger than some of the
effects in the table above. Each ratio is taken against its own run's baseline,
and the variants are timed round-robin so drift hits them equally.

## What this does not claim

It does not make reading the dispatch packet free. The read is of uncached queue
memory either way; the fix removes the vector path and the `vmcnt` wait, not the
access. That is why the control's ~1.3x survives the fix, and why the
short-kernel regime — where the read's own latency is most of the kernel — only
halves rather than disappearing.

The magnitude also depends on the LLVM release doing the lowering and on where
the runtime places the queue. The figure to trust on any given machine is the one
that machine prints.

## Layout

| path | what it is |
|---|---|
| `workgroupdim_bandwidth.jl` | the reproducer; no dependencies beyond a backend |
| `unpatched/` | project pinning AMDGPU.jl from the registry |
| `patched/` | project pinning the branch carrying the fix |
| `cudacheck/` | project pinning CUDA.jl, for the control on NVIDIA |
| `mwe_result_*.txt` | per-run numbers, written by the script, gitignored |
