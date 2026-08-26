# `workgroupDim()` reads the dispatch packet through the vector path

`AMDGPU.workgroupDim()` (and its alias `blockDim()`) reads the workgroup size out
of the AQL dispatch packet. The packet's fields are 16 bits wide and SMEM has no
sub-dword load, so putting that read on the scalar path requires LLVM to widen the
i16 access. It manages that for a lone `.x` and stops managing it the moment a
second field is read, at which point every wave issues a **vector** load of
uncached queue memory and the index arithmetic behind it waits on `vmcnt`.

CUDA never has this problem: `blockDim` is `%ntid`, a register.

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

CUDA control, on an NVIDIA machine:

```bash
julia --project=. -e 'using Pkg; Pkg.add("CUDA")'
MWE_BACKEND=CUDA julia --project=. workgroupdim_bandwidth.jl
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
