# workgroupdim_bandwidth.jl
#
# Minimal reproducer for the cost of `AMDGPU.workgroupDim()` lowering to a VMEM
# read of uncached AQL dispatch-packet memory instead of an SMEM (`s_load`)
# read, and the proof that the same source is unaffected on CUDA.
#
# The experiment is self-normalising: each kernel is compiled twice from the
# same source, differing only in where the block size comes from.
#
#   dim  -- `(blockIdx-1) * blockDim + threadIdx`, the expression
#           ParallelStencil's `@parallel_indices` emits and the one in
#           AMDGPU.jl's own regression test for this change.
#   arg  -- identical arithmetic, but the block size arrives as a kernel
#           argument (a preloaded SGPR) instead of from `workgroupDim()`.
#
# Identical memory traffic, identical arithmetic, identical launch geometry.
# Any gap between `dim` and `arg` is the dispatch-packet read alone, so the
# headline number is the ratio `arg / dim` -- no cross-machine comparison and
# no vendor peak-bandwidth figure needed to read the result.
#
# Expected:
#   CUDA (any version)      arg/dim ~ 1.00   -- `blockDim` is %ntid, a register
#   AMDGPU without the fix  arg/dim >> 1     -- per-wave uncached VMEM read
#   AMDGPU with the fix     arg/dim ~ 1.00   -- and `dim` matches `arg` absolutely
#
# ---------------------------------------------------------------------------
# Running it
#
#   mkdir mwe && cd mwe && cp /path/to/workgroupdim_bandwidth.jl .
#
# AMDGPU, without the fix (whatever the registry currently has):
#   julia --project=. -e 'using Pkg; Pkg.add("AMDGPU")'
#   julia --project=. workgroupdim_bandwidth.jl
#
# AMDGPU, with the fix:
#   julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/wave-photonics/AMDGPU.jl", rev="widen-dispatch-packet-loads")'
#   julia --project=. workgroupdim_bandwidth.jl
#
# Swap back and forth freely -- Pkg recompiles on the change, and the ISA probe
# at the end of every run states which lowering is actually in the binary, so a
# stale cache cannot be mistaken for a result.
#
# CUDA control:
#   julia --project=. -e 'using Pkg; Pkg.add("CUDA")'
#   MWE_BACKEND=CUDA julia --project=. workgroupdim_bandwidth.jl
#
# Optional ParallelStencil section (the original real-world trigger):
#   julia --project=. -e 'using Pkg; Pkg.add("ParallelStencil")'
#   MWE_PS=1 julia --project=. workgroupdim_bandwidth.jl
#
# Knobs: MWE_BACKEND=CUDA|AMDGPU  MWE_N=<cube side, default 512>
#        MWE_BLOCK=<x,y,z, default 256,1,1>  MWE_REPS=<default 10>  MWE_PS=0|1
#
# Sizing: N should put a single array comfortably past the last-level cache, or
# the copy stops being DRAM-bound and the whole table compresses. N=512 is
# 512 MiB per array, which clears MI300/MI350-class 256 MB LLCs. Drop it only if
# the device cannot hold three of those.
#
# ---------------------------------------------------------------------------
# Measured on gfx1151 (Radeon 8060S, Strix Halo), N=384, AMDGPU.jl 2.8.0 from
# the registry vs. the branch above, Julia 1.12.6. An APU, where the dispatch
# packet sits in the same LPDDR5X as everything else -- about the mildest this
# bug ever looks:
#
#                   without the fix      with the fix
#   copy  dim         173.5 GB/s          214.8 GB/s
#   copy  arg         204.2 GB/s          205.2 GB/s
#   copy  arg/dim       1.18x               0.96x
#   triad arg/dim       1.11x               1.02x
#   PS    arg/dim       1.09x               0.96x
#   probe ISA         VMEM fallback       s_load, scalar
#
# Run-to-run scatter here is a few percent, so once the fix is in, every ratio
# is 1.00 to within noise and which side lands on top means nothing. What is not
# noise is `dim` alone moving 173.5 -> 214.8 GB/s, up to the `copyto!` reference
# and the `arg` control.
#
# The ISA probe flips on every AMD part; how much bandwidth that costs does not.
# Discrete CDNA reaches the dispatch packet over a much longer path than an APU
# does, which is where the large factors show up.
# ---------------------------------------------------------------------------

import Pkg

const BACKEND = let e = get(ENV, "MWE_BACKEND", "")
    if e != ""
        Symbol(e)
    elseif Base.find_package("AMDGPU") !== nothing
        :AMDGPU
    elseif Base.find_package("CUDA") !== nothing
        :CUDA
    else
        error("No GPU backend found. Pkg.add(\"AMDGPU\") or Pkg.add(\"CUDA\"), " *
              "or set MWE_BACKEND.")
    end
end

const N     = parse(Int, get(ENV, "MWE_N", "512"))
const REPS  = parse(Int, get(ENV, "MWE_REPS", "10"))
const BLOCK = Tuple(parse.(Int, split(get(ENV, "MWE_BLOCK", "256,1,1"), ',')))
const USE_PS = get(ENV, "MWE_PS", "0") == "1"

# --- backend shim ----------------------------------------------------------
# The only backend-specific code in the file. `gidx`/`gdim`/`lidx` are inlined
# away; the ISA dump at the end of the run confirms nothing survives of them.

@static if BACKEND === :CUDA
    using CUDA
    const Arr = CuArray
    @inline gidx() = blockIdx()
    @inline gdim() = blockDim()
    @inline lidx() = threadIdx()
    devsync() = CUDA.synchronize()
    devname() = CUDA.name(CUDA.device())
    macro launch(nb, nt, call)
        esc(:(CUDA.@cuda blocks = $nb threads = $nt $call))
    end
elseif BACKEND === :AMDGPU
    using AMDGPU
    const Arr = ROCArray
    @inline gidx() = workgroupIdx()
    @inline gdim() = workgroupDim()
    @inline lidx() = workitemIdx()
    devsync() = AMDGPU.synchronize()
    devname() = AMDGPU.HIP.gcn_arch(AMDGPU.device())
    # `gridsize` is the workgroup count: @roc forwards it straight to
    # hipModuleLaunchKernel's gridDim, so it means what CUDA's `blocks` means.
    macro launch(nb, nt, call)
        esc(:(AMDGPU.@roc gridsize = $nb groupsize = $nt $call))
    end
else
    error("MWE_BACKEND must be CUDA or AMDGPU, got $BACKEND")
end

# --- kernels ---------------------------------------------------------------
# Each pair is the same source with one substitution: gdim().x -> bx.

function copy_dim!(dst, src, nx, ny, nz)
    i = (gidx().x - 1) * gdim().x + lidx().x
    j = (gidx().y - 1) * gdim().y + lidx().y
    k = (gidx().z - 1) * gdim().z + lidx().z
    if i <= nx && j <= ny && k <= nz
        @inbounds dst[i, j, k] = src[i, j, k]
    end
    return
end

function copy_arg!(dst, src, nx, ny, nz, bx, by, bz)
    # `% typeof(...)` keeps the arithmetic bit-identical to copy_dim! on either
    # backend (CUDA's index intrinsics are Int32, AMDGPU's UInt32) without this
    # file hardcoding either. It has to be the wrapping conversion, not a
    # checked one: Int32(bx) on a kernel argument the compiler cannot prove
    # in-range plants a throw path in the control kernel that `dim` does not
    # have, which would bias the comparison in the bug's favour.
    i = (gidx().x - 1) * (bx % typeof(lidx().x)) + lidx().x
    j = (gidx().y - 1) * (by % typeof(lidx().y)) + lidx().y
    k = (gidx().z - 1) * (bz % typeof(lidx().z)) + lidx().z
    if i <= nx && j <= ny && k <= nz
        @inbounds dst[i, j, k] = src[i, j, k]
    end
    return
end

function triad_dim!(d, a, b, s, nx, ny, nz)
    i = (gidx().x - 1) * gdim().x + lidx().x
    j = (gidx().y - 1) * gdim().y + lidx().y
    k = (gidx().z - 1) * gdim().z + lidx().z
    if i <= nx && j <= ny && k <= nz
        @inbounds d[i, j, k] = a[i, j, k] + s * b[i, j, k]
    end
    return
end

function triad_arg!(d, a, b, s, nx, ny, nz, bx, by, bz)
    i = (gidx().x - 1) * (bx % typeof(lidx().x)) + lidx().x
    j = (gidx().y - 1) * (by % typeof(lidx().y)) + lidx().y
    k = (gidx().z - 1) * (bz % typeof(lidx().z)) + lidx().z
    if i <= nx && j <= ny && k <= nz
        @inbounds d[i, j, k] = a[i, j, k] + s * b[i, j, k]
    end
    return
end

# Write-only probe, so the only memory *reads* in its ISA are the dispatch
# packet. This is the kernel from AMDGPU.jl's regression test, verbatim.
function probe!(A)
    i = (gidx().x - 1) * gdim().x + lidx().x
    j = (gidx().y - 1) * gdim().y + lidx().y
    k = (gidx().z - 1) * gdim().z + lidx().z
    n = i + j + k
    n <= length(A) && (@inbounds A[n] = n)
    return
end

# --- optional ParallelStencil section --------------------------------------
# The original real-world trigger: @parallel_indices emits exactly the `dim`
# expression above, and gives the user no way to pass the block size in, so it
# has no `arg` control of its own -- compare it against copy/arg.
#
# Each statement below is its own top-level block on purpose. `@static if` fully
# macro-expands its body, so `using ParallelStencil` has to have already run in
# an earlier statement before @init_parallel_stencil can be expanded.

@static if USE_PS
    using ParallelStencil
end

@static if USE_PS && BACKEND === :CUDA
    @init_parallel_stencil(CUDA, Float32, 3)
end

@static if USE_PS && BACKEND === :AMDGPU
    @init_parallel_stencil(AMDGPU, Float32, 3)
end

@static if USE_PS
    @parallel_indices (i, j, k) function ps_copy!(dst, src)
        @inbounds dst[i, j, k] = src[i, j, k]
        return
    end

    function ps_copy_launch!(dst, src, nx, ny, nz)
        @parallel (1:nx, 1:ny, 1:nz) ps_copy!(dst, src)
        return
    end
end

# --- harness ---------------------------------------------------------------

function bench(run!, reps)
    run!(); devsync()                       # compile + warm up
    ts = Float64[]
    for _ in 1:reps
        devsync()
        t0 = time_ns()
        run!()
        devsync()
        push!(ts, (time_ns() - t0) * 1e-9)
    end
    return minimum(ts)
end

nblocks(n, b) = cld(n, b)

# So that a pasted run identifies the stack it came from without a covering note.
# The ISA probe at the end of the run is the ground truth for whether the fix is
# present; this is just here to name the stack that produced the numbers.
function backend_provenance()
    try
        for (_, p) in Pkg.dependencies()
            p.name == string(BACKEND) || continue
            src = p.is_tracking_path ? p.source :
                  p.is_tracking_repo ? "$(p.git_source) @ $(p.git_revision)" :
                  "registry"
            return "v$(p.version)  [$src]"
        end
    catch
    end
    return "version unknown"
end

function main()
    nx = ny = nz = N
    cells = nx * ny * nz
    nb = (nblocks(nx, BLOCK[1]), nblocks(ny, BLOCK[2]), nblocks(nz, BLOCK[3]))
    waves = cld(prod(BLOCK), BACKEND === :AMDGPU ? 64 : 32) * prod(nb)

    println("julia      : $VERSION")
    println("backend    : $BACKEND $(backend_provenance())")
    println("device     : $(devname())")
    println("grid       : $(nx)x$(ny)x$(nz) = $(cells) Float32 cells")
    println("launch     : blocks $nb x threads $BLOCK  ($(waves) waves)")
    println("arrays     : $(round(cells * 4 / 2^20, digits = 1)) MiB each")
    println("reps       : $REPS (best of)\n")

    src = Arr(fill(1.0f0, nx, ny, nz))
    dst = Arr(zeros(Float32, nx, ny, nz))
    bsrc = Arr(fill(2.0f0, nx, ny, nz))
    s = 3.0f0

    bx, by, bz = BLOCK
    rows = Tuple{String, String, Float64, Float64, Bool}[]

    function record(kernel, variant, run!, bytes, check)
        fill!(dst, 0.0f0)
        t = bench(run!, REPS)
        push!(rows, (kernel, variant, t, bytes / t / 1e9, check()))
    end

    cpbytes = 2 * cells * 4
    trbytes = 3 * cells * 4

    # Reference: the vendor's own device-to-device copy, which uses no
    # hand-written index arithmetic at all. If `arg` below lands far under
    # this, the launch geometry is wrong and nothing else in the table means
    # anything -- that check is what caught a 256x-oversubscribed grid here.
    fill!(dst, 0.0f0)
    tref = bench(() -> copyto!(dst, src), REPS)
    push!(rows, ("copyto!", "-", tref, cpbytes / tref / 1e9, all(==(1.0f0), dst)))
    allcopy() = all(==(1.0f0), dst)
    alltriad() = all(==(1.0f0 + s * 2.0f0), dst)

    record("copy", "dim", () -> (@launch nb BLOCK copy_dim!(dst, src, nx, ny, nz)),
        cpbytes, allcopy)
    record("copy", "arg",
        () -> (@launch nb BLOCK copy_arg!(dst, src, nx, ny, nz, bx, by, bz)),
        cpbytes, allcopy)
    record("triad", "dim",
        () -> (@launch nb BLOCK triad_dim!(dst, src, bsrc, s, nx, ny, nz)),
        trbytes, alltriad)
    record("triad", "arg",
        () -> (@launch nb BLOCK triad_arg!(dst, src, bsrc, s, nx, ny, nz, bx, by, bz)),
        trbytes, alltriad)

    @static if USE_PS
        record("PS", "dim", () -> ps_copy_launch!(dst, src, nx, ny, nz), cpbytes, allcopy)
    end

    println(rpad("kernel", 8), rpad("variant", 9), lpad("best ms", 10),
        lpad("GB/s", 10), "  correct")
    println("-"^46)
    for (k, v, t, gbs, ok) in rows
        println(rpad(k, 8), rpad(v, 9), lpad(round(t * 1e3, digits = 3), 10),
            lpad(round(gbs, digits = 1), 10), "  ", ok ? "yes" : "NO")
    end

    if !all(r -> r[5], rows)
        println("\n!! a kernel produced the wrong result; the timings above are void")
    end

    println()
    for k in ("copy", "triad")
        d = findfirst(r -> r[1] == k && r[2] == "dim", rows)
        a = findfirst(r -> r[1] == k && r[2] == "arg", rows)
        (d === nothing || a === nothing) && continue
        println("$(rpad(k, 6)) arg/dim = $(round(rows[a][4] / rows[d][4], digits = 2))x")
    end
    @static if USE_PS
        p = findfirst(r -> r[1] == "PS", rows)
        a = findfirst(r -> r[1] == "copy" && r[2] == "arg", rows)
        if p !== nothing && a !== nothing
            println("PS     arg/dim = $(round(rows[a][4] / rows[p][4], digits = 2))x " *
                    "(ParallelStencil @parallel_indices vs the argument-passed control)")
        end
    end

    isa_report()
    return nothing
end

# --- ISA evidence ----------------------------------------------------------
# Diagnostic only: never allowed to take the benchmark down with it.

function isa_report()
    println()
    try
        io = IOBuffer()
        @static if BACKEND === :AMDGPU
            tt = Tuple{AMDGPU.Device.ROCDeviceVector{Float32, AMDGPU.Device.AS.Global}}
            AMDGPU.code_gcn(io, probe!, tt; kernel = true)
            asm = String(take!(io))
            scalar = occursin("s_load", asm)
            vector = occursin("global_load", asm) || occursin("flat_load", asm)
            println("dispatch-packet probe ISA: ",
                vector ? "VMEM fallback -- global_load/flat_load present" :
                scalar ? "scalar -- s_load only, no VMEM read" : "inconclusive")
            println("  (the probe only writes, so any load in its ISA is the " *
                    "dispatch packet)")
        else
            tt = Tuple{CuDeviceVector{Float32, CUDA.AS.Global}}
            CUDA.code_ptx(io, probe!, tt; kernel = true)
            ptx = String(take!(io))
            println("blockDim probe PTX: ",
                occursin("%ntid", ptx) ? "%ntid -- a register, no memory read" :
                "unexpected (no %ntid found)")
        end
    catch e
        println("ISA probe unavailable: ", sprint(showerror, e))
    end
    return nothing
end

main()
