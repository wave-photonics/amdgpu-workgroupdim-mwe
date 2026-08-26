# workgroupdim_bandwidth.jl
#
# Minimal reproducer for `AMDGPU.workgroupDim()` lowering to a *vector* read of
# the AQL dispatch packet instead of a scalar (`s_load`) one, and for what that
# costs. Also shows that the same source is unaffected on CUDA.
#
# Two things have to line up for the cost to appear, which is why it is easy to
# measure something much smaller than the real thing.
#
# 1. HOW MANY DIMENSIONS THE KERNEL READS. The packet fields are 16 bits wide and
#    SMEM has no sub-dword load, so selecting `s_load` needs LLVM to widen the
#    i16 access. It manages that for a lone `.x` and stops managing it as soon as
#    a second field is read:
#
#      dims read | without the fix                        | with the fix
#      ----------+---------------------------------------+----------------
#      1 (x)     | s_load_dword                          | s_load_dword
#      2 (x,y)   | global_load_dword                     | s_load_dword
#      3 (x,y,z) | global_load_dword + global_load_ushort | s_load_dwordx2
#
# 2. HOW MANY LOADS THE KERNEL HAS IN FLIGHT. Once the packet read is a vector
#    load, the index arithmetic that depends on it sits behind an `s_waitcnt
#    vmcnt` -- and vmcnt is a counter over *all* outstanding vector memory
#    operations, not just that one. A kernel streaming many arrays at once
#    therefore has its own memory-level parallelism drained at the point it
#    computes its indices. A two-array copy has almost none to lose and shows
#    almost nothing; a kernel touching a dozen arrays loses most of it.
#
# So the reproducer sweeps the second axis: the same fused update over 2, 6 and
# 14 array accesses per thread, at one grid and one block shape. The 2-access row
# is the shape most reproducers reach for first, and it understates the defect by
# roughly four times.
#
# CUDA never has the problem: `blockDim` is `%ntid`, a register.
#
# Controls. Each kernel is compiled twice, differing only in where the block size
# comes from:
#
#   dim -- `(blockIdx-1) * blockDim + threadIdx`
#   arg -- identical arithmetic, block size arriving as a kernel argument
#          (a preloaded SGPR) instead of from `workgroupDim()`
#
# and `mix` reads exactly one packet field with everything else held identical,
# so it is scalar in both builds and the fix cannot change one instruction of it.
# It is a control for the fix, not a noise floor: a scalar packet read is not free
# either, and `mix` is what "as good as it gets" looks like on this machine.
#
# Identical traffic, identical arithmetic, identical launch geometry, so `dim/arg`
# isolates the dispatch-packet read. No cross-machine comparison and no vendor
# peak-bandwidth figure is needed to read the result.
#
# ---------------------------------------------------------------------------
# Running it
#
# Both builds, in either order -- each run writes its numbers next to the script,
# and the second run prints the before/after table:
#
#   julia --project=. -e 'using Pkg; Pkg.add("AMDGPU")'
#   julia --project=. workgroupdim_bandwidth.jl
#
#   julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/wave-photonics/AMDGPU.jl", rev="widen-dispatch-packet-loads")'
#   julia --project=. workgroupdim_bandwidth.jl
#
# Switching builds: pass the revision to `Pkg.add`, as above, and check the line
# the run prints for which lowering is in the binary. Editing a `[sources]` entry
# and calling `Pkg.resolve()` is NOT enough -- Pkg keeps the manifest's
# `git-tree-sha1` and reports the new revision while loading the old tree, which
# silently turns an A/B into an A/A.
#
# CUDA control:
#   julia --project=. -e 'using Pkg; Pkg.add("CUDA")'
#   MWE_BACKEND=CUDA julia --project=. workgroupdim_bandwidth.jl
#
# Optional ParallelStencil section: its `@parallel_indices` emits the 3-dimension
# `dim` expression and offers no way to pass the block size in, so it has no
# `arg` control of its own -- compare it against stream2/arg.
#   julia --project=. -e 'using Pkg; Pkg.add("ParallelStencil")'
#   MWE_PS=1 julia --project=. workgroupdim_bandwidth.jl
#
# Knobs: MWE_BACKEND=CUDA|AMDGPU  MWE_N=<cube side, default 256>
#        MWE_BLOCK=<threads in x, default 256>  MWE_REPS=<default 30>
#        MWE_PREC=F64|F32 (default F64)  MWE_PS=0|1
#
# Sizing: 14 arrays at N=256 is 1.9 GiB in Float64, and one array (128 MiB) does
# not fit a 256 MB last-level cache alongside the others. Ratios inflate at sizes
# that do fit, the control's included, so results are only comparable at equal N
# and precision -- which is why the saved results are keyed by both.
#
# ---------------------------------------------------------------------------
# Measured on gfx950 (MI355X, ROCm 7.14), Julia 1.13.0-rc3, N=256, Float64,
# AMDGPU.jl 2.8.0 from the registry vs. the branch above. Cost of the packet read
# as dim/arg, and what the fix takes back:
#
#   kernel        accesses   packet loads     unfixed        fixed   recovered
#   stream2       2          2 -> 1         1.8-2.5x     1.4-1.7x    +26-71%
#   stream6       6          2 -> 1         3.0-3.2x   1.05-1.07x  +185-203%
#   stream14      14         2 -> 1         4.6-4.7x   1.02-1.03x  +347-361%
#   stream2/mix   2          1 -> 1         1.29-1.30x 1.31-1.38x    -1 to -6%
#   short kernels -          -              8.8-9.4x     5.5-5.6x    +58-67%
#
# (Ranges are three runs each. stream2/mix is the control.)
#
# In absolute terms the 14-access kernel runs at 1050-1068 GB/s unfixed and
# 4657-4734 GB/s fixed. That is the row to quote: 4.6x, and it lands on the
# argument-passed baseline once the read is scalar.
#
# The 2-access row is the shape a reproducer reaches for first, and it understates
# the defect by more than ten times. Anyone measuring a copy kernel and concluding
# "about 20%" has measured the one case with no memory-level parallelism to lose.
#
# Scatter, which the run reports per row: the 6- and 14-access rows settle within
# 1-4% once fixed, against 20-36% unfixed. The 2-access rows are unstable in both
# builds (5-50%) -- one more reason not to rest a conclusion on them.
#
# What this does not claim: it does not make reading the dispatch packet free. The
# read is of uncached queue memory either way, and the patch removes the vector
# path and the vmcnt wait, not the access -- which is why the control's ~1.3x
# survives it, and why the short-kernel regime, where the read's own latency is
# most of the kernel, only halves. And the magnitude depends on the LLVM release
# doing the lowering and on where the runtime places the queue, so the figure to
# trust on any machine is the one that machine prints. Compare ratios rather than
# GB/s across runs: absolute bandwidth drifts by more between two runs on a
# clock-throttled part than the effect being measured.
#
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

const N      = parse(Int, get(ENV, "MWE_N", "256"))
const REPS   = parse(Int, get(ENV, "MWE_REPS", "30"))
const BX     = parse(Int, get(ENV, "MWE_BLOCK", "256"))
const USE_PS = get(ENV, "MWE_PS", "0") == "1"
const T      = get(ENV, "MWE_PREC", "F64") == "F32" ? Float32 : Float64
const CHURN  = 200          # launches per timing in the short-kernel regime

# --- backend shim ----------------------------------------------------------

@static if BACKEND === :CUDA
    using CUDA
    const Arr = CuArray
    @inline gidx() = blockIdx()
    @inline gdim() = blockDim()
    @inline lidx() = threadIdx()
    devsync() = CUDA.synchronize()
    devname() = CUDA.name(CUDA.device())
    devarray(n, S = T) = CuDeviceArray{S, n, CUDA.AS.Global}
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
    devarray(n, S = T) = AMDGPU.Device.ROCDeviceArray{S, n, AMDGPU.Device.AS.Global}
    # `gridsize` is the workgroup count: @roc forwards it straight to
    # hipModuleLaunchKernel's gridDim, so it means what CUDA's `blocks` means.
    macro launch(nb, nt, call)
        esc(:(AMDGPU.@roc gridsize = $nb groupsize = $nt $call))
    end
else
    error("MWE_BACKEND must be CUDA or AMDGPU, got $BACKEND")
end

# --- kernels ---------------------------------------------------------------
# `% typeof(...)` keeps the arithmetic bit-identical to the `dim` variant on
# either backend (CUDA's index intrinsics are Int32, AMDGPU's UInt32) without
# hardcoding either. It has to be the wrapping conversion, not a checked one:
# Int32(bx) on an argument the compiler cannot prove in-range plants a throw path
# in the control kernel that `dim` does not have, which would bias the comparison
# in the defect's favour.

# ISA-only: how many dimensions it takes to lose the scalar path.
dims1!(A, n) = (i = (gidx().x - 1) * gdim().x + lidx().x;
                i <= n && (@inbounds A[i] = one(eltype(A))); nothing)
function dims2!(A, nx, ny)
    i = (gidx().x - 1) * gdim().x + lidx().x
    j = (gidx().y - 1) * gdim().y + lidx().y
    (i <= nx && j <= ny) && (@inbounds A[i, j] = one(eltype(A)))
    return
end

# 2 accesses per thread: one read, one write.
function stream2_dim!(dst, src, nx, ny, nz)
    i = (gidx().x - 1) * gdim().x + lidx().x
    j = (gidx().y - 1) * gdim().y + lidx().y
    k = (gidx().z - 1) * gdim().z + lidx().z
    (i <= nx && j <= ny && k <= nz) && (@inbounds dst[i, j, k] = src[i, j, k])
    return
end
function stream2_arg!(dst, src, nx, ny, nz, bx, by, bz)
    i = (gidx().x - 1) * (bx % typeof(lidx().x)) + lidx().x
    j = (gidx().y - 1) * (by % typeof(lidx().y)) + lidx().y
    k = (gidx().z - 1) * (bz % typeof(lidx().z)) + lidx().z
    (i <= nx && j <= ny && k <= nz) && (@inbounds dst[i, j, k] = src[i, j, k])
    return
end
# The control: identical launch, arrays and traffic, but ONE packet field read,
# so it is scalar in both builds and the fix cannot touch it.
function stream2_mix!(dst, src, nx, ny, nz, by, bz)
    i = (gidx().x - 1) * gdim().x + lidx().x
    j = (gidx().y - 1) * (by % typeof(lidx().y)) + lidx().y
    k = (gidx().z - 1) * (bz % typeof(lidx().z)) + lidx().z
    (i <= nx && j <= ny && k <= nz) && (@inbounds dst[i, j, k] = src[i, j, k])
    return
end

# 6 accesses: five reads and a read-modify-write.
function stream6_dim!(o, a, b, c, d, s, nx, ny, nz)
    i = (gidx().x - 1) * gdim().x + lidx().x
    j = (gidx().y - 1) * gdim().y + lidx().y
    k = (gidx().z - 1) * gdim().z + lidx().z
    if i <= nx && j <= ny && k <= nz
        @inbounds o[i, j, k] += s * (a[i, j, k] - b[i, j, k] + c[i, j, k] - d[i, j, k])
    end
    return
end
function stream6_arg!(o, a, b, c, d, s, nx, ny, nz, bx, by, bz)
    i = (gidx().x - 1) * (bx % typeof(lidx().x)) + lidx().x
    j = (gidx().y - 1) * (by % typeof(lidx().y)) + lidx().y
    k = (gidx().z - 1) * (bz % typeof(lidx().z)) + lidx().z
    if i <= nx && j <= ny && k <= nz
        @inbounds o[i, j, k] += s * (a[i, j, k] - b[i, j, k] + c[i, j, k] - d[i, j, k])
    end
    return
end

# 14 accesses: eight reads plus three read-modify-writes, i.e. a lot of loads in
# flight at the moment the indices are needed.
function stream14_dim!(o1, o2, o3, a, b, c, d, e, f, g, h, s, nx, ny, nz)
    i = (gidx().x - 1) * gdim().x + lidx().x
    j = (gidx().y - 1) * gdim().y + lidx().y
    k = (gidx().z - 1) * gdim().z + lidx().z
    if i <= nx && j <= ny && k <= nz
        @inbounds begin
            o1[i, j, k] += s * (a[i, j, k] - b[i, j, k] + c[i, j, k] - d[i, j, k])
            o2[i, j, k] += s * (e[i, j, k] - f[i, j, k] + g[i, j, k] - h[i, j, k])
            o3[i, j, k] += s * (a[i, j, k] - e[i, j, k] + c[i, j, k] - g[i, j, k])
        end
    end
    return
end
function stream14_arg!(o1, o2, o3, a, b, c, d, e, f, g, h, s, nx, ny, nz, bx, by, bz)
    i = (gidx().x - 1) * (bx % typeof(lidx().x)) + lidx().x
    j = (gidx().y - 1) * (by % typeof(lidx().y)) + lidx().y
    k = (gidx().z - 1) * (bz % typeof(lidx().z)) + lidx().z
    if i <= nx && j <= ny && k <= nz
        @inbounds begin
            o1[i, j, k] += s * (a[i, j, k] - b[i, j, k] + c[i, j, k] - d[i, j, k])
            o2[i, j, k] += s * (e[i, j, k] - f[i, j, k] + g[i, j, k] - h[i, j, k])
            o3[i, j, k] += s * (a[i, j, k] - e[i, j, k] + c[i, j, k] - g[i, j, k])
        end
    end
    return
end

# Short-kernel regime: too little work per launch to amortise wave entry.
function churn_dim!(A, n)
    i = (gidx().x - 1) * gdim().x + lidx().x
    j = (gidx().y - 1) * gdim().y + lidx().y
    k = (gidx().z - 1) * gdim().z + lidx().z
    m = i + j + k
    m <= n && (@inbounds A[m] = one(eltype(A)))
    return
end
function churn_arg!(A, n, bx, by, bz)
    i = (gidx().x - 1) * (bx % typeof(lidx().x)) + lidx().x
    j = (gidx().y - 1) * (by % typeof(lidx().y)) + lidx().y
    k = (gidx().z - 1) * (bz % typeof(lidx().z)) + lidx().z
    m = i + j + k
    m <= n && (@inbounds A[m] = one(eltype(A)))
    return
end

# --- optional ParallelStencil section --------------------------------------
# Each statement is its own top-level block on purpose: `@static if` fully
# macro-expands its body, so `using ParallelStencil` has to have run in an
# earlier statement before @init_parallel_stencil can be expanded.

@static if USE_PS
    using ParallelStencil
end
@static if USE_PS && BACKEND === :CUDA
    @init_parallel_stencil(CUDA, T, 3)
end
@static if USE_PS && BACKEND === :AMDGPU
    @init_parallel_stencil(AMDGPU, T, 3)
end
@static if USE_PS
    @parallel_indices (i, j, k) function ps_stream2!(dst, src)
        @inbounds dst[i, j, k] = src[i, j, k]
        return
    end
    function ps_launch!(dst, src, nx, ny, nz)
        @parallel (1:nx, 1:ny, 1:nz) ps_stream2!(dst, src)
        return
    end
end

# --- ISA evidence ----------------------------------------------------------
# The dispatch pointer arrives in a user SGPR pair whose index depends on which
# other user SGPRs the kernel enabled, so it is read off the kernel descriptor
# rather than assumed. User SGPRs are allocated in a fixed order, and only the
# private segment buffer (4 registers) can precede the dispatch pointer (2).

const LOAD_RE = r"^(s_load|global_load|flat_load|buffer_load)_\w+"

function dispatch_reg(asm)
    enabled(name) = (m = match(Regex("\\.amdhsa_user_sgpr_" * name * "\\s+(\\d+)"), asm);
                     m === nothing ? 0 : parse(Int, m[1]))
    enabled("dispatch_ptr") == 1 || return nothing
    base = 4 * enabled("private_segment_buffer")
    return "s[$base:$(base + 1)]"
end

"""Loads that read through the dispatch pointer, as mnemonics, in program order.

A register pair counts only where it is a *source*: `s_load_dwordx2 s[0:1], ...`
writes the pair and is somebody else's load, not a packet read.
"""
function packet_loads(f, tt)
    io = IOBuffer()
    @static if BACKEND === :AMDGPU
        AMDGPU.code_gcn(io, f, tt; kernel = true)
        asm = String(take!(io))
        reg = dispatch_reg(asm)
        reg === nothing && return String[]
        out = String[]
        for l in eachline(IOBuffer(asm))
            ls = strip(l)
            m = match(LOAD_RE, ls)
            m === nothing && continue
            c = findfirst(',', ls)
            c === nothing && continue
            occursin(reg, ls[c:end]) || continue     # source operands only
            push!(out, m.match)
        end
        return out
    else
        CUDA.code_ptx(io, f, tt; kernel = true)
        return occursin("%ntid", String(take!(io))) ? ["%ntid (register)"] : String[]
    end
end

isvector(l) = startswith(l, "global_load") || startswith(l, "flat_load") ||
              startswith(l, "buffer_load")

classify(loads) =
    isempty(loads)                   ? "none" :
    any(isvector, loads)             ? "VECTOR" :
    all(startswith("s_load"), loads) ? "scalar" : "other"

# --- harness ---------------------------------------------------------------

function timeone(run!)
    devsync()
    t0 = time_ns()
    run!()
    devsync()
    return (time_ns() - t0) * 1e-9
end

"""Time several variants round-robin, so clock and power drift hits them equally.

Taking a minimum per variant and only then a ratio is not enough on a part that
ramps: whichever variant is measured while the clocks are still climbing loses.
`scatter` is measured against the median, so one scheduling hiccup in one round
does not read as instability.
"""
function benchN(fs, reps)
    for _ in 1:10                            # let the clocks ramp before recording
        for f in fs
            f()
        end
    end
    devsync()
    ts = [Float64[] for _ in fs]
    for _ in 1:reps, (i, f) in enumerate(fs)
        push!(ts[i], timeone(f))
    end
    spread(t) = (sort(t)[cld(length(t), 2)] - minimum(t)) / minimum(t)
    return [minimum(t) for t in ts], [spread(t) for t in ts]
end

# So a pasted run identifies the stack it came from. The ISA table is the ground
# truth for which lowering is present; this only names the stack.
function provenance()
    try
        for (_, p) in Pkg.dependencies()
            p.name == string(BACKEND) || continue
            return "v$(p.version)  [" * (p.is_tracking_path ? p.source :
                   p.is_tracking_repo ? "$(p.git_source) @ $(p.git_revision)" : "registry") * "]"
        end
    catch
    end
    return "version unknown"
end

# Keyed by device, geometry and precision as well as by build: ratios from
# different sizes are not comparable (below last-level-cache capacity the kernels
# stop being DRAM-bound and every ratio inflates, the control included), so a run
# must not be able to pick up a saved run it does not belong with.
resultfile(tag) = joinpath(@__DIR__,
    "mwe_result_$(BACKEND)_$(replace(devname(), r"[^A-Za-z0-9]" => ""))" *
    "_N$(N)_b$(BX)_$(T)_$(tag).txt")

function save(tag, rows)
    open(resultfile(tag), "w") do io
        println(io, "# ", devname(), "  N=", N, "  block=", BX, "  ", T, "  ", provenance())
        for (k, acc, loads, val, ratio) in rows
            println(io, k, ",", acc, ",", loads, ",", val, ",", ratio)
        end
    end
end

function load(tag)
    f = resultfile(tag)
    isfile(f) || return nothing
    hdr, rows = "", Dict{String, Tuple{Int, Float64, Float64}}()
    for l in eachline(f)
        if startswith(l, "#")
            hdr = strip(l, ['#', ' '])
        else
            p = split(l, ',')
            rows[p[1]] = (parse(Int, p[3]), parse(Float64, p[4]), parse(Float64, p[5]))
        end
    end
    return hdr, rows
end

function main()
    nx = ny = nz = N
    cells = nx * ny * nz
    A1, A2, A3 = devarray(1), devarray(2), devarray(3)

    println("julia      : $VERSION")
    println("backend    : $BACKEND $(provenance())")
    println("device     : $(devname())")
    println("precision  : $T")
    println("grid       : $(nx)x$(ny)x$(nz) = $cells cells, block ($BX,1,1)")
    println("arrays     : $(round(cells * sizeof(T) / 2^20, digits = 1)) MiB each, 14 of them")
    println("reps       : $REPS (best of)")

    isa_rows = [
        ("1 (x)", packet_loads(dims1!, Tuple{A1, Int})),
        ("2 (x,y)", packet_loads(dims2!, Tuple{A2, Int, Int})),
        ("3 (x,y,z)", packet_loads(stream2_dim!, Tuple{A3, A3, Int, Int, Int})),
    ]
    println("\nISA -- loads that read through the dispatch pointer:\n")
    println(rpad("dims read", 11), rpad("path", 9), "instructions")
    println("-"^64)
    for (d, loads) in isa_rows
        println(rpad(d, 11), rpad(classify(loads), 9), isempty(loads) ? "-" : join(loads, " + "))
    end
    vec = any(r -> classify(r[2]) == "VECTOR", isa_rows)
    tag = vec ? "unfixed" : "fixed"
    println("\nthis build: ", vec ?
        "UNFIXED -- reading more than one dimension leaves the scalar path" :
        "FIXED -- every packet read is scalar")

    o1, o2, o3 = Arr(zeros(T, nx, ny, nz)), Arr(zeros(T, nx, ny, nz)), Arr(zeros(T, nx, ny, nz))
    a, b, c, d = (Arr(fill(T(i), nx, ny, nz)) for i in 1:4)
    e, f, g, h = (Arr(fill(T(i), nx, ny, nz)) for i in 5:8)
    s = T(0.5)
    nb = (cld(nx, BX), ny, nz)
    acc(n) = n * cells * sizeof(T)
    okwrote(x) = all(isfinite, x) && !all(iszero, x)

    variants = [
        ("stream2/dim", 2, () -> (@launch nb (BX, 1, 1) stream2_dim!(o1, a, nx, ny, nz)),
         Tuple{A3, A3, Int, Int, Int}, stream2_dim!, () -> okwrote(o1)),
        ("stream2/mix", 2, () -> (@launch nb (BX, 1, 1) stream2_mix!(o1, a, nx, ny, nz, 1, 1)),
         Tuple{A3, A3, Int, Int, Int, Int, Int}, stream2_mix!, () -> okwrote(o1)),
        ("stream2/arg", 2, () -> (@launch nb (BX, 1, 1) stream2_arg!(o1, a, nx, ny, nz, BX, 1, 1)),
         Tuple{A3, A3, Int, Int, Int, Int, Int, Int}, stream2_arg!, () -> okwrote(o1)),
        ("stream6/dim", 6, () -> (@launch nb (BX, 1, 1) stream6_dim!(o2, a, b, c, d, s, nx, ny, nz)),
         Tuple{A3, A3, A3, A3, A3, T, Int, Int, Int}, stream6_dim!, () -> okwrote(o2)),
        ("stream6/arg", 6,
         () -> (@launch nb (BX, 1, 1) stream6_arg!(o2, a, b, c, d, s, nx, ny, nz, BX, 1, 1)),
         Tuple{A3, A3, A3, A3, A3, T, Int, Int, Int, Int, Int, Int}, stream6_arg!, () -> okwrote(o2)),
        ("stream14/dim", 14,
         () -> (@launch nb (BX, 1, 1) stream14_dim!(o1, o2, o3, a, b, c, d, e, f, g, h, s, nx, ny, nz)),
         Tuple{A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, T, Int, Int, Int},
         stream14_dim!, () -> okwrote(o3)),
        ("stream14/arg", 14,
         () -> (@launch nb (BX, 1, 1) stream14_arg!(o1, o2, o3, a, b, c, d, e, f, g, h, s,
             nx, ny, nz, BX, 1, 1)),
         Tuple{A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, T, Int, Int, Int, Int, Int, Int},
         stream14_arg!, () -> okwrote(o3)),
    ]

    mins, scat = benchN([v[3] for v in variants], REPS)
    base = Dict(split(k, '/')[1] => acc(n) / mins[i] / 1e9
                for (i, (k, n, _, _, _, _)) in enumerate(variants) if endswith(k, "/arg"))

    println("\n", rpad("kernel", 13), rpad("acc", 5), rpad("loads", 7), rpad("path", 9),
        lpad("GB/s", 9), lpad("vs arg", 9), lpad("scatter", 9), "  correct")
    println("-"^72)
    rows = Tuple{String, Int, Int, Float64, Float64}[]
    for (i, (k, n, run!, tt, fn, ok)) in enumerate(variants)
        loads = packet_loads(fn, tt)
        gbs = acc(n) / mins[i] / 1e9
        fam = split(k, '/')[1]
        run!(); devsync()
        push!(rows, (k, n, length(loads), gbs, base[fam] / gbs))
        println(rpad(k, 13), rpad(n, 5), rpad(length(loads), 7), rpad(classify(loads), 9),
            lpad(round(gbs, digits = 1), 9),
            lpad(string(round(base[fam] / gbs, digits = 2), "x"), 9),
            lpad(string(round(100 * scat[i], digits = 1), "%"), 9), "  ", ok() ? "yes" : "NO")
    end

    @static if USE_PS
        pmin, pscat = benchN([() -> ps_launch!(o1, a, nx, ny, nz)], REPS)
        gp = acc(2) / pmin[1] / 1e9
        push!(rows, ("PS/parallel", 2, 0, gp, base["stream2"] / gp))
        println(rpad("PS/parallel", 13), rpad(2, 5), rpad("-", 7), rpad("-", 9),
            lpad(round(gp, digits = 1), 9),
            lpad(string(round(base["stream2"] / gp, digits = 2), "x"), 9),
            lpad(string(round(100 * pscat[1], digits = 1), "%"), 9),
            "  ", okwrote(o1) ? "yes" : "NO")
    end

    # Short-kernel regime, reported separately because its unit is per launch and
    # its story is different: there the read's own latency dominates, so the fix
    # recovers only the part that comes from taking the vector path to get it.
    small = Arr(zeros(T, 1 << 20))
    nsm = length(small)
    cmins, cscat = benchN([
            () -> (for _ in 1:CHURN
                (@launch (8, 8, 8) (64, 1, 1) churn_dim!(small, nsm))
            end),
            () -> (for _ in 1:CHURN
                (@launch (8, 8, 8) (64, 1, 1) churn_arg!(small, nsm, 64, 1, 1))
            end)], max(3, REPS ÷ 5))
    cd_, ca_ = cmins[1] / CHURN * 1e6, cmins[2] / CHURN * 1e6
    println("\nshort-kernel regime ($CHURN launches of a 512-workitem kernel):")
    println("  dim ", round(cd_, digits = 2), " us/launch,  arg ", round(ca_, digits = 2),
        " us/launch,  dim/arg ", round(cd_ / ca_, digits = 2), "x",
        "  (scatter ", round(100 * maximum(cscat), digits = 1), "%)")
    push!(rows, ("churn", 0, 0, cd_, cd_ / ca_))

    println("\nstream2/mix is the control: same launch, arrays and traffic as",
        "\nstream2/dim, one packet field instead of three, scalar in both builds.",
        "\nThe defect's cost is what the 6- and 14-access rows do that it does not.")

    save(tag, rows)
    other = load(tag == "fixed" ? "unfixed" : "fixed")
    if other === nothing
        println("\nSaved to $(basename(resultfile(tag))). Run the other build for the",
            "\nbefore/after table.")
    else
        hdr, prev = other
        # Ratios, not GB/s: two runs minutes apart on a clock-throttled part drift
        # by more than the effect. Each ratio is against its own run's `arg` row,
        # so drift cancels inside it; `recovered` is what the fix took back.
        println("\nAcross the two builds, as the cost of the packet read",
            "\n[other run: $hdr]:\n")
        println(rpad("kernel", 13), rpad("loads", 12), lpad("unfixed", 10), lpad("fixed", 8),
            lpad("recovered", 11))
        println("-"^54)
        for (k, _, loads, _, ratio) in rows
            haskey(prev, k) || continue
            pl, _, pr = prev[k]
            (ul, ur), (fl, fr) = tag == "fixed" ? ((pl, pr), (loads, ratio)) :
                                                  ((loads, ratio), (pl, pr))
            println(rpad(k, 13), rpad("$ul -> $fl", 12),
                lpad(string(round(ur, digits = 2), "x"), 10),
                lpad(string(round(fr, digits = 2), "x"), 8),
                lpad(string(ur > fr ? "+" : "", round(100 * (ur / fr - 1), digits = 1), "%"), 11))
        end
    end
    return nothing
end

main()
