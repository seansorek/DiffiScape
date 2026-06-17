"""
Differentiable circuit-theory solver for landscape connectivity.

Pure-Julia, Enzyme.jl-compatible implementation of:
- Grid-based conductance from resistance (harmonic mean)
- Matrix-free Laplacian via 4-connected stencil
- Preconditioned conjugate gradient solver
- Moving-window cumulative current (Omniscape-equivalent)

Designed for grids >500×500 with threaded window parallelism.
"""

export SolverConfig, cumulative_current, solve_single_window,
       compute_block_centers, window_bounds

# ========================== Configuration ===================================

"""
    SolverConfig(; kwargs...)

Configuration for the differentiable circuit solver.

# Fields
- `radius::Int`: Moving-window radius in pixels (default 13).
- `block_size::Int`: Source block side length in pixels (default 5).
- `source_from_resistance::Bool`: Use 1/R as source strength (default true).
- `R_min::Float64`: Minimum resistance clamp (default 1.0).
- `R_max::Float64`: Maximum resistance clamp (default 5000.0).
- `cg_tol::Float64`: CG relative tolerance (default 1e-8).
- `cg_maxiter::Int`: CG maximum iterations (default 2000).
- `ground_conductance::Float64`: Penalty for grounding boundary cells (default 1e10).
"""
Base.@kwdef struct SolverConfig
    radius::Int              = 13
    block_size::Int          = 5
    source_from_resistance::Bool = true
    R_min::Float64           = 1.0
    R_max::Float64           = 5000.0
    cg_tol::Float64          = 1e-8
    cg_maxiter::Int          = 2000
    ground_conductance::Float64 = 1e10
    output::String           = "current"  # "current" | "voltage" | "both"
end


# ========================== Grid Conductance ================================

"""
    resistance_to_conductance(R::Matrix{Float64})

Compute edge conductances from resistance using harmonic mean.

Returns `(edge_h, edge_v)` where:
- `edge_h[r,c]` = conductance between cells `(r,c)` and `(r,c+1)`
- `edge_v[r,c]` = conductance between cells `(r,c)` and `(r+1,c)`

Edges touching cells with R ≤ 0 or non-finite R get conductance 0.
"""
function resistance_to_conductance(R::Matrix{Float64})
    nrows, ncols = size(R)
    edge_h = zeros(nrows, max(ncols - 1, 0))
    edge_v = zeros(max(nrows - 1, 0), ncols)

    @inbounds for c in 1:ncols-1, r in 1:nrows
        ri = R[r, c]
        rj = R[r, c + 1]
        if ri > 0.0 && rj > 0.0 && isfinite(ri) && isfinite(rj)
            edge_h[r, c] = 2.0 / (ri + rj)
        end
    end

    @inbounds for c in 1:ncols, r in 1:nrows-1
        ri = R[r, c]
        rj = R[r + 1, c]
        if ri > 0.0 && rj > 0.0 && isfinite(ri) && isfinite(rj)
            edge_v[r, c] = 2.0 / (ri + rj)
        end
    end

    return edge_h, edge_v
end


# ========================== Matrix-Free Laplacian ===========================

"""
    laplacian_matvec!(y, x, edge_h, edge_v, ground_mask, g_cond)

Compute `y = (L + g_cond · diag(ground_mask)) · x` without forming L.

Uses a 4-connected stencil: for each cell `(r,c)`,
`(Lx)[r,c] = Σ_j G[r,c,j] · (x[r,c] - x[j])` over the four neighbors.
Ground cells get an additional diagonal term `g_cond · x[r,c]`.
"""
function laplacian_matvec!(y::Matrix{Float64}, x::Matrix{Float64},
                           edge_h::Matrix{Float64}, edge_v::Matrix{Float64},
                           ground_mask::AbstractMatrix{Bool}, g_cond::Float64)
    nrows, ncols = size(x)
    @inbounds for c in 1:ncols, r in 1:nrows
        val = 0.0
        xi = x[r, c]

        if c < ncols
            g = edge_h[r, c]
            val += g * (xi - x[r, c + 1])
        end
        if c > 1
            g = edge_h[r, c - 1]
            val += g * (xi - x[r, c - 1])
        end
        if r < nrows
            g = edge_v[r, c]
            val += g * (xi - x[r + 1, c])
        end
        if r > 1
            g = edge_v[r - 1, c]
            val += g * (xi - x[r - 1, c])
        end

        if ground_mask[r, c]
            val += g_cond * xi
        end

        y[r, c] = val
    end
    return nothing
end


# ========================== Diagonal Preconditioner =========================

"""
    compute_diag_inv!(diag_inv, edge_h, edge_v, ground_mask, g_cond)

Compute the inverse diagonal of the modified Laplacian for preconditioning.
`M_ii = 1 / L_ii` where `L_ii = Σ_j G_ij + g_cond · ground_mask[i]`.
"""
function compute_diag_inv!(diag_inv::Matrix{Float64},
                            edge_h::Matrix{Float64},
                            edge_v::Matrix{Float64},
                            ground_mask::AbstractMatrix{Bool},
                            g_cond::Float64)
    nrows, ncols = size(diag_inv)
    @inbounds for c in 1:ncols, r in 1:nrows
        d = 0.0
        if c < ncols; d += edge_h[r, c]; end
        if c > 1;     d += edge_h[r, c - 1]; end
        if r < nrows; d += edge_v[r, c]; end
        if r > 1;     d += edge_v[r - 1, c]; end
        if ground_mask[r, c]
            d += g_cond
        end
        diag_inv[r, c] = d > 0.0 ? 1.0 / d : 0.0
    end
    return nothing
end


# ========================== CG Solver =======================================

"""
    cg_solve(b, edge_h, edge_v, ground_mask, g_cond; tol, maxiter)

Preconditioned conjugate gradient solve for `(L + g_cond·D_g) v = b`.

Returns the solution vector `v` (same shape as `b`).
Uses diagonal preconditioning for the grid Laplacian.
"""
function cg_solve(b::Matrix{Float64},
                  edge_h::Matrix{Float64}, edge_v::Matrix{Float64},
                  ground_mask::AbstractMatrix{Bool}, g_cond::Float64;
                  tol::Float64=1e-8, maxiter::Int=2000)
    wr, wc = size(b)
    v  = zeros(wr, wc)
    r  = copy(b)
    Ap = zeros(wr, wc)

    # Preconditioner
    diag_inv = zeros(wr, wc)
    compute_diag_inv!(diag_inv, edge_h, edge_v, ground_mask, g_cond)

    # z = M^{-1} r
    z = zeros(wr, wc)
    @inbounds for i in eachindex(z, r, diag_inv)
        z[i] = diag_inv[i] * r[i]
    end

    p = copy(z)

    rz = 0.0
    @inbounds for i in eachindex(r, z)
        rz += r[i] * z[i]
    end

    bnorm = 0.0
    @inbounds for i in eachindex(b)
        bnorm += b[i] * b[i]
    end
    bnorm = sqrt(bnorm)
    if bnorm < 1e-15
        return v
    end
    tol_abs = tol * bnorm

    for _ in 1:maxiter
        laplacian_matvec!(Ap, p, edge_h, edge_v, ground_mask, g_cond)

        pAp = 0.0
        @inbounds for i in eachindex(p, Ap)
            pAp += p[i] * Ap[i]
        end
        if pAp < 1e-30
            break
        end

        alpha = rz / pAp

        @inbounds for i in eachindex(v, p)
            v[i] += alpha * p[i]
        end
        @inbounds for i in eachindex(r, Ap)
            r[i] -= alpha * Ap[i]
        end

        rnorm_sq = 0.0
        @inbounds for i in eachindex(r)
            rnorm_sq += r[i] * r[i]
        end
        if sqrt(rnorm_sq) < tol_abs
            break
        end

        @inbounds for i in eachindex(z, r, diag_inv)
            z[i] = diag_inv[i] * r[i]
        end

        rz_new = 0.0
        @inbounds for i in eachindex(r, z)
            rz_new += r[i] * z[i]
        end

        beta = rz_new / rz

        @inbounds for i in eachindex(p, z)
            p[i] = z[i] + beta * p[i]
        end

        rz = rz_new
    end

    return v
end


# ========================== Current Computation =============================

"""
    compute_current(v, edge_h, edge_v)

Compute per-cell current magnitude from voltages and conductances.

For each cell: `I[r,c] = 0.5 · Σ_j |G_j · (v[r,c] - v[j])|`
(half because each edge is counted from both endpoints).
"""
function compute_current(v::Matrix{Float64},
                         edge_h::Matrix{Float64},
                         edge_v::Matrix{Float64})
    nrows, ncols = size(v)
    current = zeros(nrows, ncols)
    @inbounds for c in 1:ncols, r in 1:nrows
        vi = v[r, c]
        cur = 0.0
        if c < ncols
            cur += abs(edge_h[r, c] * (vi - v[r, c + 1]))
        end
        if c > 1
            cur += abs(edge_h[r, c - 1] * (vi - v[r, c - 1]))
        end
        if r < nrows
            cur += abs(edge_v[r, c] * (vi - v[r + 1, c]))
        end
        if r > 1
            cur += abs(edge_v[r - 1, c] * (vi - v[r - 1, c]))
        end
        current[r, c] = 0.5 * cur
    end
    return current
end


# ========================== Window Utilities ================================

"""
    compute_block_centers(nrows, ncols, block_size)

Compute the (row, col) centers of all source blocks for the moving window.
Blocks are spaced `block_size` apart, starting `block_size÷2 + 1` from edge.
"""
function compute_block_centers(nrows::Int, ncols::Int, block_size::Int)
    half = div(block_size, 2)
    centers = Tuple{Int,Int}[]
    cr = 1 + half
    while cr <= nrows
        cc = 1 + half
        while cc <= ncols
            push!(centers, (cr, cc))
            cc += block_size
        end
        cr += block_size
    end
    return centers
end


"""
    window_bounds(cr, cc, radius, nrows, ncols)

Bounding box `(r1, r2, c1, c2)` of a circular window, clipped to grid.
"""
function window_bounds(cr::Int, cc::Int, radius::Int, nrows::Int, ncols::Int)
    r1 = max(1, cr - radius)
    r2 = min(nrows, cr + radius)
    c1 = max(1, cc - radius)
    c2 = min(ncols, cc + radius)
    return r1, r2, c1, c2
end


"""
    prepare_window!(R_win, ground_mask, source_mask, cr, cc, r1, c1,
                    wr, wc, radius, block_size)

Build masks and modify `R_win` in-place for a single window solve.

- Cells outside the circle or with invalid R: `R_win = 0`, grounded.
- Cells on the circle boundary (within 1 pixel of edge): grounded.
- Center block cells: marked as source.
"""
function prepare_window!(R_win::Matrix{Float64},
                          ground_mask::AbstractMatrix{Bool},
                          source_mask::AbstractMatrix{Bool},
                          cr::Int, cc::Int, r1::Int, c1::Int,
                          wr::Int, wc::Int,
                          radius::Int, block_size::Int)
    half = div(block_size, 2)
    radius_sq = radius * radius
    inner_radius_sq = (radius - 1) * (radius - 1)

    @inbounds for jj in 1:wc, ii in 1:wr
        gr = r1 + ii - 1
        gc = c1 + jj - 1
        dr = gr - cr
        dc = gc - cc
        dist_sq = dr * dr + dc * dc

        r_val = R_win[ii, jj]
        is_valid = isfinite(r_val) && r_val > 0.0
        in_circle = dist_sq <= radius_sq

        if !in_circle || !is_valid
            # Outside circle or nodata: disconnect and ground
            R_win[ii, jj] = 0.0
            ground_mask[ii, jj] = true
            source_mask[ii, jj] = false
        elseif dist_sq > inner_radius_sq
            # Circle boundary: keep R value but ground
            ground_mask[ii, jj] = true
            source_mask[ii, jj] = false
        else
            # Interior cell
            ground_mask[ii, jj] = false
            source_mask[ii, jj] = (abs(dr) <= half && abs(dc) <= half)
        end
    end
    return nothing
end


"""
    build_source_vector(source_mask, R_win, source_from_resistance)

Build the RHS vector for the circuit solve (current injection at sources).
Total injected current is normalized to 1 amp per window.
"""
function build_source_vector(source_mask::AbstractMatrix{Bool},
                              R_win::Matrix{Float64},
                              source_from_resistance::Bool)
    wr, wc = size(source_mask)
    b = zeros(wr, wc)
    total = 0.0

    @inbounds for i in eachindex(source_mask)
        if source_mask[i]
            if source_from_resistance
                r_val = R_win[i]
                s = r_val > 0.0 ? 1.0 / r_val : 0.0
            else
                s = 1.0
            end
            b[i] = s
            total += s
        end
    end

    # Normalize to 1 amp total
    if total > 0.0
        inv_total = 1.0 / total
        @inbounds for i in eachindex(b)
            b[i] *= inv_total
        end
    end

    return b
end


# ========================== Per-Window Solve ================================

"""
    solve_single_window(R_local, cr, cc, r1, c1, config)

Solve the circuit for a single moving window and return the current map.

# Arguments
- `R_local`: Resistance values in the window (will be copied, not modified).
- `cr, cc`: Window center in global grid coordinates.
- `r1, c1`: Top-left corner of the bounding box in global coordinates.
- `config`: Solver configuration.

# Returns
`Matrix{Float64}` of current density, same size as `R_local`.
"""
function solve_single_window(R_local::Matrix{Float64},
                              cr::Int, cc::Int, r1::Int, c1::Int,
                              config::SolverConfig)
    wr, wc = size(R_local)

    # Copy resistance and build masks
    R_win = copy(R_local)
    ground_mask = falses(wr, wc)
    source_mask = falses(wr, wc)
    prepare_window!(R_win, ground_mask, source_mask, cr, cc, r1, c1,
                    wr, wc, config.radius, config.block_size)

    # Check for any source cells
    has_source = false
    @inbounds for i in eachindex(source_mask)
        if source_mask[i]
            has_source = true
            break
        end
    end
    if !has_source
        return (; voltage = zeros(wr, wc), current = zeros(wr, wc))
    end

    # Conductances
    edge_h, edge_v = resistance_to_conductance(R_win)

    # Source injection
    b = build_source_vector(source_mask, R_win, config.source_from_resistance)

    # CG solve
    v = cg_solve(b, edge_h, edge_v, ground_mask, config.ground_conductance;
                 tol=config.cg_tol, maxiter=config.cg_maxiter)

    # Voltage and current density
    current = compute_current(v, edge_h, edge_v)
    return (; voltage = v, current = current)
end


# ========================== Cumulative Current ==============================

"""
    cumulative_current(R::Matrix{Float64}, config::SolverConfig)

Compute the cumulative current map (Omniscape-equivalent).

Iterates over all source blocks within the grid, solves the circuit in
a moving window of `config.radius` around each source, and accumulates
current densities into a global map. Windows are solved in parallel
using `Threads.@threads`.

# Arguments
- `R`: Resistance surface (nrows × ncols). Cells with R ≤ 0 or NaN are nodata.
- `config`: See [`SolverConfig`](@ref).

# Returns
`Matrix{Float64}` of cumulative current, same size as `R`.
"""
function cumulative_current(R::Matrix{Float64}, config::SolverConfig)
    nrows, ncols = size(R)
    centers = compute_block_centers(nrows, ncols, config.block_size)
    n_windows = length(centers)

    want_current = config.output == "current" || config.output == "both"
    want_voltage = config.output == "voltage" || config.output == "both"

    if n_windows == 0
        if config.output == "both"
            return (; current = zeros(nrows, ncols), voltage = zeros(nrows, ncols))
        else
            return zeros(nrows, ncols)
        end
    end

    # maxthreadid() (Julia ≥ 1.11) covers interactive + default pool threads.
    # Older fallback uses nthreads(), which equals maxthreadid() pre-1.11.
    nt = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : max(Threads.nthreads(), 1)
    local_bufs_current = want_current ? [zeros(nrows, ncols) for _ in 1:nt] : nothing
    local_bufs_voltage = want_voltage ? [zeros(nrows, ncols) for _ in 1:nt] : nothing

    Threads.@threads :static for i in 1:n_windows
        tid = Threads.threadid()
        cr, cc = centers[i]
        r1, r2, c1, c2 = window_bounds(cr, cc, config.radius, nrows, ncols)

        # Extract local resistance (creates a copy)
        R_local = R[r1:r2, c1:c2]

        # Solve — returns named tuple (; voltage, current)
        result = solve_single_window(R_local, cr, cc, r1, c1, config)

        # Accumulate into thread-local buffers
        wr = r2 - r1 + 1
        wc = c2 - c1 + 1
        if want_current
            buf = local_bufs_current[tid]
            @inbounds for jj in 1:wc, ii in 1:wr
                buf[r1 + ii - 1, c1 + jj - 1] += result.current[ii, jj]
            end
        end
        if want_voltage
            buf = local_bufs_voltage[tid]
            @inbounds for jj in 1:wc, ii in 1:wr
                buf[r1 + ii - 1, c1 + jj - 1] += result.voltage[ii, jj]
            end
        end
    end

    # Sum thread-local buffers
    if want_current
        cum_current = local_bufs_current[1]
        for t in 2:nt
            @inbounds for i in eachindex(cum_current, local_bufs_current[t])
                cum_current[i] += local_bufs_current[t][i]
            end
        end
    end
    if want_voltage
        cum_voltage = local_bufs_voltage[1]
        for t in 2:nt
            @inbounds for i in eachindex(cum_voltage, local_bufs_voltage[t])
                cum_voltage[i] += local_bufs_voltage[t][i]
            end
        end
    end

    if config.output == "both"
        return (; current = cum_current, voltage = cum_voltage)
    elseif config.output == "voltage"
        return cum_voltage
    else
        return cum_current
    end
end


# ========================== R Bridge Wrapper ================================

"""
    cumulative_current(R, radius, block_size)

Convenience wrapper for the R bridge. Constructs a `SolverConfig` from
positional arguments and calls the main solver. Returns cumulative current.
"""
function cumulative_current(R::Matrix{Float64}, radius::Int, block_size::Int)
    config = SolverConfig(radius=radius, block_size=block_size)
    return cumulative_current(R, config)
end


"""
    cumulative_current(R, radius, block_size, output)

Convenience wrapper for the R bridge with output selection.

# Arguments
- `output`: `"current"`, `"voltage"`, or `"both"`.
  - `"current"` → `Matrix{Float64}` of cumulative current density.
  - `"voltage"` → `Matrix{Float64}` of cumulative voltage (flow potential).
  - `"both"` → named tuple `(; current, voltage)` with both matrices.
"""
function cumulative_current(R::Matrix{Float64}, radius::Int, block_size::Int, output::String)
    config = SolverConfig(radius=radius, block_size=block_size, output=output)
    return cumulative_current(R, config)
end
