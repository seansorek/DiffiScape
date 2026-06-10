"""
Enzyme.jl automatic differentiation for the differentiable circuit solver.

Provides:
- `enzyme_available()`: check if Enzyme.jl is installed
- `cumulative_current_vjp()`: VJP through the cumulative current computation
- `resistance_gradient()`: chain rule from ∂loss/∂R to ∂loss/∂params
- `enzyme_gradient()`: full gradient of connectivity w.r.t. resistance params
- `enzyme_hvp()`: Hessian-vector product via finite differences on gradient

The VJP uses per-window Enzyme reverse-mode AD, keeping memory bounded
by the window size rather than the full grid.
"""

export enzyme_gradient, enzyme_available, enzyme_hvp,
       cumulative_current_vjp, resistance_gradient,
       enzyme_gradient_generic, resistance_gradient_generic


"""
    enzyme_available() -> Bool

Check whether Enzyme.jl is installed and loadable.
"""
function enzyme_available()::Bool
    try
        @eval import Enzyme
        return true
    catch
        return false
    end
end


# ========================== VJP Through Cumulative Current ==================

"""
    single_window_scalar(R_local, d_cum_local, cr, cc, r1, c1, config)

Compute `dot(current(R_local), d_cum_local)` for a single window.
This is the function Enzyme differentiates to get the per-window VJP.
"""
function single_window_scalar(R_local::Matrix{Float64},
                               d_cum_local::Matrix{Float64},
                               cr::Int, cc::Int, r1::Int, c1::Int,
                               config::SolverConfig)
    current = solve_single_window(R_local, cr, cc, r1, c1, config).current
    s = 0.0
    @inbounds for i in eachindex(current, d_cum_local)
        s += current[i] * d_cum_local[i]
    end
    return s
end


"""
    cumulative_current_vjp(R, d_cum, config)

Vector-Jacobian product through `cumulative_current`.

Given `d_cum = ∂loss/∂cum_current`, computes `∂loss/∂R` by running
Enzyme reverse-mode AD on each window independently. This keeps
peak memory proportional to a single window rather than the full grid.

Requires Enzyme.jl to be installed.
"""
function cumulative_current_vjp(R::Matrix{Float64},
                                 d_cum::Matrix{Float64},
                                 config::SolverConfig)
    if !enzyme_available()
        error("Enzyme.jl is required for gradient computation. Install with: ] add Enzyme")
    end

    @eval import Enzyme

    nrows, ncols = size(R)
    centers = compute_block_centers(nrows, ncols, config.block_size)
    dR = zeros(nrows, ncols)

    for (cr, cc) in centers
        r1, r2, c1, c2 = window_bounds(cr, cc, config.radius, nrows, ncols)
        wr = r2 - r1 + 1
        wc = c2 - c1 + 1

        d_cum_local = d_cum[r1:r2, c1:c2]

        # Skip windows where cotangent is all zero
        any_nonzero = false
        @inbounds for j in eachindex(d_cum_local)
            if d_cum_local[j] != 0.0
                any_nonzero = true
                break
            end
        end
        if !any_nonzero
            continue
        end

        R_local = R[r1:r2, c1:c2]

        # Enzyme gradient of the scalar function w.r.t. R_local.
        # Wrap the closure in Const so Enzyme doesn't treat captured variables
        # (d_cum_local, config, etc.) as potentially holding derivative data,
        # avoiding EnzymeMutabilityException in Enzyme ≥ 0.13.
        dR_local = zero(R_local)
        f_local = Rl -> single_window_scalar(Rl, d_cum_local, cr, cc, r1, c1, config)
        Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(f_local), Enzyme.Active,
                        Enzyme.Duplicated(copy(R_local), dR_local))

        @inbounds for jj in 1:wc, ii in 1:wr
            dR[r1 + ii - 1, c1 + jj - 1] += dR_local[ii, jj]
        end
    end

    return dR
end


# ========================== Chain Rule: ∂R → ∂params ========================

"""
    resistance_gradient(params, basis_values, dR_flat, R_flat, config)

Chain rule from `∂loss/∂R` to `∂loss/∂params`.

The resistance parameterisation is:
    `R = clamp(exp(r₀ + Σ zₖ·φₖ), R_min, R_max)`

So `∂loss/∂r₀ = Σᵢ (∂loss/∂Rᵢ)·Rᵢ` and
   `∂loss/∂zₖ = Σᵢ (∂loss/∂Rᵢ)·Rᵢ·φₖ(i)` for unclamped cells.

# Arguments
- `params`: `[r₀, z₁, ..., zₖ]`
- `basis_values`: `n_cells × K` matrix of basis function values (flat, row-major)
- `dR_flat`: `∂loss/∂R` as a flat vector (row-major, same order as basis_values)
- `R_flat`: resistance values as a flat vector (row-major)
- `config`: solver configuration (for R_min, R_max)
"""
function resistance_gradient(params::Vector{Float64},
                              basis_values::Matrix{Float64},
                              dR_flat::Vector{Float64},
                              R_flat::Vector{Float64},
                              config::SolverConfig)
    # Delegate to generic form with dR/dη = R (exp link)
    n_cells = length(R_flat)
    dR_deta = zeros(n_cells)
    @inbounds for i in 1:n_cells
        ri = R_flat[i]
        if ri > config.R_min && ri < config.R_max
            dR_deta[i] = ri
        end
    end
    return resistance_gradient_generic(basis_values, dR_flat, dR_deta)
end


"""
    resistance_gradient_generic(basis_values, dR_flat, dR_deta)

Link-agnostic chain rule from `∂loss/∂R` to `∂loss/∂params`.

The gradient w.r.t. additive parameters `[r₀, z₁, ..., zₖ]` is:
    `∂loss/∂r₀ = Σᵢ (∂loss/∂Rᵢ)·(dR/dη)ᵢ`
    `∂loss/∂zₖ = Σᵢ (∂loss/∂Rᵢ)·(dR/dη)ᵢ·φₖ(i)`

R is responsible for computing `dR_deta` via the link object's `deriv_fn`.
This function only applies the chain rule with the provided derivative.

# Arguments
- `basis_values`: `n_cells × K` matrix of basis function values
- `dR_flat`: `∂loss/∂R` as a flat vector
- `dR_deta`: `dR/dη` as a flat vector (precomputed by R-side link)
"""
function resistance_gradient_generic(basis_values::Matrix{Float64},
                                      dR_flat::Vector{Float64},
                                      dR_deta::Vector{Float64})
    n_basis = size(basis_values, 2)
    n_params = n_basis + 1
    grad = zeros(n_params)

    @inbounds for i in eachindex(dR_flat, dR_deta)
        chain = dR_flat[i] * dR_deta[i]
        grad[1] += chain
        for k in 1:n_basis
            grad[k + 1] += chain * basis_values[i, k]
        end
    end

    return grad
end


# ========================== Full Gradient ===================================

"""
    enzyme_gradient(params, basis_values, nrows, ncols, config)

Compute the gradient of `sum(log1p(cumulative_current))` w.r.t.
resistance `params` using Enzyme.jl.  Uses the hardcoded exp link.

See also [`enzyme_gradient_generic`](@ref) for the link-agnostic version.

Requires Enzyme.jl.
"""
function enzyme_gradient(params::Vector{Float64},
                          basis_values::Matrix{Float64},
                          nrows::Int, ncols::Int,
                          config::SolverConfig)
    n_cells = nrows * ncols
    n_basis = size(basis_values, 2)

    # Forward: params → R (flat, row-major) — exp link
    R_flat = zeros(n_cells)
    @inbounds for i in 1:n_cells
        log_r = params[1]
        for k in 1:n_basis
            log_r += params[k + 1] * basis_values[i, k]
        end
        R_flat[i] = clamp(exp(log_r), config.R_min, config.R_max)
    end

    return enzyme_gradient_generic(R_flat, R_flat, basis_values,
                                    nrows, ncols, config)
end


"""
    enzyme_gradient_generic(R_flat, dR_deta, basis_values, nrows, ncols, config)

Link-agnostic gradient of `sum(log1p(cumulative_current))` w.r.t.
resistance parameters.

R precomputes:
- `R_flat`: resistance values (flat, row-major)
- `dR_deta`: derivative of the link function `dR/dη` at each cell

Julia computes the VJP through the solver and applies the chain rule
with the provided `dR_deta`.

Requires Enzyme.jl.

# Arguments
- `R_flat`: precomputed resistance as a flat vector (row-major)
- `dR_deta`: precomputed dR/dη as a flat vector (row-major)
- `basis_values`: `n_cells × K` matrix
- `nrows, ncols`: grid dimensions
- `config`: solver configuration

# Returns
Gradient vector of length `K+1`.
"""
function enzyme_gradient_generic(R_flat::Vector{Float64},
                                  dR_deta::Vector{Float64},
                                  basis_values::Matrix{Float64},
                                  nrows::Int, ncols::Int,
                                  config::SolverConfig)

    # Reshape to column-major matrix (row-major vec → matrix)
    R_mat = Matrix(reshape(R_flat, ncols, nrows)')

    # Forward: cumulative current — always force output="current" here since the
    # VJP objective only needs current density, regardless of config.output.
    current_config = SolverConfig(
        radius               = config.radius,
        block_size           = config.block_size,
        source_from_resistance = config.source_from_resistance,
        R_min                = config.R_min,
        R_max                = config.R_max,
        cg_tol               = config.cg_tol,
        cg_maxiter           = config.cg_maxiter,
        ground_conductance   = config.ground_conductance,
        output               = "current",
    )
    cum_current = cumulative_current(R_mat, current_config)

    # Objective: sum of log1p(current) over all valid cells
    # Cotangent: ∂obj/∂C
    d_cum = zeros(nrows, ncols)
    @inbounds for c in 1:ncols, r in 1:nrows
        cv = cum_current[r, c]
        if cv > 0.0
            d_cum[r, c] = 1.0 / (1.0 + cv)
        end
    end

    # VJP: ∂obj/∂R (as matrix)
    dR_mat = cumulative_current_vjp(R_mat, d_cum, config)

    # Convert dR_mat back to flat row-major (collect: adjoint is lazy, vec returns
    # ReshapedArray which doesn't match Vector{Float64} type annotation)
    dR_flat = collect(vec(dR_mat'))

    # Chain rule: ∂obj/∂params
    return resistance_gradient_generic(basis_values, dR_flat, dR_deta)
end


# ========================== Hessian-Vector Product ==================================================

"""
    enzyme_hvp(params, v, basis_values, nrows, ncols, config; eps=1e-5)

Hessian-vector product `H·v` via central finite differences on the gradient.
Uses the hardcoded exp link path.

`H·v ≈ (∇f(params + ε·v) - ∇f(params - ε·v)) / (2ε)`

Requires Enzyme.jl (calls `enzyme_gradient` twice).
"""
function enzyme_hvp(params::Vector{Float64},
                     v::Vector{Float64},
                     basis_values::Matrix{Float64},
                     nrows::Int, ncols::Int,
                     config::SolverConfig;
                     eps::Float64=1e-5)
    g_plus  = enzyme_gradient(params .+ eps .* v, basis_values, nrows, ncols, config)
    g_minus = enzyme_gradient(params .- eps .* v, basis_values, nrows, ncols, config)
    return (g_plus .- g_minus) ./ (2.0 * eps)
end
