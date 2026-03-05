"""
Enzyme.jl automatic differentiation stubs.

This module will provide gradients of connectivity with respect to
resistance parameters via Enzyme.jl reverse-mode AD.  Currently a
placeholder — the main pipeline uses a GP surrogate instead.
"""

export enzyme_gradient, enzyme_available

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


"""
    enzyme_gradient(resistance_params, basis_values, connectivity_fn;
                    R_min=0.01, R_max=1e6)

Compute the gradient of the connectivity objective with respect to
`resistance_params` using Enzyme.jl reverse-mode AD.

# Arguments
- `resistance_params::Vector{Float64}`: Parameter vector `[r_0, z_1, ..., z_K]`.
- `basis_values::Matrix{Float64}`: `n_cells × K` matrix of basis function values.
- `connectivity_fn::Function`: Function mapping a resistance vector to a
  scalar connectivity metric.
- `R_min::Float64`: Minimum resistance clamp.
- `R_max::Float64`: Maximum resistance clamp.

# Returns
A `Vector{Float64}` of gradients, same length as `resistance_params`.

# Status
**Not yet implemented.** Returns an error explaining that Enzyme integration
is pending.
"""
function enzyme_gradient(resistance_params::Vector{Float64},
                         basis_values::Matrix{Float64},
                         connectivity_fn::Function;
                         R_min::Float64=0.01,
                         R_max::Float64=1e6)

    error("""
    Enzyme.jl gradient computation is not yet implemented.

    This function will eventually:
    1. Construct the resistance surface: R(x) = exp(r_0 + Σ z_k * φ_k(x))
    2. Use Enzyme.autodiff(Reverse, ...) to differentiate through
       the connectivity solver w.r.t. the resistance parameters.
    3. Return the gradient vector ∂L/∂θ.

    For now, use the GP surrogate optimiser in R:
        optimize_resistance(basis_stack, obs_points)
    """)
end


"""
    enzyme_hessian(resistance_params, basis_values, connectivity_fn;
                   R_min=0.01, R_max=1e6)

Compute the Hessian of the connectivity objective via Enzyme.jl.
**Not yet implemented.**
"""
function enzyme_hessian(resistance_params::Vector{Float64},
                        basis_values::Matrix{Float64},
                        connectivity_fn::Function;
                        R_min::Float64=0.01,
                        R_max::Float64=1e6)

    error("Enzyme.jl Hessian computation is not yet implemented.")
end
