"""
DiffiScape — Julia backend for the DiffiScape R package.

Provides connectivity computation (Omniscape, Circuitscape) and a stub
for Enzyme.jl automatic differentiation of connectivity w.r.t. resistance.
"""
module DiffiScape

include("differentiable_solver.jl")
include("connectivity.jl")
include("enzyme_gradients.jl")
include("setup.jl")

end # module
