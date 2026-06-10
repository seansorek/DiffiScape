using Test

# Include individual source files directly rather than the full module,
# which requires Omniscape/Circuitscape that may not be installed.
const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "differentiable_solver.jl"))
include(joinpath(SRC, "enzyme_gradients.jl"))

# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------
include("test_differentiable_solver.jl")
include("test_enzyme_gradients.jl")

# Setup and connectivity tests — included unconditionally; each test inside
# guards against missing packages via @test_skip.
try
    include(joinpath(SRC, "setup.jl"))
    include("test_setup.jl")
catch e
    @warn "Skipping setup tests: $e"
end

include("test_connectivity.jl")
