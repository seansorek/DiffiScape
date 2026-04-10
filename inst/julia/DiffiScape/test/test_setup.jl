using Test

@testset "setup.jl" begin

    @testset "check_dependencies returns Dict{String,Bool}" begin
        deps = check_dependencies()
        @test deps isa Dict{String,Bool}
        @test haskey(deps, "Omniscape")
        @test haskey(deps, "Circuitscape")
        @test haskey(deps, "Enzyme")
    end

    @testset "get_version_info returns Dict{String,String}" begin
        info = get_version_info()
        @test info isa Dict{String,String}
        @test haskey(info, "julia")
        @test startswith(info["julia"], string(VERSION.major))
    end

    @testset "enzyme_available consistent with check_dependencies" begin
        deps = check_dependencies()
        @test deps["Enzyme"] == enzyme_available()
    end

end
