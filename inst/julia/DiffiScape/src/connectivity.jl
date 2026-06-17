"""
Connectivity computation via Omniscape.jl and Circuitscape.jl.
"""

using Omniscape
using Circuitscape

export run_omniscape, run_circuitscape

"""
    run_omniscape(resistance_file, output_dir, radius, block_size, source_from_resistance)

Run Omniscape on a resistance raster file, writing outputs into
`output_dir/omniscape_run/`. R reads `cum_currmap.tif` and
`flow_potential.tif` from that subdirectory.

# Arguments
- `resistance_file::String`: Path to a GeoTIFF resistance raster.
- `output_dir::String`: Directory managed by the R caller.
- `radius::Int`: Moving-window radius (pixels).
- `block_size::Int`: Block aggregation size.
- `source_from_resistance::Bool`: Derive source strength from resistance.
"""
function run_omniscape(resistance_file::String, output_dir::String,
                       radius::Int, block_size::Int,
                       source_from_resistance::Bool)

    run_dir = joinpath(output_dir, "omniscape_run")
    mkpath(run_dir)

    ini_path = joinpath(output_dir, "omniscape_config.ini")
    open(ini_path, "w") do io
        println(io, "[Required]")
        println(io, "resistance_file = $resistance_file")
        println(io, "radius = $radius")
        println(io, "block_size = $block_size")
        println(io, "project_name = $(joinpath(run_dir, "omniscape"))")
        println(io, "")
        println(io, "[Output options]")
        println(io, "write_raw_currmap = false")
        println(io, "calc_flow_potential = true")
        println(io, "calc_normalized_current = false")
        println(io, "")
        println(io, "[Optional]")
        println(io, "source_from_resistance = $(source_from_resistance ? "true" : "false")")
        println(io, "r_cutoff = Inf")
    end

    @eval import Logging
    Logging.with_logger(Logging.NullLogger()) do
        Omniscape.run_omniscape(ini_path)
    end

    # Omniscape prefixes project_name to each output filename.
    # Rename to the bare names that R expects.
    for (pattern, dest) in [
        ("cum_curr",      "cum_currmap.tif"),
        ("flow_potential", "flow_potential.tif"),
    ]
        src = findfirst(f -> occursin(pattern, f), readdir(run_dir))
        if !isnothing(src) && src != dest
            mv(joinpath(run_dir, src), joinpath(run_dir, dest), force=true)
        end
    end

    return nothing
end


"""
    run_circuitscape(resistance_file, focal_file, output_dir, mode)

Run Circuitscape on a resistance raster with focal nodes, writing the
current map to `output_dir/curmap.tif` (if Circuitscape produces a
GeoTIFF; otherwise the file is left in its native format for the R
caller to handle).

# Arguments
- `resistance_file::String`: Path to resistance GeoTIFF.
- `focal_file::String`: Path to focal nodes file (CSV or raster).
- `output_dir::String`: Directory managed by the R caller.
- `mode::String`: `"pairwise"` or `"one-to-all"`.
"""
function run_circuitscape(resistance_file::String, focal_file::String,
                          output_dir::String, mode::String)

    ini_file = joinpath(output_dir, "cs_config.ini")
    scenario = mode == "one-to-all" ? "one-to-all" : "pairwise"

    open(ini_file, "w") do io
        println(io, "[circuitscape options]")
        println(io, "data_type = raster")
        println(io, "scenario = $scenario")
        println(io, "")
        println(io, "[habitat raster]")
        println(io, "habitat_file = $resistance_file")
        println(io, "habitat_map_is_resistances = true")
        println(io, "")
        println(io, "[point file]")
        println(io, "point_file = $focal_file")
        println(io, "")
        println(io, "[output options]")
        println(io, "write_cur_maps = true")
        println(io, "output_file = $(joinpath(output_dir, "cs_output"))")
    end

    @eval import Logging
    Logging.with_logger(Logging.NullLogger()) do
        Circuitscape.compute(ini_file)
    end

    # Rename the current map to curmap.tif (R reads from there).
    src = findfirst(f -> occursin("curmap", f), readdir(output_dir))
    if !isnothing(src) && src != "curmap.tif"
        mv(joinpath(output_dir, src), joinpath(output_dir, "curmap.tif"), force=true)
    end

    return nothing
end
