"""
Connectivity computation via Omniscape.jl and Circuitscape.jl.
"""

using Omniscape
using Circuitscape

export run_omniscape, run_circuitscape

"""
    run_omniscape(resistance_file; radius=13, block_size=5, source_threshold=0.0)

Run Omniscape on a resistance raster file and return paths to the output
rasters (cumulative current and flow potential).

# Arguments
- `resistance_file::String`: Path to a GeoTIFF resistance raster.
- `radius::Int`: Moving-window radius (pixels).
- `block_size::Int`: Block aggregation size.
- `source_threshold::Float64`: Minimum source strength.

# Returns
A `Dict{String,String}` with keys `"cum_current"` and `"flow_potential"`
pointing to output GeoTIFF paths.
"""
function run_omniscape(resistance_file::String;
                       radius::Int=13,
                       block_size::Int=5,
                       source_threshold::Float64=0.0)

    output_dir = mktempdir()

    # Build Omniscape configuration
    config = Dict{String,String}(
        "resistance_file"         => resistance_file,
        "radius"                  => string(radius),
        "block_size"              => string(block_size),
        "source_threshold"        => string(source_threshold),
        "project_name"            => joinpath(output_dir, "omniscape"),
        "source_from_resistance"  => "true",
        "r_cutoff"                => "Inf",
        "calc_normalized_current" => "false",
        "calc_flow_potential"     => "true",
        "write_raw_currmap"       => "false",
    )

    # Run
    Omniscape.run_omniscape(config)

    # Locate outputs
    cum_file  = joinpath(output_dir, "omniscape_cum_currmap.tif")
    flow_file = joinpath(output_dir, "omniscape_flow_potential.tif")

    if !isfile(cum_file)
        # Try alternative naming
        possible = filter(f -> occursin("cum_curr", f), readdir(output_dir))
        if !isempty(possible)
            cum_file = joinpath(output_dir, first(possible))
        end
    end

    if !isfile(flow_file)
        possible = filter(f -> occursin("flow_potential", f), readdir(output_dir))
        if !isempty(possible)
            flow_file = joinpath(output_dir, first(possible))
        end
    end

    return Dict{String,String}(
        "cum_current"    => cum_file,
        "flow_potential" => flow_file,
        "output_dir"     => output_dir
    )
end


"""
    run_circuitscape(resistance_file, focal_file; mode="pairwise")

Run Circuitscape on a resistance raster with focal nodes.

# Arguments
- `resistance_file::String`: Path to resistance GeoTIFF.
- `focal_file::String`: Path to focal nodes file (CSV or raster).
- `mode::String`: `"pairwise"` or `"one-to-all"`.

# Returns
A `Dict{String,String}` with key `"current_map"` pointing to the output.
"""
function run_circuitscape(resistance_file::String,
                          focal_file::String;
                          mode::String="pairwise")

    output_dir = mktempdir()
    ini_file   = joinpath(output_dir, "cs_config.ini")

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

    Circuitscape.compute(ini_file)

    current_file = joinpath(output_dir, "cs_output_curmap.asc")
    if !isfile(current_file)
        possible = filter(f -> occursin("curmap", f), readdir(output_dir))
        if !isempty(possible)
            current_file = joinpath(output_dir, first(possible))
        end
    end

    return Dict{String,String}(
        "current_map" => current_file,
        "output_dir"  => output_dir
    )
end
