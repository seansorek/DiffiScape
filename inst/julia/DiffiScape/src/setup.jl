"""
Setup utilities for the DiffiScape Julia module.

Called once during R package initialisation via `ds_julia_setup()`.
"""

export check_dependencies, get_version_info

"""
    check_dependencies() -> Dict{String,Bool}

Verify that all required Julia packages are available.
"""
function check_dependencies()::Dict{String,Bool}
    deps = Dict{String,Bool}()

    for pkg in ["Omniscape", "Circuitscape"]
        deps[pkg] = try
            @eval import $(Symbol(pkg))
            true
        catch
            false
        end
    end

    # Enzyme is optional
    deps["Enzyme"] = enzyme_available()

    return deps
end


"""
    get_version_info() -> Dict{String,String}

Return version strings for Julia and key packages.
"""
function get_version_info()::Dict{String,String}
    info = Dict{String,String}()
    info["julia"] = string(VERSION)

    for pkg in ["Omniscape", "Circuitscape", "Enzyme"]
        info[pkg] = try
            @eval import Pkg
            deps = Pkg.dependencies()
            for (_, dep) in deps
                if dep.name == pkg
                    string(dep.version)
                end
            end
            "unknown"
        catch
            "not installed"
        end
    end

    return info
end
