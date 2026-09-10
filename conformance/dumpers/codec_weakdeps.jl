# Shared weakdep loader for the Julia conformance dumpers.
#
# EarthSciIO keeps its chunk codecs in weakdep EXTENSIONS so a base install stays
# light (the same culture as the Python reader's lazy `numcodecs` import):
#
#   * `Blosc`     -> `EarthSciIOBloscExt` — the Blosc(zstd)+shuffle container used
#                    by the `:diagnostic` / `:checkpoint` output profiles.
#   * `CodecZstd` -> `EarthSciIOZstdExt`  — the PLAIN Zarr v3 `zstd` codec used by
#                    the `:wasm` output profile. That profile exists because a
#                    WebAssembly/browser Zarr reader cannot decode the Blosc
#                    container (`zarrs`' blosc support comes from `blosc-src`,
#                    whose vendored C sources don't build for
#                    `wasm32-unknown-unknown`), while the standard v3 `zstd` codec
#                    is pure Rust there.
#
# Both are loaded here so ONE write/read driver covers every codec profile.
#
# Neither is importable under a bare `--project=julia` — they are `[weakdeps]`,
# and `Pkg.instantiate()` does not install those. The write drivers therefore run
# in the environment `dumpers/julia_env.jl` prepares, where EarthSciIO and every
# one of its weakdeps are resolved TOGETHER, and a plain import just works.
#
# This deliberately no longer falls back to resolving the package into a
# throwaway environment pushed onto `LOAD_PATH`. That fallback mixes two
# independent resolutions — Julia resolves a stacked package's dependencies
# through the PRIMARY environment's manifest — and it broke the read harness
# outright once the graphs diverged (`julia_env.jl` has the case). It worked here
# only because Blosc's and CodecZstd's dependency graphs happen not to overlap
# EarthSciIO's; that is luck, not a design.

function _load_weakdep!(name::AbstractString)
    try
        @eval import $(Symbol(name))
        return true
    catch err
        @warn """
              cannot import codec weakdep $name; stores using it will fail.
              Run this driver in the prepared conformance environment:
                  julia --project="\$(julia conformance/dumpers/julia_env.jl)" ...
              or just run conformance/run_write_conformance.sh.
              """ active_project = Base.active_project() err
        return false
    end
end

"""
    load_codec_weakdeps!()

Load the `Blosc` and `CodecZstd` weakdeps and activate the corresponding
EarthSciIO extensions, so this process can encode/decode BOTH the Blosc-based
(`:diagnostic`/`:checkpoint`) and the plain-zstd (`:wasm`) codec profiles.
"""
function load_codec_weakdeps!()
    if Base.get_extension(EarthSciIO, :EarthSciIOBloscExt) === nothing
        _load_weakdep!("Blosc")
    end
    if Base.get_extension(EarthSciIO, :EarthSciIOZstdExt) === nothing
        _load_weakdep!("CodecZstd")
    end
    Base.retry_load_extensions()
    return nothing
end
