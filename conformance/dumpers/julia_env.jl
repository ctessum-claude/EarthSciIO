# Builds (and prints the path of) the ONE Julia environment the conformance
# dumpers run in: EarthSciIO plus its weakdeps, resolved TOGETHER.
#
# Why this exists
# ---------------
# The corpus cases decode through weakdep EXTENSIONS — `Blosc` →
# `EarthSciIOBloscExt` (zarr), `Shapefile` → `EarthSciIOShapefileExt`,
# `Parquet2` → `EarthSciIOParquet2Ext`, `TiffImages` → `EarthSciIOTiffImagesExt`,
# `CodecZstd` → `EarthSciIOZstdExt` (the `wasm` write profile). They are
# `[weakdeps]` in `julia/Project.toml` so a base install stays light, which means
# they are NOT importable under `--project=julia`.
#
# The dumpers used to paper over that by resolving each missing package into a
# throwaway environment and pushing it onto `LOAD_PATH`. That is broken, and not
# occasionally: a stacked environment is resolved INDEPENDENTLY, but Julia
# resolves a stacked package's dependencies by walking `LOAD_PATH` in order, so
# the PRIMARY environment answers first for any dependency name it also knows.
# The two manifests silently mix. Concretely, on Julia 1.10:
#
#     --project=julia    JSON 1.x accepts Parsers "2.8.8, 3"  -> Parsers 3.0.0
#     stacked env        WeakRefStrings 1.4.3 needs Parsers "2" -> Parsers 2.8.8
#     stacked lookup     WeakRefStrings' `Parsers` -> the PRIMARY 3.0.0
#     result             `UndefVarError: PosLen not defined` while precompiling
#                        WeakRefStrings -> DBFTables -> Shapefile -> dumper dies
#
# Adding `[compat]` to the throwaway environment cannot fix that: the conflict is
# on `Parsers`, a TRANSITIVE dependency `julia/Project.toml` says nothing about.
# The only fix is to stop stacking — resolve EarthSciIO's own dependencies and
# its weakdeps in ONE environment, so `Parsers` is pinned once to a version that
# satisfies both. That is exactly what `[targets] test` does for `Pkg.test`,
# which is why the unit-test workflow was never hit by this.
#
# The environment is a build product (like `julia/Manifest.toml`): gitignored,
# rebuilt whenever `julia/Project.toml` or the Julia version changes, and
# otherwise reused as-is so a warm run touches neither the resolver nor the
# network. Override its location with `$ESIO_JULIA_ENV`.
#
# Usage:  julia conformance/dumpers/julia_env.jl     # prints the env path

import Pkg
import TOML
import SHA

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PKG_DIR = joinpath(REPO_ROOT, "julia")
const PKG_TOML = joinpath(PKG_DIR, "Project.toml")

# Packages the dumpers `import` themselves, beyond EarthSciIO. They must be
# DIRECT deps of the environment to be importable, and their uuids are taken from
# `julia/Project.toml` so there is exactly one place naming a version of them.
const DRIVER_DEPS = ["JSON"]

"Where the prepared environment lives (a build product, never committed)."
env_path() = get(ENV, "ESIO_JULIA_ENV", joinpath(REPO_ROOT, "conformance", ".julia-env"))

# The environment's direct deps: every weakdep of EarthSciIO (so every extension
# can be activated by one driver) plus what the drivers import directly.
function _direct_deps()
    proj = TOML.parsefile(PKG_TOML)
    deps = Dict{String,Any}()
    for (name, uuid) in get(proj, "weakdeps", Dict{String,Any}())
        deps[name] = uuid
    end
    for name in DRIVER_DEPS
        haskey(proj["deps"], name) || error("$name is not a dep of $PKG_TOML")
        deps[name] = proj["deps"][name]
    end
    return deps
end

# What the resolution depends on. Anything here changing invalidates the manifest.
_stamp(deps) = string(VERSION, "\n",
                      bytes2hex(SHA.sha256(read(PKG_TOML))), "\n",
                      join(sort!(collect(keys(deps))), ","), "\n")

# Resolve/install OFFLINE first — the harness's whole claim is that it touches no
# network — and only reach out when the depot genuinely lacks something.
function _offline_first(f)
    was = get(ENV, "JULIA_PKG_OFFLINE", nothing)
    try
        ENV["JULIA_PKG_OFFLINE"] = "true"
        f()
    catch
        delete!(ENV, "JULIA_PKG_OFFLINE")
        f()
    finally
        was === nothing ? delete!(ENV, "JULIA_PKG_OFFLINE") : (ENV["JULIA_PKG_OFFLINE"] = was)
    end
end

"""
    prepare_julia_env() -> String

Materialise the conformance environment and return its path. Idempotent: a warm
environment whose stamp still matches is only instantiated (a no-op that needs no
network), never re-resolved.
"""
function prepare_julia_env(; io = stderr)
    env = env_path()
    deps = _direct_deps()
    stamp_file = joinpath(env, ".esio-env-stamp")
    warm = isfile(joinpath(env, "Manifest.toml")) &&
           isfile(stamp_file) && read(stamp_file, String) == _stamp(deps)
    if !warm
        mkpath(env)
        # A stale manifest must not survive a Project.toml change.
        rm(joinpath(env, "Manifest.toml"); force = true)
        open(f -> TOML.print(f, Dict("deps" => deps)), joinpath(env, "Project.toml"), "w")
    end
    Pkg.activate(env; io = io)
    if !warm
        # `develop` (not `add`) — EarthSciIO is read from THIS working tree, so
        # the dumpers exercise the checkout under test and the resolver sees its
        # `[compat]`. It resolves the deps written above in the same pass.
        _offline_first(() -> Pkg.develop(Pkg.PackageSpec(path = PKG_DIR); io = io))
    end
    _offline_first(() -> Pkg.instantiate(; io = io))
    warm || write(stamp_file, _stamp(deps))
    return env
end

if abspath(PROGRAM_FILE) == @__FILE__
    println(prepare_julia_env())
end
