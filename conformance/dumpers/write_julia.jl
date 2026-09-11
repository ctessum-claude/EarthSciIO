# Julia track's WRITE driver for the cross-language write-conformance harness.
#
# Drives the **Julia Zarr v3 sharded writer** (`EarthSciIO.ZarrWriter`, the
# reference implementation in `julia/src/zarr_write.jl`) from the shared,
# language-neutral input spec (`conformance/write_spec.json`) and emits a Zarr v3
# sharded store into an output directory. The write mirror of `dump_julia.jl`
# (streaming-output-sinks RFC, Wave 5).
#
# The store this produces is read back by every available track's reader
# (`read_python.py` / `read_julia.jl`) and cross-checked, and its `zarr.json`
# metadata is structurally compared against the Python- and Rust-written stores.
# Conformance is TOLERANCE-BASED on decoded arrays (RFC §16.6), never byte identity.
#
# Usage:  julia --project="$(julia conformance/dumpers/julia_env.jl)" \
#             conformance/dumpers/write_julia.jl OUT_DIR [SPEC.json]
#         (`conformance/run_write_conformance.sh` does both steps.)

using EarthSciIO
import JSON

# The Julia Zarr writer ENcodes chunks through a weakdep extension: `Blosc` /
# `EarthSciIOBloscExt` for the `:diagnostic`/`:checkpoint` profiles, `CodecZstd` /
# `EarthSciIOZstdExt` for the plain-zstd `:wasm` profile. Load both so this one
# driver can write every codec profile variant.
include(joinpath(@__DIR__, "codec_weakdeps.jl"))
load_codec_weakdeps!()

const _DTYPE = Dict("float64" => Float64, "float32" => Float32,
                    "int32" => Int32, "int64" => Int64)

# Build the Julia `OutputSchema` from the language-neutral spec dict.
function build_schema(spec)
    dims = [String(d[1]) => Int(d[2]) for d in spec["dims"]]
    coords = Pair{String,Tuple{Vector,Dict{String,Any}}}[]
    for co in spec["coords"]
        vals = Float64[Float64(v) for v in co["values"]]
        attrs = Dict{String,Any}(String(k) => v for (k, v) in co["attrs"])
        push!(coords, String(co["name"]) => (vals, attrs))
    end
    vars = Pair{String,OutputVar}[]
    for v in spec["vars"]
        vdims = String[String(d) for d in v["dims"]]
        attrs = Dict{String,Any}(String(k) => x for (k, x) in v["attrs"])
        push!(vars, String(v["name"]) => OutputVar(vdims, _DTYPE[String(v["dtype"])]; attrs = attrs))
    end
    chunk = Dict{String,Int}(String(k) => Int(v) for (k, v) in spec["chunk_shape"])
    shard = Dict{String,Int}(String(k) => Int(v) for (k, v) in spec["shard_shape"])
    gattrs = Dict{String,Any}(String(k) => v for (k, v) in spec["group_attrs"])
    return OutputSchema(; dims = dims, time_dim = String(spec["time_dim"]),
                        vars = vars, chunk_shape = chunk, shard_shape = shard,
                        coords = coords, profile = Symbol(spec["profile"]),
                        attrs = gattrs, time_dtype = _DTYPE[String(get(spec, "time_dtype", "float64"))])
end

# One record's spatial slab for a var: the JSON block is [lat][lon]; the writer
# wants an Array over the var's non-time dims in order (lat, lon here).
function slab(block)
    nlat = length(block)
    nlon = length(block[1])
    return Float64[Float64(block[i][j]) for i in 1:nlat, j in 1:nlon]
end

function main()
    length(ARGS) >= 1 || error("usage: write_julia.jl OUT_DIR [SPEC.json]")
    out_dir = abspath(ARGS[1])
    conf = normpath(joinpath(@__DIR__, ".."))
    spec_path = length(ARGS) >= 2 ? ARGS[2] : joinpath(conf, "write_spec.json")
    spec = JSON.parsefile(spec_path)

    schema = build_schema(spec)
    w = ZarrWriter()
    h = write_open!(w, nothing, out_dir, schema)
    for rec in spec["records"]
        t = Float64(rec["t"])
        arrays = Dict{String,Any}(String(nm) => slab(block) for (nm, block) in rec["vars"])
        write_record!(w, h, t, arrays)
    end
    m = write_close!(w, h)
    codec_id = get(m.codec, "id", "?")
    println("[julia-writer] wrote $(m.n_records) records to $out_dir " *
            "(profile=$(spec["profile"]), $(length(spec["vars"])) vars, " *
            "codec=$(codec_id))")
end

main()
