# The cadence Provider (esio-9nb.5) — the acceptance: "Provider over the shared
# fixture returns native arrays matching the Python track; CONST/DISCRETE
# correct; refresh_times() matches the cadence." Drives the FULL component
# (a)+(b) pipeline OFFLINE: cache (shared corpus) → format reader → native
# arrays, plus the cadence surface the solver consumes
# (materialize/refresh/refresh_times/prefetch). Reuses the corpus comparison
# helpers from test_readers.jl (included first by runtests.jl).

# A reader that wraps the real netcdf one and RECORDS what the Provider pushed
# down. It is how the record-pushdown testset below asserts the other half of the
# contract: not just that the bracket is unchanged, but that the Provider asked
# for exactly the records it needed — file-local and 0-based — in each of the four
# bracket cases. (Top level, because a struct and its methods cannot be defined
# inside a `@testset begin` block.)
struct RecordSpyReader <: EarthSciIO.Reader end
const RECORDS_PUSHED = Vector{Any}()
function EarthSciIO.read_native(::RecordSpyReader, path::AbstractString;
                                variables = nothing, select = nothing, records = nothing)
    # Record the RESOLVED list. The Provider pushes `indices` as a
    # `len -> indices` callable (so the axis length costs no second open), so
    # resolve it here against the same length the real reader will use — that is
    # what makes the assertions below read as the record numbers they are.
    if records !== nothing
        d = String(records["dim"])
        want = records["indices"]
        len = dim_length(FORMAT_REGISTRY["netcdf"], path, d)
        push!(RECORDS_PUSHED,
              Dict{String,Any}("dim" => d,
                               "indices" => want isa Base.Callable ?
                                            Int[Int(i) for i in want(len)] :
                                            Int[Int(i) for i in want]))
    else
        push!(RECORDS_PUSHED, records)
    end
    return read_native(FORMAT_REGISTRY["netcdf"], path; variables = variables,
                       select = select, records = records)
end
EarthSciIO.dim_length(::RecordSpyReader, path::AbstractString, dim::AbstractString) =
    dim_length(FORMAT_REGISTRY["netcdf"], path, dim)
register!(FORMAT_REGISTRY, "netcdf-recordspy", RecordSpyReader())

# Field-for-field BYTE equality of two decoded datasets — names, dims, attrs,
# eltypes and the bit patterns themselves. `≈` would hide a NaN that moved cell,
# and a Float32 that reached Float64 by a different route.
function same_native(a, b)
    variable_names(a) == variable_names(b) || return false
    coord_names(a) == coord_names(b) || return false
    for n in vcat(variable_names(a), coord_names(a))
        a[n].dims == b[n].dims || return false
        a[n].attrs == b[n].attrs || return false
        size(a[n].data) == size(b[n].data) || return false
        eltype(a[n].data) == eltype(b[n].data) || return false
        reinterpret(UInt8, vec(collect(a[n].data))) ==
            reinterpret(UInt8, vec(collect(b[n].data))) || return false
    end
    return true
end

@testset "cadence Provider — materialize/refresh/refresh_times/prefetch (offline)" begin
    store = LocalStore(joinpath(CORPUS, "cache"))
    cache = Cache(store; offline = true, verify = true)
    era5 = "https://data.earthsci.dev/era5/2018/11/20181108.nc"
    openaq = "https://openaq-data-archive.s3.amazonaws.com/records/openaq/locationid=1/2018-11-08.csv"
    era5_case = JSON.parsefile(joinpath(CORPUS, "cases", "era5-grid-sub-tile.json"))

    @testset "CONST grid: empty cadence, native arrays match the oracle" begin
        p = const_provider(cache, era5; format = "netcdf", source_loader = "era5")
        @test is_const(p)
        @test refresh_times(p) == Float64[]          # CONST ⇒ never refreshes

        nds = materialize(p)
        @test Set(variable_names(nds)) == Set(["t2m", "sp"])
        @test Set(coord_names(nds)) == Set(["latitude", "longitude", "time"])

        # full native-array equality vs the Python track (checks 3–4 via Provider)
        for group in ("variables", "coords")
            for (name, spec) in era5_case["expected"][group]
                @test cmp_native_numeric(nds[name].data, spec["data"]) === nothing
            end
        end
        # the raw time axis is undecoded with its calendar carried for ESS
        @test eltype(nds["time"].data) == Int32
        @test nds["time"].attrs["calendar"] == "gregorian"
    end

    @testset "DISCRETE grid: refresh_times match cadence, per-tick slice" begin
        # cadence taken from the file's own (raw) time axis: [0.0, 1.0]
        full = materialize(const_provider(cache, era5; format = "netcdf"))
        times = Float64.(full["time"].data)
        p = discrete_provider(cache, era5, times; format = "netcdf", time_dim = "time")

        @test !is_const(p)
        @test refresh_times(p) == times              # matches the cadence

        s0 = refresh(p, 0.0)
        s1 = refresh(p, 1.0)
        @test s0["t2m"].dims == ["latitude", "longitude"]   # time record sliced out
        @test size(s0["t2m"].data) == (3, 3)
        @test s0["t2m"].data[1, 1] ≈ 282.5
        @test s1["t2m"].data[1, 1] ≈ 282.6           # a different record per tick
        @test isnan(s1["t2m"].data[3, 3])            # the masked cell survives the slice
        @test !haskey(s0, "time")                    # the sliced dim's coord is dropped
        @test refresh(p, 0.0)["sp"].data == s0["sp"].data   # refresh == materialize

        # a tick between grid points resolves to the active (last ≤ t) record
        @test materialize(p, 0.5)["t2m"].data == s0["t2m"].data
    end

    @testset "DISCRETE records_per_sample=2: the 2-record bracket" begin
        # cadence taken from the file's own (raw) time axis: [0.0, 1.0]. The axis is
        # "hours since 2018-11-08 00:00:00", so raw hours 0/1/2 decode to these Unix
        # epoch seconds (== datetime2unix(DateTime(2018,11,8,h))):
        epoch_h0 = 1.5416352e9   # 2018-11-08 00:00:00Z
        epoch_h1 = 1.5416388e9   # 2018-11-08 01:00:00Z
        epoch_h2 = epoch_h1 + 3600.0   # 2018-11-08 02:00:00Z

        full = materialize(const_provider(cache, era5; format = "netcdf"))
        times = Float64.(full["time"].data)                          # [0.0, 1.0]
        p = discrete_provider(cache, era5, times; format = "netcdf",
                              time_dim = "time", records_per_sample = 2)

        @testset "two records, time dim retained, epoch-seconds coord" begin
            b = refresh(p, 0.0)
            # time axis RETAINED at length 2 (floor + successor), not sliced out
            @test b["t2m"].dims == ["time", "latitude", "longitude"]
            @test size(b["t2m"].data) == (2, 3, 3)
            @test b["t2m"].data[1, 1, 1] ≈ 282.5      # record 0 (hour 0)
            @test b["t2m"].data[2, 1, 1] ≈ 282.6      # record 1 (hour 1)
            @test isnan(b["t2m"].data[2, 3, 3])       # masked cell survives in record 1
            # the time coord carries the two bracket timestamps as epoch seconds
            @test haskey(b, "time")
            @test length(b["time"].data) == 2
            @test b["time"].data ≈ [epoch_h0, epoch_h1]
            @test eltype(b["time"].data) == Float64
            @test b["time"].attrs["units"] == "seconds since 1970-01-01T00:00:00Z"
            @test Set(coord_names(b)) == Set(["latitude", "longitude", "time"])
        end

        @testset "flooring within an interval keeps the same bracket" begin
            at = refresh(p, 0.0)["t2m"].data
            between = materialize(p, 0.5)                 # snaps down to hour 0
            @test size(between["t2m"].data) == (2, 3, 3)
            @test isequal(at, between["t2m"].data)        # isequal: NaN == NaN
            # ... the bracket timestamps still describe hour 0 -> hour 1
            @test between["time"].data ≈ [epoch_h0, epoch_h1]
        end

        @testset "cross-file successor (url-function, cadence past the file axis)" begin
            # 3-tick cadence over a 2-record file: hour 1 is the file's last record,
            # so its successor is record 1 of the NEXT file (same corpus blob here).
            pc = discrete_provider(cache, _ -> era5, [0.0, 1.0, 2.0];
                                   format = "netcdf", time_dim = "time",
                                   records_per_sample = 2)
            b = refresh(pc, 1.0)
            @test size(b["t2m"].data) == (2, 3, 3)
            @test b["t2m"].data[1, 1, 1] ≈ 282.6      # this file, record 1 (hour 1)
            @test b["t2m"].data[2, 1, 1] ≈ 282.5      # next file, record 0 (hour 0)
            @test b["time"].data ≈ [epoch_h1, epoch_h2]
        end

        @testset "end clamp: last tick degenerates to [last, last]" begin
            b = refresh(p, 1.0)                          # hour 1 is the last tick
            @test size(b["t2m"].data) == (2, 3, 3)
            @test b["t2m"].data[1, 1, 1] ≈ 282.6
            @test b["t2m"].data[2, 1, 1] ≈ 282.6         # successor == floor (held)
            @test b["time"].data[1] == b["time"].data[2] # degenerate → equal stamps
            @test b["time"].data ≈ [epoch_h1, epoch_h1]

            # a time PAST the last tick clamps to the same degenerate bracket (no throw)
            past = materialize(p, 5.0)
            @test past["t2m"].data[1, 1, 1] ≈ 282.6
            @test past["time"].data[1] == past["time"].data[2]
        end

        @testset "records_per_sample guards" begin
            # only nothing, 1, or 2 is accepted
            @test_throws ArgumentError discrete_provider(cache, era5, [0.0];
                format = "netcdf", time_dim = "time", records_per_sample = 3)
            # records_per_sample=2 needs a time_dim to bracket along
            @test_throws ArgumentError discrete_provider(cache, era5, [0.0];
                format = "netcdf", records_per_sample = 2)
        end
    end

    @testset "record pushdown: the Provider narrows the decode, not the result" begin
        # The Provider owns the cadence, so it — not the reader — decides which
        # records it wants; what it pushes down is an absolute, file-local, 0-based
        # record list. Asserted in both directions: the right list reaches the
        # reader in each of the four bracket cases, AND the dataset that comes back
        # is byte-for-byte what decoding the whole record axis and slicing
        # afterwards produced.
        #
        # Two "days" of the 2-record era5 fixture under two URLs, so the cadence
        # genuinely crosses a file seam (record values repeat per file:
        # record 1 = 282.5, record 2 = 282.6).
        day1, day2 = era5, "https://data.earthsci.dev/era5/2018/11/20181110.nc"
        root = mktempdir()
        src = joinpath(CORPUS, "cache", "v1", "blobs", cache_key(era5)[1:2],
                       cache_key(era5) * ".nc")
        for u in (day1, day2)
            k = cache_key(u)
            d = joinpath(root, "v1", "blobs", k[1:2]); mkpath(d)
            cp(src, joinpath(d, k * ".nc"))
        end
        # verify=false: the copies carry no manifest to check the digest against.
        two = Cache(LocalStore(root); offline = true, verify = false)
        urls = t -> t < 2.0 ? day1 : day2
        times = [0.0, 1.0, 2.0, 3.0]          # 4 ticks over 2 files of 2 records

        # The reference side bakes `records = nothing`, which is exactly the
        # historical path: the Provider leaves the pushdown alone when the caller
        # has spelled a `records` of its own, and the reader then reads the record
        # axis whole and the Provider slices afterwards.
        mkspy(rps) = discrete_provider(two, urls, times; format = "netcdf-recordspy",
                                       time_dim = "time", records_per_sample = rps)
        mkref(rps) = discrete_provider(two, urls, times; format = "netcdf",
                                       time_dim = "time", records_per_sample = rps,
                                       reader_kwargs = (records = nothing,))
        rec(idxs) = Dict{String,Any}("dim" => "time", "indices" => idxs)
        spy2, ref2 = mkspy(2), mkref(2)
        spy1, ref1 = mkspy(1), mkref(1)

        @testset "interior bracket: both records, ONE decode of one file" begin
            empty!(RECORDS_PUSHED)
            b = refresh(spy2, 0.0)                  # file 1, records 1+2
            @test RECORDS_PUSHED == [rec([0, 1])]   # one decode covers the pair
            @test same_native(b, refresh(ref2, 0.0))
            @test b["t2m"].data[1, 1, 1] ≈ 282.5
            @test b["t2m"].data[2, 1, 1] ≈ 282.6
        end

        @testset "cross-file successor: one record out of each file" begin
            empty!(RECORDS_PUSHED)
            b = refresh(spy2, 1.0)                  # file 1 rec 2 -> file 2 rec 1
            @test RECORDS_PUSHED == [rec([1]), rec([0])]
            @test same_native(b, refresh(ref2, 1.0))
            @test b["t2m"].data[1, 1, 1] ≈ 282.6
            @test b["t2m"].data[2, 1, 1] ≈ 282.5
        end

        @testset "end of data: the degenerate [last, last] never throws" begin
            empty!(RECORDS_PUSHED)
            b = refresh(spy2, 3.0)                  # last tick: there is no successor
            # ONE record is read and stacked with itself; nothing may be asked of a
            # file past the end of the cadence.
            @test RECORDS_PUSHED == [rec([1])]
            @test same_native(b, refresh(ref2, 3.0))
            @test b["t2m"].data[1, 1, 1] == b["t2m"].data[2, 1, 1]
            @test b["time"].data[1] == b["time"].data[2]
            empty!(RECORDS_PUSHED)
            past = materialize(spy2, 9.0)           # past the end clamps the same way
            @test RECORDS_PUSHED == [rec([1])]
            @test same_native(past, materialize(ref2, 9.0))
        end

        @testset "records_per_sample=1: one record, time_dim dropped" begin
            empty!(RECORDS_PUSHED)
            s = refresh(spy1, 2.0)                  # file 2, record 1
            @test RECORDS_PUSHED == [rec([0])]
            @test !haskey(s, "time")                # the record axis is dropped
            @test s["t2m"].dims == ["latitude", "longitude"]
            @test same_native(s, refresh(ref1, 2.0))
            @test s["t2m"].data[1, 1] ≈ 282.5
            empty!(RECORDS_PUSHED)
            @test same_native(refresh(spy1, 3.0), refresh(ref1, 3.0))
            @test RECORDS_PUSHED == [rec([1])]
        end

        @testset "the pushdown is opt-in, per reader" begin
            # The `csv` reader declares no `records` option, so nothing is pushed
            # and the Provider must not try — this is what keeps the change
            # additive for every other format.
            @test !(:records in reader_option_keys(FORMAT_REGISTRY["csv"]))
            @test :records in reader_option_keys(FORMAT_REGISTRY["netcdf"])
            @test EarthSciIO._records_pushdown(discrete_provider(two, urls, times;
                format = "netcdf", time_dim = "time", records_per_sample = 2))
            @test !EarthSciIO._records_pushdown(mkref(2))          # a baked one wins
            @test !EarthSciIO._records_pushdown(discrete_provider(two, urls, times;
                format = "netcdf", records_per_sample = nothing))  # no time_dim
        end

        @testset "a baked `records` beside a time_dim is refused, not honoured" begin
            # `records = nothing` opts out of the pushdown, and that is the only
            # value a provider that OWNS a record axis may carry. Any other value
            # narrows the axis BEHIND the Provider's tick->record resolution, so
            # the tick is located inside the already-narrowed axis: a one-record
            # `records` makes every tick `mod1(tick, 1) == 1`, so `refresh` hands
            # back the same record at every tick and a `records_per_sample=2`
            # bracket degenerates to [r, r] — a constant field where the model
            # expected to interpolate. Silently wrong numbers, refused at
            # construction (spec/registries.md §2.1).
            e = try
                discrete_provider(two, urls, times; format = "netcdf",
                                  time_dim = "time", records_per_sample = 2,
                                  reader_kwargs = (records = Dict("dim" => "time",
                                                                  "indices" => [0]),))
                nothing
            catch err
                err
            end
            @test e isa ArgumentError
            @test occursin("`records` may not be baked", sprint(showerror, e))
            @test occursin("time_dim", sprint(showerror, e))
            # ...while `nothing` (the opt-out) and a provider with no record axis
            # of its own are both fine.
            @test mkref(2) isa Provider
            @test discrete_provider(two, urls, times; format = "netcdf",
                reader_kwargs = (records = Dict("dim" => "time",
                                               "indices" => [0]),)) isa Provider
        end
    end

    @testset "record pushdown == whole-file read, over a real multi-record cadence" begin
        # THE equivalence gate for the pushdown. The fixture above holds only TWO
        # records, where the interior bracket `[0, 1]` is the whole axis and a
        # narrowed decode cannot be told from a full one. These files hold three
        # and four, so every bracket below is a genuine subset — and every sample
        # the pushdown produces is compared, field for field and byte for byte,
        # against the SAME provider forced onto the historical whole-file path.
        #
        # Written here rather than taken from the corpus because the corpus is the
        # cross-language gate and this is a Julia-track decode optimisation; the
        # numbers it must not change are the ones the reference path produces.
        NC = EarthSciIO.NCDatasets
        function write_cadence_nc(path; nrec::Int, base::Float64, t0::Float64)
            NC.NCDataset(path, "c") do ds
                NC.defDim(ds, "time", nrec)
                NC.defDim(ds, "latitude", 2)
                NC.defDim(ds, "longitude", 3)
                tv = NC.defVar(ds, "time", Float64, ("time",))
                tv.attrib["units"] = "hours since 2018-11-09 00:00:00"
                tv.attrib["calendar"] = "standard"
                tv[:] = collect(t0:(t0 + nrec - 1))
                lat = NC.defVar(ds, "latitude", Float64, ("latitude",))
                lat[:] = [40.0, 41.0]
                lon = NC.defVar(ds, "longitude", Float64, ("longitude",))
                lon[:] = [-90.0, -89.0, -88.0]
                # `t2m` carries the record axis: record r is the constant base+r,
                # so a mis-bracketed record shows up as a wrong NUMBER, not a
                # wrong shape. `orog` carries no record axis and must pass through
                # both paths untouched.
                v = NC.defVar(ds, "t2m", Float64, ("longitude", "latitude", "time"))
                for r in 1:nrec
                    v[:, :, r] .= base + r
                end
                o = NC.defVar(ds, "orog", Float64, ("longitude", "latitude"))
                o[:, :] .= 7.0
            end
            return path
        end
        function stage(specs)                     # url => (nrec, base, t0)
            root = mktempdir()
            for (u, (nrec, base, t0)) in specs
                k = cache_key(u)
                d = joinpath(root, "v1", "blobs", k[1:2]); mkpath(d)
                write_cadence_nc(joinpath(d, k * ".nc"); nrec = nrec, base = base, t0 = t0)
            end
            return Cache(LocalStore(root); offline = true, verify = false)
        end
        push_ref(c, urls, ts; kw...) = (
            discrete_provider(c, urls, ts; format = "netcdf", kw...),
            discrete_provider(c, urls, ts; format = "netcdf",
                              reader_kwargs = (records = nothing,), kw...))
        # Every sample of every cadence, at each `records_per_sample`, must be
        # byte-identical between the two paths.
        function assert_equivalent(c, urls, ticks, probes; kw...)
            for rps in (nothing, 1, 2)
                p, r = push_ref(c, urls, ticks; records_per_sample = rps, kw...)
                @test EarthSciIO._records_pushdown(p)      # the pushdown is really on
                @test !EarthSciIO._records_pushdown(r)     # ...and really off here
                for t in probes
                    @test same_native(materialize(p, t), materialize(r, t))
                end
            end
        end

        u4a = "https://data.earthsci.dev/rec/4a.nc"
        u4b = "https://data.earthsci.dev/rec/4b.nc"
        u3a = "https://data.earthsci.dev/rec/3a.nc"
        u3b = "https://data.earthsci.dev/rec/3b.nc"

        @testset "two 4-record files, 8-tick cadence" begin
            c = stage([u4a => (4, 100.0, 0.0), u4b => (4, 200.0, 4.0)])
            urls = t -> t < 4.0 ? u4a : u4b
            ticks = collect(0.0:7.0)
            # ticks, half-steps between them, and past the end of the cadence
            assert_equivalent(c, urls, ticks,
                              vcat(ticks, [0.5, 3.5, 4.5, 6.5, 50.0]); time_dim = "time")
            # ...and the numbers themselves, so the comparison cannot pass with
            # BOTH paths mis-bracketed. Records are base+r; file 2 is base 200.
            p, _ = push_ref(c, urls, ticks; time_dim = "time", records_per_sample = 2)
            @test materialize(p, 1.0)["t2m"].data[:, 1, 1] == [102.0, 103.0]  # interior
            @test materialize(p, 3.0)["t2m"].data[:, 1, 1] == [104.0, 201.0]  # file seam
            @test materialize(p, 3.5)["t2m"].data[:, 1, 1] == [104.0, 201.0]  # off-tick
            @test materialize(p, 7.0)["t2m"].data[:, 1, 1] == [204.0, 204.0]  # degenerate
            @test materialize(p, 50.0)["t2m"].data[:, 1, 1] == [204.0, 204.0] # past the end
            @test materialize(p, 7.0)["orog"].dims == ["latitude", "longitude"]
            # The interior bracket is a genuine SUBSET here: records 1,2 of four.
            empty!(RECORDS_PUSHED)
            spy = discrete_provider(c, urls, ticks; format = "netcdf-recordspy",
                                    time_dim = "time", records_per_sample = 2)
            @test materialize(spy, 1.0)["t2m"].data[:, 1, 1] == [102.0, 103.0]
            @test RECORDS_PUSHED == [Dict{String,Any}("dim" => "time",
                                                      "indices" => [1, 2])]
        end

        @testset "files of UNEQUAL length (3 then 4 records)" begin
            # The Provider's mod1 record arithmetic assumes uniform files, so this
            # cadence is mis-tracked by DESIGN past the seam (documented on
            # `Provider`). The pushdown must reproduce that behaviour exactly —
            # an optimisation may not change a number, right or wrong.
            c = stage([u3a => (3, 100.0, 0.0), u3b => (4, 200.0, 3.0)])
            urls = t -> t < 3.0 ? u3a : u3b
            ticks = collect(0.0:6.0)
            assert_equivalent(c, urls, ticks, vcat(ticks, [2.5, 3.5, 99.0]); time_dim = "time")
        end

        @testset "single-record files" begin
            c = stage([u4a => (1, 100.0, 0.0), u4b => (1, 200.0, 1.0)])
            urls = t -> t < 1.0 ? u4a : u4b
            assert_equivalent(c, urls, [0.0, 1.0], [0.0, 0.5, 1.0, 9.0]; time_dim = "time")
            p, _ = push_ref(c, urls, [0.0, 1.0]; time_dim = "time", records_per_sample = 2)
            @test materialize(p, 0.0)["t2m"].data[:, 1, 1] == [101.0, 201.0]  # across files
            @test materialize(p, 1.0)["t2m"].data[:, 1, 1] == [201.0, 201.0]  # degenerate
        end

        @testset "cadence longer than the file: the mod1 WRAP, one url" begin
            # One file of three records under a constant url and a 7-tick cadence:
            # tick 3 -> record 3, tick 4 -> record 1, so the bracket at tick 3 is
            # the DECREASING pair [2, 0]. That is the one shape the reader cannot
            # read as a strided slab, and it must still come back in the order
            # asked for rather than sorted.
            c = stage([u3a => (3, 100.0, 0.0)])
            ticks = collect(0.0:6.0)
            assert_equivalent(c, u3a, ticks, vcat(ticks, [5.5]); time_dim = "time")
            p, _ = push_ref(c, u3a, ticks; time_dim = "time", records_per_sample = 2)
            @test materialize(p, 1.0)["t2m"].data[:, 1, 1] == [102.0, 103.0]
            @test materialize(p, 2.0)["t2m"].data[:, 1, 1] == [103.0, 101.0]  # the wrap
            @test materialize(p, 6.0)["t2m"].data[:, 1, 1] == [101.0, 101.0]
        end

        @testset "a spatial `select` rides along unchanged" begin
            # `select` narrows space, `records` narrows the record axis, and the
            # two must compose without either standing in for the other.
            c = stage([u4a => (4, 100.0, 0.0), u4b => (4, 200.0, 4.0)])
            urls = t -> t < 4.0 ? u4a : u4b
            ticks = collect(0.0:7.0)
            sel = Dict("axes" => Any["all", "all", Dict("indices" => [2, 0])])
            for rps in (1, 2)
                p, r = push_ref(c, urls, ticks; time_dim = "time", records_per_sample = rps)
                for t in [0.0, 3.0, 4.0, 7.0]
                    @test same_native(materialize(p, t; select = sel),
                                      materialize(r, t; select = sel))
                end
                w = materialize(p, 3.0; select = sel)
                @test w["longitude"].data == [-88.0, -90.0]     # order given, not sorted
                @test size(w["t2m"].data, ndims(w["t2m"].data)) == 2
            end
        end
    end

    @testset "multi-file cadence: the record is located inside its OWN file" begin
        # A cadence that spans several files is the ordinary case for a long run
        # (GEOS-FP publishes one file per day; a week is seven of them). The
        # record index must restart at 1 on every file seam, because the tick
        # index counts the whole cadence while each file only holds its own
        # slice of it: tick 3 of a 4-tick cadence over two 2-record files is
        # record 1 of file 2, NOT record 3 of file 1 -- which does not exist.
        #
        # Two "days" published under two URLs, both served by the one era5
        # fixture blob, so the resolver genuinely switches files mid-cadence
        # (values repeat per file: record 1 = 282.5, record 2 = 282.6).
        day1, day2 = era5, "https://data.earthsci.dev/era5/2018/11/20181109.nc"
        root = mktempdir()
        src = joinpath(CORPUS, "cache", "v1", "blobs", cache_key(era5)[1:2],
                       cache_key(era5) * ".nc")
        for u in (day1, day2)
            k = cache_key(u)
            d = joinpath(root, "v1", "blobs", k[1:2]); mkpath(d)
            cp(src, joinpath(d, k * ".nc"))
        end
        # verify=false: the copies carry no manifest to check the digest against.
        two = Cache(LocalStore(root); offline = true, verify = false)
        urls = t -> t < 2.0 ? day1 : day2
        times = [0.0, 1.0, 2.0, 3.0]          # 4 ticks over 2 files of 2 records

        @testset "sliced (records_per_sample=1)" begin
            p = discrete_provider(two, urls, times; format = "netcdf", time_dim = "time")
            @test refresh(p, 0.0)["t2m"].data[1, 1] ≈ 282.5   # file 1, record 1
            @test refresh(p, 1.0)["t2m"].data[1, 1] ≈ 282.6   # file 1, record 2
            @test refresh(p, 2.0)["t2m"].data[1, 1] ≈ 282.5   # file 2, record 1
            @test refresh(p, 3.0)["t2m"].data[1, 1] ≈ 282.6   # file 2, record 2
        end

        @testset "bracketed (records_per_sample=2)" begin
            p = discrete_provider(two, urls, times; format = "netcdf",
                                  time_dim = "time", records_per_sample = 2)
            seam = refresh(p, 1.0)              # last record of file 1 -> first of file 2
            @test seam["t2m"].data[1, 1, 1] ≈ 282.6
            @test seam["t2m"].data[2, 1, 1] ≈ 282.5

            past = refresh(p, 2.0)              # BOTH ends inside the second file
            @test past["t2m"].data[1, 1, 1] ≈ 282.5
            @test past["t2m"].data[2, 1, 1] ≈ 282.6

            last = refresh(p, 3.0)              # final tick still degenerates
            @test last["t2m"].data[1, 1, 1] ≈ 282.6
            @test last["t2m"].data[2, 1, 1] ≈ 282.6
            @test last["time"].data[1] == last["time"].data[2]

            @test length(prefetch(p)) == 2      # one fetch per distinct file
        end
    end

    @testset "DISCRETE per-tick URLs (url-function form, no internal slice)" begin
        # url resolver form: the same fixture stands in for every tick; without
        # time_dim the provider returns the file's full native arrays per tick.
        p = discrete_provider(cache, _ -> era5, [0.0, 1.0]; format = "netcdf")
        @test refresh_times(p) == [0.0, 1.0]
        @test size(refresh(p, 1.0)["t2m"].data) == (2, 3, 3)
    end

    @testset "CSV points provider + variable selection" begin
        p = const_provider(cache, openaq; format = "csv", source_loader = "openaq",
                           reader_kwargs = (numeric_columns = ["latitude", "longitude", "value"],),
                           variables = ["value", "location_id"])
        nds = materialize(p)
        @test Set(variable_names(nds)) == Set(["value", "location_id"])   # restricted
        @test nds["value"].data == [152.3, 168.7, 98.1, 110.4]
        @test eltype(nds["value"].data) == Float64
        @test nds["location_id"].data == ["1", "1", "2", "2"]             # digit text stays string
    end

    @testset "prefetch warms the cache (offline hits, no decode)" begin
        p = const_provider(cache, era5; format = "netcdf")
        entries = prefetch(p)
        @test length(entries) == 1
        @test entries[1].status == :hit

        # DISCRETE per-tick URLs that collapse to one unique blob ⇒ one fetch
        pd = discrete_provider(cache, _ -> era5, [0.0, 1.0]; format = "netcdf")
        @test length(prefetch(pd)) == 1
    end

    @testset "construction + use guards" begin
        @test_throws ArgumentError const_provider(cache, era5; format = "netcdf", times = [1.0])
        @test_throws ArgumentError discrete_provider(cache, era5, Float64[]; format = "netcdf")
        @test_throws ArgumentError const_provider(cache, era5; format = "nonesuch")
        @test_throws ArgumentError Provider(cache, era5; format = "netcdf", time_dim = "time")  # CONST + time_dim
        # a DISCRETE provider needs an explicit time
        @test_throws ArgumentError materialize(discrete_provider(cache, era5, [0.0]; format = "netcdf"))
        # selecting a variable absent from the blob is a clear error
        bad = const_provider(cache, era5; format = "netcdf", variables = ["nope"])
        @test_throws ArgumentError materialize(bad)
    end

    # spec/registries.md §2.1 / ESM_COMPLIANCE_VALIDATION_MATRIX FORMAT-08-A-006:
    # a decode option the bound reader does not recognise is an ERROR AT
    # CONSTRUCTION, never a silently-ignored key. The failure this prevents is
    # not loud: a mis-typed `member_filter` selects nothing and the table reads
    # back short, arbitrarily far from its cause.
    @testset "unrecognised reader_kwargs are refused at construction (§2.1)" begin
        # every registered reader declares its own decode options...
        @test :member_glob in reader_option_keys(FF10Reader())
        @test :skip_header_row in reader_option_keys(FF10Reader())
        @test Set(reader_option_keys(ZarrReader())) == Set([:variables, :select])
        # netcdf takes a third: `records`, the cadence owner's record pushdown
        @test Set(reader_option_keys(NetCDFReader())) == Set([:variables, :select, :records])

        # ...and an option outside that set fails the Provider, naming it.
        e = try
            const_provider(cache, era5; format = "ff10",
                           reader_kwargs = (; member_filter = "*egu*"))
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("member_filter", e.msg)
        @test occursin("member_glob", e.msg)       # says what it DOES take

        # the correctly-spelled option constructs fine (the check is not a blanket ban)
        @test const_provider(cache, era5; format = "ff10",
                             reader_kwargs = (; member_glob = "*egu*")) isa Provider

        # a store-backed reader is checked through `read_store`, not `read_native`
        @test_throws ArgumentError const_provider(cache, era5; format = "zarr",
                                                  reader_kwargs = (; selct = nothing))
        @test const_provider(cache, era5; format = "zarr",
                             reader_kwargs = (; select = nothing)) isa Provider
    end
end
