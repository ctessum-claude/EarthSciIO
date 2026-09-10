# Transport backends (spec/registries.md §1).
#
# A transport fetches a resolved URL's bytes into a staging file. Transports are
# constructed and called ONLY when offline=false (offline mode bypasses the
# transport registry entirely — see cache.jl / spec/offline-mode.md §2).

"""Result of a transport fetch. `:not_modified` means a 304 — reuse the cache."""
struct FetchResult
    status::Symbol            # :downloaded | :not_modified
    etag::Union{String,Nothing}
    last_modified::Union{String,Nothing}
    bytes_written::Int
end

# --- auth seam --------------------------------------------------------------
# Pluggable per-realm credential resolver, injected into the transport. The
# realm name (cds/firms/openaq/rda) reaches the manifest; the credential never
# does. New realms are new resolvers — not a transport edit.

"""Resolves request auth headers for a realm. Never stored in the manifest."""
abstract type AuthResolver end

"""No authentication (the default)."""
struct NoAuth <: AuthResolver end

"""Bearer/token auth (CDS / FIRMS / OpenAQ / RDA tokens, generic bearer)."""
struct BearerAuth <: AuthResolver
    token::String
end

auth_headers(::NoAuth, ::AbstractString) = Pair{String,String}[]
auth_headers(a::BearerAuth, ::AbstractString) = ["Authorization" => string("Bearer ", a.token)]

# Resolve the auth for a realm from whatever the caller supplied: nothing → no
# auth; a single resolver → used as-is; a realm→resolver map → looked up.
resolve_auth(::Nothing, ::Any) = NoAuth()
resolve_auth(a::AuthResolver, ::Any) = a
resolve_auth(m::AbstractDict, realm) =
    realm === nothing ? NoAuth() : get(m, realm, NoAuth())

# --- http(s) transport (Downloads / libcurl) --------------------------------

# Robustness (esio zarr-over-S3): a plain `Downloads.request` sets NO timeout. Two
# distinct failures wedge a chunked zarr scan of hundreds of objects FOREVER at 0%
# CPU: (1) a STALLED socket (S3 accepts the request then delivers nothing), and
# (2) a LOST-WAKEUP deadlock in Downloads.jl's async coordination — the transfer's
# completion notification is dropped and the waiting task never resumes (the sample
# shows the scheduler parked in uv_run with no runnable task, NO curl activity). A
# libcurl low-speed abort fixes (1) but not (2); a Downloads-level `timeout` fixes
# (2) because its Timer still fires on the (alive) event loop and cancels the
# request. We apply BOTH, plus: rebuild the Downloader after any failure so a
# poisoned multi-handle can't wedge later chunks, and retry with capped backoff.
# All knobs are env-overridable; defaults suit large chunked reads over flaky S3.
_http_env_int(name, default) = parse(Int, get(ENV, name, string(default)))
const _HTTP_DOWNLOADER = Ref{Any}(nothing)
function _http_downloader()
    d = _HTTP_DOWNLOADER[]
    d === nothing || return d
    d = Downloads.Downloader()
    lo_limit = _http_env_int("EARTHSCIIO_HTTP_LOW_SPEED_LIMIT", 1024)  # bytes/s floor
    lo_time  = _http_env_int("EARTHSCIIO_HTTP_LOW_SPEED_TIME", 30)     # ...for this long → abort
    conn_to  = _http_env_int("EARTHSCIIO_HTTP_CONNECT_TIMEOUT", 30)    # connect timeout (s)
    d.easy_hook = (easy, info) -> begin
        Downloads.Curl.setopt(easy, Downloads.Curl.CURLOPT_LOW_SPEED_LIMIT, lo_limit)
        Downloads.Curl.setopt(easy, Downloads.Curl.CURLOPT_LOW_SPEED_TIME,  lo_time)
        Downloads.Curl.setopt(easy, Downloads.Curl.CURLOPT_CONNECTTIMEOUT,  conn_to)
    end
    _HTTP_DOWNLOADER[] = d
    return d
end
# Drop the cached Downloader so the next fetch builds a fresh multi-handle (called
# after a failed/aborted transfer, whose handle may be wedged).
_reset_http_downloader!() = (_HTTP_DOWNLOADER[] = nothing)

# The ceiling on one fetch (both on any single extended attempt and, with the
# retry floor below, on the whole call). It depends on WHAT is being fetched, and
# only the CALLER knows that -- the transport must never infer it from the URL or
# from a size it has not fetched yet:
#
#   * a WHOLE BLOB (`store_read=false`, the default) is one self-contained file
#     and can legitimately be multi-GB -- a 0.25 deg GEOS-FP A3dyn day is 3.80 GB
#     -- so it keeps the long 7200 s ceiling: better a slow success than a failure
#     that forces the whole file to be downloaded again from byte zero.
#   * a STORE-BACKED READ (`store_read=true`) is ONE OBJECT of a directory-like
#     store -- a Zarr chunk, `.zarray`, `.zattrs` -- and `fetch!` is called once
#     PER OBJECT (zarr.jl), hundreds of times in one scan. A chunk is small by
#     construction (the pinned ISRM store's are ~21 MB decompressed; a chunk that
#     needed hours would defeat the point of chunking), so 2 h per object is the
#     wrong shape of bound: a pathological source would block on ONE chunk for
#     two hours and the scan would take that times its object count. Minutes, not
#     hours: 600 s still covers a 100 MB chunk at a very poor 200 KB/s, and still
#     leaves room for one full 90 -> 360 s extension step plus a retry after it.
#
# Both go through the same `max(ceiling, tries * timeout)` floor at the call site,
# so NEITHER ceiling can take away retries the un-extended schedule would have had
# -- lowering it only shortens the extension ladder.
_http_store_read_ceiling(store_read::Bool) =
    store_read ? _http_env_int("EARTHSCIIO_HTTP_TIMEOUT_MAX_STORE", 600) :
                 _http_env_int("EARTHSCIIO_HTTP_TIMEOUT_MAX", 7200)

"""HTTP/HTTPS transport: GET with conditional-GET revalidation. Mirror failover
is handled at the call site (the cache tries mirror URLs in order)."""
struct HttpTransport <: Transport end
schemes(::HttpTransport) = ["http", "https"]

function fetch!(::HttpTransport, url::AbstractString, dest::AbstractString;
                conditional = NamedTuple(), auth::AuthResolver = NoAuth(),
                store_read::Bool = false)
    headers = Pair{String,String}[]
    append!(headers, auth_headers(auth, url))
    et = get(conditional, :etag, nothing)
    lm = get(conditional, :last_modified, nothing)
    et === nothing || push!(headers, "If-None-Match" => et)
    lm === nothing || push!(headers, "If-Modified-Since" => lm)

    tries   = max(1, _http_env_int("EARTHSCIIO_HTTP_RETRIES", 5))
    timeout = Float64(_http_env_int("EARTHSCIIO_HTTP_TIMEOUT", 90))  # per-ATTEMPT cap (s)
    tmax    = Float64(_http_store_read_ceiling(store_read))
    base_timeout = timeout
    # `timeout` is a TOTAL per-request cap, so on its own it also caps the SIZE
    # of a blob this transport can fetch: at a realistic 40 MB/s the 90 s default
    # gives up at ~3.5 GB, and single-file scientific archives are already past
    # that (a 0.25 deg GEOS-FP A3dyn day is 3.80 GB). Raising the default instead
    # would blunt what the cap is for -- the lost-wakeup deadlock, which must not
    # be allowed to hang for the raised value.
    #
    # Distinguish the two by PROGRESS, which is exactly what separates them: a
    # deadlocked or stalled transfer delivers no bytes, a healthy large one does.
    # An attempt that timed out HAVING GROWN `dest` earns a bigger budget and is
    # not charged a retry; one that timed out at a standstill is the failure the
    # cap exists to bound, and it still dies in `timeout` seconds. The low-speed
    # abort above (bytes/s floor, independently configurable) catches a trickle
    # BELOW the floor -- but a trickle just ABOVE it is aborted by nothing, and
    # grows `dest` on every attempt, so it earns every extension.
    #
    # The ladder must nonetheless compose into a BOUNDED wall time: `fetch!` is
    # called once per zarr chunk object (zarr.jl), so an unbounded per-URL retry
    # budget is the very hang this cap exists to prevent -- a source trickling
    # just above the bytes/s floor is never aborted by libcurl and would walk the
    # whole ladder (90+360+1440+5760+7200 s) and then still be charged its
    # retries. One deadline for the whole call bounds that: `tmax` (see
    # `_http_store_read_ceiling` -- shorter for a per-object store read than for a
    # whole blob) is the ceiling on the ENTIRE fetch, and each attempt is clamped
    # to what is left.
    # The floor of `tries * timeout` keeps the deadline from ever taking away
    # retries the un-extended schedule would have had, so lowering `TIMEOUT_MAX`
    # only shortens the extension ladder -- it never makes plain retries worse.
    budget   = max(tmax, tries * timeout)
    deadline = time() + budget
    extensions = 0
    local resp
    attempt = 0
    while attempt < tries
        attempt += 1
        ok = false
        try
            resp = Downloads.request(url; method = "GET", output = dest,
                                     headers = headers, throw = false,
                                     timeout = min(timeout, max(deadline - time(), 0.001)),
                                     downloader = _http_downloader())
            # CRITICAL: with `throw=false`, a transport failure (stall abort, connect
            # timeout, or a Downloads-level `timeout` cancelling a lost-wakeup
            # deadlock) is RETURNED as a `Downloads.RequestError`, NOT thrown — only a
            # `Response` (any HTTP status) is a real reply. Treat a non-Response as a
            # failed attempt. (An HTTP 404/5xx is a Response and is handled below by
            # status, never retried here.)
            ok = resp isa Downloads.Response
        catch err
            resp = err   # belt-and-suspenders: a future version might throw instead
            ok = false
        end
        ok && break
        _reset_http_downloader!()   # rebuild the possibly-wedged multi-handle
        # Progress-earned extension: this attempt moved bytes, so it is a large
        # transfer outgrowing its budget, not a wedged one. Grow the cap and
        # refund the attempt (bounded, so a pathologically slow server still
        # terminates).
        #
        # The progress signal is the bytes THIS attempt wrote, which is exactly
        # `filesize(dest)`: every attempt reopens `dest` for writing and so
        # truncates it (see the backoff comment below), and therefore restarts
        # the transfer at byte zero. Comparing against the PREVIOUS attempt's
        # leftover size instead would mis-read a healthy transfer whose rate
        # merely dropped -- 3.6 GB in the first attempt then 1.8 GB in the
        # (longer) second reads as "shrank", i.e. as a wedge -- and would
        # reset the budget and burn the retries on a transfer that is plainly
        # moving bytes.
        grew = (isfile(dest) ? filesize(dest) : 0) > 0
        if grew && timeout < tmax && extensions < 6 && time() < deadline
            timeout = min(timeout * 4, tmax)
            extensions += 1
            attempt -= 1
            continue
        end
        # No progress: whatever this was, it is not a transfer outgrowing its
        # budget, so hand the next attempt the ORIGINAL cap -- a wedge must never
        # inherit an extension earned by an earlier, healthy attempt.
        timeout = base_timeout
        if attempt >= tries || time() >= deadline
            resp isa Exception && throw(resp)
            error("http transport: GET $url failed after $attempt attempt(s) " *
                  "(within a $(round(Int, budget)) s budget): $resp")
        end
        sleep(min(2.0^(attempt - 1), 10.0))   # partial `dest` truncated by next open
    end
    if resp.status == 304
        return FetchResult(:not_modified, et, lm, 0)
    elseif 200 <= resp.status < 300
        return FetchResult(:downloaded, _header(resp, "etag"),
                           _header(resp, "last-modified"), filesize(dest))
    else
        error("http transport: GET $url returned HTTP status $(resp.status)")
    end
end

function _header(resp::Downloads.Response, name::AbstractString)
    for (k, v) in resp.headers
        lowercase(String(k)) == name && return String(v)
    end
    return nothing
end

# --- file transport (local copy) --------------------------------------------

"""`file://` transport: copy a local file into the cache. Expands
`\${EARTHSCIDATADIR}` (and other `\$VAR`) inside `file://` templates so a
pre-populated local mirror (the `nei2016` pattern) is found."""
struct FileTransport <: Transport end
schemes(::FileTransport) = ["file"]

function fetch!(::FileTransport, url::AbstractString, dest::AbstractString;
                conditional = NamedTuple(), auth::AuthResolver = NoAuth(),
                store_read::Bool = false)   # a local copy has no timeout to bound
    src = file_url_to_path(url)
    isfile(src) || error("file transport: source not found: $src (from $url)")
    cp(src, dest; force = true)
    return FetchResult(:downloaded, nothing, nothing, filesize(dest))
end

"""Expand `\${VAR}` and `\$VAR` from the environment (empty if unset)."""
function expand_env(s::AbstractString)
    s = replace(s, r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}" => m -> get(ENV, m[3:prevind(m, lastindex(m))], ""))
    s = replace(s, r"\$([A-Za-z_][A-Za-z0-9_]*)" => m -> get(ENV, m[2:end], ""))
    return s
end

"""Map a `file://` URL to a local filesystem path, expanding env templates."""
function file_url_to_path(url::AbstractString)
    startswith(url, "file://") || error("not a file:// URL: $url")
    rest = expand_env(url[8:end])              # strip "file://"
    startswith(rest, "localhost/") && (rest = rest[10:end])
    return startswith(rest, "/") ? rest : string("/", rest)
end

# --- s3 transport (ACTIVE: anonymous rewrite over the http transport) -------

"""Default S3 region — the pinned InMAP ISRM bucket lives in us-east-2."""
const DEFAULT_S3_REGION = "us-east-2"

"""Resolve the S3 region: explicit arg -> `\$EARTHSCI_S3_REGION` ->
`\$AWS_REGION` -> [`DEFAULT_S3_REGION`]."""
function resolve_s3_region(region::Union{Nothing,AbstractString} = nothing)
    region === nothing || return String(region)
    for k in ("EARTHSCI_S3_REGION", "AWS_REGION")
        v = get(ENV, k, "")
        isempty(v) || return v
    end
    return DEFAULT_S3_REGION
end

"""Rewrite `s3://<bucket>/<key…>` to regional virtual-hosted HTTPS
(`https://<bucket>.s3.<region>.amazonaws.com/<key>`)."""
function s3_https_url(s3_url::AbstractString, region::Union{Nothing,AbstractString} = nothing)
    startswith(s3_url, "s3://") || error("not an s3:// URL: $s3_url")
    rest = s3_url[6:end]                       # strip "s3://"
    slash = findfirst('/', rest)
    slash === nothing && error("s3:// URL has no object key: $s3_url")
    bucket = rest[1:prevind(rest, slash)]
    key = rest[nextind(rest, slash):end]
    isempty(bucket) && error("s3:// URL has an empty bucket: $s3_url")
    return "https://$bucket.s3.$(resolve_s3_region(region)).amazonaws.com/$key"
end

"""Anonymous `s3://` transport: rewrite `s3://<bucket>/<key>` to regional
virtual-hosted HTTPS and delegate the plain GET to the `http` transport (no AWS
SDK / SigV4). The canonical `s3://` URL stays in the cache key + manifest. The
region defaults to us-east-2, overridable via `\$EARTHSCI_S3_REGION`/`\$AWS_REGION`
or the `region` field. Conditional GET + auth thread through the delegate."""
struct S3Transport <: Transport
    region::Union{Nothing,String}
    http::HttpTransport
end
S3Transport(; region = nothing) = S3Transport(region === nothing ? nothing : String(region),
                                              HttpTransport())
schemes(::S3Transport) = ["s3"]

function fetch!(t::S3Transport, url::AbstractString, dest::AbstractString;
                conditional = NamedTuple(), auth::AuthResolver = NoAuth(),
                store_read::Bool = false)
    https_url = s3_https_url(url, t.region)
    return fetch!(t.http, https_url, dest; conditional = conditional, auth = auth,
                  store_read = store_read)
end
