# CDS (Copernicus Climate Data Store) API v1 transport.
#
# Ports EarthSciData.jl's `cds_api.jl` ~verbatim onto the transport seam
# (spec/registries.md §1). CDS is not a plain GET: a request is SUBMITTED, the
# job is POLLED until it succeeds, and only then is an asset href DOWNLOADED.
# The transport contract is still "given a resolved URL, write its bytes into a
# staging file", so the whole submit/poll/download flow is hidden behind one
# `fetch!`. The dataset + request travel IN the URL — a `cds://` URL is the
# transport's only input channel — and the cache content-addresses on that URL,
# so an identical request is a fast-path hit (skip-if-exists) with no API call.
#
#   cds://<dataset>?<canonical-JSON of the request>
#
# The request rides as RAW canonical JSON — no `request=` parameter, no
# percent-encoding (`spec/registries.md` §1). Canonical JSON (recursively sorted
# keys, no whitespace) makes the URL — and thus the cache key
# (`sha256(resolved_url)`) — BYTE-IDENTICAL for the same logical request across
# the Python / Julia / Rust tracks, so a file one track pulls from CDS is reused
# byte-for-byte by the others.

const CDS_API_URL = "https://cds.climate.copernicus.eu/api"
const CDS_POLL_INTERVAL = 5     # seconds between job-status polls
const CDS_TIMEOUT = 600         # seconds before giving up on a job

# --- auth -------------------------------------------------------------------
# CDS v1 authenticates with a `PRIVATE-TOKEN: <key>` header (NOT bearer). The
# key is resolved, in priority order, from an injected resolver, then the
# `CDSAPI_KEY` env var, then the `key:` line of `~/.cdsapirc` — the credential
# never touches the manifest (only the realm name `cds` is recorded).

"""CDS `PRIVATE-TOKEN` auth carrying an explicit API key. Injected through the
cache's `auth` map (`auth = Dict("cds" => CdsAuth(key))`); without it the `cds`
transport falls back to `CDSAPI_KEY` / `~/.cdsapirc` (see [`cds_api_key`])."""
struct CdsAuth <: AuthResolver
    token::String
end
auth_headers(a::CdsAuth, ::AbstractString) = ["PRIVATE-TOKEN" => a.token]

# Token the transport will authenticate with: an injected resolver wins; the
# default (NoAuth) falls back to the env / ~/.cdsapirc key.
cds_token(::NoAuth) = cds_api_key()
cds_token(a::CdsAuth) = a.token
cds_token(a::BearerAuth) = a.token

"""
    _read_cdsapirc(path=joinpath(homedir(), ".cdsapirc")) -> Dict{String,String}

Parse the `url:`/`key:` lines of a `.cdsapirc` file (the standard CDS API
config). Returns an empty dict if the file is absent. Path-parametrized so it
is testable without touching the real home directory."""
function _read_cdsapirc(path::AbstractString = joinpath(homedir(), ".cdsapirc"))
    out = Dict{String,String}()
    isfile(path) || return out
    for line in eachline(path)
        m = match(r"^\s*(url|key)\s*:\s*(.*?)\s*$", line)
        m === nothing && continue
        out[m.captures[1]] = String(m.captures[2])
    end
    return out
end

"""
    cds_api_key(; rc=_read_cdsapirc()) -> String

The CDS API key, resolved from `ENV["CDSAPI_KEY"]` then the `key:` line of
`~/.cdsapirc`. Errors with setup guidance if neither is present."""
function cds_api_key(; rc::AbstractDict = _read_cdsapirc())
    haskey(ENV, "CDSAPI_KEY") && return String(ENV["CDSAPI_KEY"])
    haskey(rc, "key") && return rc["key"]
    error("CDS API key not found. Set the CDSAPI_KEY environment variable or " *
          "create ~/.cdsapirc with 'key: <your-key>'.")
end

"""
    cds_api_endpoint(; rc=_read_cdsapirc()) -> String

The CDS API base endpoint, resolved from `ENV["CDSAPI_URL"]` then the `url:`
line of `~/.cdsapirc`, defaulting to the public v1 endpoint
(`$(CDS_API_URL)`). The env override is what points the transport at a mock
server in tests; it is NOT part of the cache key (the dataset+request are)."""
function cds_api_endpoint(; rc::AbstractDict = _read_cdsapirc())
    haskey(ENV, "CDSAPI_URL") && return String(ENV["CDSAPI_URL"])
    haskey(rc, "url") && return rc["url"]
    return CDS_API_URL
end

cds_poll_interval() = parse(Float64, get(ENV, "CDSAPI_POLL_SECONDS", string(CDS_POLL_INTERVAL)))
cds_timeout() = parse(Float64, get(ENV, "CDSAPI_TIMEOUT_SECONDS", string(CDS_TIMEOUT)))

# --- canonical URL <-> (dataset, request) -----------------------------------

# Canonical JSON: recursively sorted object keys, no whitespace, so the same
# logical request always serializes to the same bytes (and thus the same cache
# key) regardless of dict insertion order or language track.
_canonical_json(x::AbstractString) = JSON.json(String(x))
_canonical_json(x::Bool) = x ? "true" : "false"
_canonical_json(x::Integer) = string(x)
_canonical_json(x::Real) = JSON.json(x)
_canonical_json(x::AbstractVector) = string("[", join((_canonical_json(v) for v in x), ","), "]")
function _canonical_json(d::AbstractDict)
    ks = sort!(String[string(k) for k in keys(d)])
    return string("{", join((string(JSON.json(k), ":", _canonical_json(d[k])) for k in ks), ","), "}")
end

"""
    cds_url(dataset, request) -> String

Build the content-addressable `cds://` URL for a CDS retrieve. `request` is the
`Dict` of CDS request parameters (see [`era5_pressure_request`]).

The shared, cross-language form is `cds://<dataset>?<canonical-request-json>`
(`spec/registries.md` §1): the canonical JSON is appended to the query verbatim
— **no** `request=` parameter and **no** percent-encoding — so the URL string
(and therefore `sha256(resolved_url)`) is byte-identical to the Rust
([`build_cds_url`]) and Python ([`encode_cds_url`]) tracks for the same request,
and a repeat is a cross-language cache hit."""
cds_url(dataset::AbstractString, request) =
    string("cds://", dataset, "?", _canonical_json(request))

"""
    parse_cds_url(url) -> (dataset::String, request)

Inverse of [`cds_url`]: recover the dataset and the request object from a
`cds://` URL. Splits on the first `?` (the JSON payload may itself contain `?`)
and parses the raw query as JSON, mirroring Rust `parse_cds_url` / Python
`decode_cds_url`. Used by the transport to rebuild the CDS submit; a malformed
URL fails here at the cache boundary, not as an opaque CDS error later."""
function parse_cds_url(url::AbstractString)
    startswith(url, "cds://") || error("not a cds:// URL: $url")
    rest = url[ncodeunits("cds://")+1:end]
    q = findfirst('?', rest)
    q === nothing && error("cds:// URL missing '?<request-json>': $url")
    dataset = rest[1:prevind(rest, q)]
    payload = rest[nextind(rest, q):end]
    isempty(dataset) && error("cds:// URL has an empty dataset: $url")
    isempty(payload) && error("cds:// URL has an empty request: $url")
    request = try
        JSON.parse(payload)
    catch
        error("cds:// request is not valid JSON: $url")
    end
    request isa AbstractDict || error("cds:// request must be a JSON object: $url")
    return String(dataset), request
end

# --- CDS API client (ported from EarthSciData.jl cds_api.jl) -----------------

# One Downloads.jl round-trip against the CDS API; returns the response body as
# a String and raises on any >=400 status (with the body for diagnosis).
function _cds_http(url::AbstractString, headers::Vector{<:Pair};
                   method::AbstractString = "GET", body = nothing)
    out = IOBuffer()
    resp = if body === nothing
        Downloads.request(url; headers = headers, output = out,
                          method = method, throw = false)
    else
        Downloads.request(url; headers = headers, output = out,
                          input = IOBuffer(body), method = method, throw = false)
    end
    text = String(take!(out))
    resp.status >= 400 && error("CDS API HTTP $(resp.status) for $url: $text")
    return text
end

"""
    cds_submit(dataset, request; api_key, endpoint) -> jobID

POST a CDS retrieve request (`{inputs: request}`) and return the accepted job
ID. Authenticated with the `PRIVATE-TOKEN` header."""
function cds_submit(dataset::AbstractString, request;
                    api_key::AbstractString = cds_api_key(),
                    endpoint::AbstractString = cds_api_endpoint())
    url = "$(endpoint)/retrieve/v1/processes/$(dataset)/execution"
    body = JSON.json(Dict("inputs" => request))
    headers = ["PRIVATE-TOKEN" => api_key,
               "Content-Type" => "application/json",
               "Expect" => ""]   # suppress libcurl 100-continue for large bodies
    resp = _cds_http(url, headers; method = "POST", body = body)
    data = JSON.parse(resp)
    get(data, "status", "") in ("accepted", "running", "successful") ||
        error("CDS API request failed: $resp")
    haskey(data, "jobID") || error("CDS API submit returned no jobID: $resp")
    return data["jobID"]
end

"""
    cds_wait(job_id; api_key, endpoint, poll_interval, timeout) -> href

Poll a CDS job until it succeeds, then return the result asset's download
href. Errors on a failed job or on timeout."""
function cds_wait(job_id::AbstractString;
                  api_key::AbstractString = cds_api_key(),
                  endpoint::AbstractString = cds_api_endpoint(),
                  poll_interval::Real = cds_poll_interval(),
                  timeout::Real = cds_timeout())
    url = "$(endpoint)/retrieve/v1/jobs/$(job_id)"
    headers = ["PRIVATE-TOKEN" => api_key]
    start = time()
    while true
        resp = _cds_http(url, headers)
        status = get(JSON.parse(resp), "status", "")
        if status == "successful"
            results = JSON.parse(_cds_http("$(url)/results", headers))
            return results["asset"]["value"]["href"]
        elseif status == "failed"
            error("CDS API job $job_id failed: $resp")
        elseif time() - start > timeout
            error("CDS API job $job_id timed out after $timeout seconds.")
        end
        sleep(poll_interval)
    end
end

"""
    cds_retrieve(dataset, request, output_path; api_key, endpoint) -> output_path

Submit a CDS request, wait for it, and download the result to `output_path`
(skipping the network round-trip if the file already exists). Convenience for
manual/standalone use; the cache fetch path uses the `cds` transport instead."""
function cds_retrieve(dataset::AbstractString, request, output_path::AbstractString;
                      api_key::AbstractString = cds_api_key(),
                      endpoint::AbstractString = cds_api_endpoint())
    isfile(output_path) && return output_path
    mkpath(dirname(output_path))
    job_id = cds_submit(dataset, request; api_key = api_key, endpoint = endpoint)
    href = cds_wait(job_id; api_key = api_key, endpoint = endpoint)
    Downloads.download(href, output_path)
    return output_path
end

# --- the cds transport ------------------------------------------------------

"""`cds://` transport: runs the CDS submit → poll → download flow into the
cache's staging file. The dataset + request are carried in the URL (see
[`cds_url`]); present blobs are immutable (a closed past period), so the cache's
content-addressed fast path is the skip-if-exists — `conditional` is unused
because CDS has no conditional-GET."""
struct CdsTransport <: Transport end
schemes(::CdsTransport) = ["cds"]

function fetch!(::CdsTransport, url::AbstractString, dest::AbstractString;
                conditional = NamedTuple(), auth::AuthResolver = NoAuth(),
                store_read::Bool = false)   # CDS serves whole blobs, never a store
    dataset, request = parse_cds_url(url)
    api_key = cds_token(auth)
    endpoint = cds_api_endpoint()
    job_id = cds_submit(dataset, request; api_key = api_key, endpoint = endpoint)
    href = cds_wait(job_id; api_key = api_key, endpoint = endpoint)
    Downloads.download(href, dest)
    return FetchResult(:downloaded, nothing, nothing, filesize(dest))
end
