# HTTP transport (spec/registries.md §1): real download + conditional-GET
# revalidation, exercised against a hermetic localhost server (no external
# network). The offline conformance suite opens no socket; this online-path
# unit test deliberately does, against 127.0.0.1 only.

using Sockets

function _read_http_request(io)
    line = readline(io)
    headers = Dict{String,String}()
    while true
        h = readline(io)
        isempty(h) && break
        kv = split(h, ":"; limit = 2)
        length(kv) == 2 && (headers[lowercase(strip(kv[1]))] = strip(kv[2]))
    end
    return line, headers
end

# Minimal HTTP/1.1 server: 200 with body+ETag, or 304 when If-None-Match
# matches. One request per connection (Connection: close). Serves until the
# listen socket is closed.
function _start_test_server(payload::Vector{UInt8}, etag::AbstractString)
    server = listen(Sockets.localhost, 0)
    port = Int(getsockname(server)[2])
    task = @async begin
        try
            while true
                conn = accept(server)
                @async try
                    _, headers = _read_http_request(conn)
                    if occursin(etag, get(headers, "if-none-match", ""))
                        write(conn, "HTTP/1.1 304 Not Modified\r\n",
                              "ETag: \"$etag\"\r\nConnection: close\r\n\r\n")
                    else
                        write(conn, "HTTP/1.1 200 OK\r\n",
                              "Content-Length: $(length(payload))\r\n",
                              "ETag: \"$etag\"\r\nConnection: close\r\n\r\n")
                        write(conn, payload)
                    end
                catch
                finally
                    close(conn)
                end
            end
        catch
            # listen socket closed -> accept throws -> exit
        end
    end
    return server, port, task
end

@testset "http transport — download + conditional GET (localhost)" begin
    payload = rand(UInt8, 2048)
    etag = "abc123"
    server, port, _ = _start_test_server(payload, etag)
    try
        root = mktempdir()
        url = "http://127.0.0.1:$port/data.nc"
        c = Cache(LocalStore(root); offline = false, verify = true)

        # initial GET -> downloaded, ETag captured into the manifest
        e1 = fetch_blob(c, url; source_loader = "httploader")
        @test e1.status == :downloaded
        @test read(e1.path) == payload
        @test e1.manifest.etag !== nothing
        @test occursin(etag, e1.manifest.etag)
        @test e1.manifest.bytes == length(payload)

        # forced revalidation -> server sees If-None-Match -> 304 -> reuse blob
        e2 = fetch_blob(c, url; revalidate = true)
        @test e2.status == :not_modified
        @test read(e2.path) == payload

        # plain re-fetch -> fast-path hit
        @test fetch_blob(c, url).status == :hit

        # TTL: age the manifest, then a finite ttl makes the blob stale and
        # forces the same conditional GET -> 304 (spec §4 step 3).
        key = cache_key(url)
        m = EarthSciIO.get_meta(c.store, key)
        aged = EarthSciIO.Manifest(m.url, m.etag, m.last_modified, m.sha256_content,
                                   m.bytes, "2000-01-01T00:00:00Z", m.source_loader,
                                   m.auth_realm)
        EarthSciIO.put_meta!(c.store, key, aged)
        @test fetch_blob(c, url; ttl = 3600).status == :not_modified

        # offline re-read of the http-fetched blob
        co = Cache(LocalStore(root); offline = true, verify = true)
        @test read(fetch_blob(co, url).path) == payload
    finally
        close(server)
    end
end

# Opt-in live smoke test (spec/offline-mode.md §4): never in CI. Set
# EARTHSCI_LIVE=1 to actually hit the network and seed a fixture.
if lowercase(get(ENV, "EARTHSCI_LIVE", "")) in ("1", "true", "yes")
    @testset "http transport — live smoke (EARTHSCI_LIVE)" begin
        root = mktempdir()
        url = "https://raw.githubusercontent.com/EarthSciML/EarthSciIO/main/README.md"
        c = Cache(LocalStore(root); offline = false)
        e = fetch_blob(c, url)
        @test e.status in (:downloaded, :hit)
        @test filesize(e.path) > 0
        # now reuse it offline
        co = Cache(LocalStore(root); offline = true)
        @test fetch_blob(co, url).status == :hit
    end
end

# --- progress-earned timeout extension ---------------------------------------
# A server that writes the body in slow pieces, so a transfer that is HEALTHY
# but larger than the per-request cap is indistinguishable from a stall until
# you look at whether bytes moved. Serves each connection more slowly than
# EARTHSCIIO_HTTP_TIMEOUT allows on the first attempt.
function _start_slow_server(payload::Vector{UInt8}, etag::AbstractString;
                            pieces::Int = 4, pause::Float64 = 0.6)
    server = listen(Sockets.localhost, 0)
    port = Int(getsockname(server)[2])
    task = @async begin
        try
            while true
                conn = accept(server)
                @async try
                    _read_http_request(conn)
                    write(conn, "HTTP/1.1 200 OK\r\n",
                          "Content-Length: $(length(payload))\r\n",
                          "ETag: \"$etag\"\r\nConnection: close\r\n\r\n")
                    n = cld(length(payload), pieces)
                    for i in 1:pieces
                        lo = (i - 1) * n + 1
                        lo > length(payload) && break
                        write(conn, @view payload[lo:min(lo + n - 1, length(payload))])
                        flush(conn)
                        sleep(pause)
                    end
                catch
                finally
                    close(conn)
                end
            end
        catch
        end
    end
    return server, port, task
end

@testset "http transport — a big-but-healthy transfer outgrows its cap" begin
    payload = rand(UInt8, 64 * 1024)
    server, port, _ = _start_slow_server(payload, "slow1"; pieces = 4, pause = 0.6)
    dest = tempname()
    try
        # 1 s cap against a ~2.4 s body: the first attempt times out having
        # moved bytes, which must buy a longer budget rather than a failure.
        withenv("EARTHSCIIO_HTTP_TIMEOUT" => "1",
                "EARTHSCIIO_HTTP_RETRIES" => "2",
                "EARTHSCIIO_HTTP_LOW_SPEED_TIME" => "60") do
            EarthSciIO._reset_http_downloader!()
            r = EarthSciIO.fetch!(EarthSciIO.HttpTransport(),
                                  "http://127.0.0.1:$port/slow.bin", dest)
            @test r.status == :downloaded
        end
        @test read(dest) == payload
    finally
        close(server); rm(dest; force = true)
        EarthSciIO._reset_http_downloader!()
    end
end

@testset "http transport — a wedged transfer is still bounded" begin
    # Accepts the connection, sends headers, then never sends the body: no
    # progress is ever made, so no extension may be earned and the call must
    # fail promptly rather than hanging for the extended budget.
    server = listen(Sockets.localhost, 0)
    port = Int(getsockname(server)[2])
    @async try
        while true
            conn = accept(server)
            @async try
                _read_http_request(conn)
                write(conn, "HTTP/1.1 200 OK\r\nContent-Length: 4096\r\n",
                      "Connection: close\r\n\r\n")
                sleep(30)     # ... and nothing else
            catch
            finally
                close(conn)
            end
        end
    catch
    end
    dest = tempname()
    try
        t0 = time()
        withenv("EARTHSCIIO_HTTP_TIMEOUT" => "1",
                "EARTHSCIIO_HTTP_RETRIES" => "2",
                "EARTHSCIIO_HTTP_LOW_SPEED_TIME" => "60") do
            EarthSciIO._reset_http_downloader!()
            @test_throws Exception EarthSciIO.fetch!(
                EarthSciIO.HttpTransport(), "http://127.0.0.1:$port/wedged.bin", dest)
        end
        @test time() - t0 < 15    # 2 attempts x 1 s cap + backoff, not 6 extensions
    finally
        close(server); rm(dest; force = true)
        EarthSciIO._reset_http_downloader!()
    end
end

# --- progress is per-attempt, not a diff against the last attempt ------------
# Every attempt reopens `dest` for writing and therefore RESTARTS the transfer
# at byte zero, so the only correct progress signal is "did THIS attempt write
# anything". A server whose delivered prefix shrinks from one attempt to the
# next (a rate drop, a slower mirror behind the same name, a throttle kicking
# in) is still plainly moving bytes and must keep earning budget.
function _start_shrinking_server(payload::Vector{UInt8}, prefixes::Vector{Int})
    server = listen(Sockets.localhost, 0)
    port = Int(getsockname(server)[2])
    conn_no = Threads.Atomic{Int}(0)
    @async begin
        try
            while true
                conn = accept(server)
                i = Threads.atomic_add!(conn_no, 1) + 1
                @async try
                    _read_http_request(conn)
                    write(conn, "HTTP/1.1 200 OK\r\n",
                          "Content-Length: $(length(payload))\r\n",
                          "ETag: \"shrink\"\r\nConnection: close\r\n\r\n")
                    if i <= length(prefixes)
                        write(conn, @view payload[1:prefixes[i]])
                        flush(conn)
                        sleep(20)            # ... then stall, so the attempt times out
                    else
                        write(conn, payload) # healthy: serve it all at once
                        flush(conn)
                    end
                catch
                finally
                    close(conn)
                end
            end
        catch
        end
    end
    return server, port
end

@testset "http transport — a shrinking prefix is still progress" begin
    payload = rand(UInt8, 64 * 1024)
    # Attempt 1 delivers 60 KiB then stalls; attempt 2 delivers only 24 KiB then
    # stalls. Both moved bytes, so both must earn budget; comparing attempt 2
    # against attempt 1's leftover instead calls it a wedge and gives up.
    server, port = _start_shrinking_server(payload, [60 * 1024, 24 * 1024])
    dest = tempname()
    try
        withenv("EARTHSCIIO_HTTP_TIMEOUT" => "1",
                "EARTHSCIIO_HTTP_RETRIES" => "1",
                "EARTHSCIIO_HTTP_LOW_SPEED_TIME" => "60") do
            EarthSciIO._reset_http_downloader!()
            r = EarthSciIO.fetch!(EarthSciIO.HttpTransport(),
                                  "http://127.0.0.1:$port/shrink.bin", dest)
            @test r.status == :downloaded
        end
        @test read(dest) == payload
    finally
        close(server); rm(dest; force = true)
        EarthSciIO._reset_http_downloader!()
    end
end

@testset "http transport — the whole fetch is bounded by TIMEOUT_MAX" begin
    # A source that trickles above the bytes/s floor forever is never aborted by
    # libcurl and grows `dest` on every attempt, so it earns every extension and
    # is then charged every retry with backoff. The whole call must be bounded by
    # `max(EARTHSCIIO_HTTP_TIMEOUT_MAX, RETRIES * TIMEOUT)` -- here 8 s -- not
    # just one attempt.
    server = listen(Sockets.localhost, 0)
    port = Int(getsockname(server)[2])
    @async try
        while true
            conn = accept(server)
            @async try
                _read_http_request(conn)
                write(conn, "HTTP/1.1 200 OK\r\nContent-Length: 1048576\r\n",
                      "Connection: close\r\n\r\n")
                for _ in 1:200
                    write(conn, zeros(UInt8, 512)); flush(conn); sleep(0.25)
                end
            catch
            finally
                close(conn)
            end
        end
    catch
    end
    dest = tempname()
    try
        t0 = time()
        withenv("EARTHSCIIO_HTTP_TIMEOUT" => "1",
                "EARTHSCIIO_HTTP_TIMEOUT_MAX" => "2",
                "EARTHSCIIO_HTTP_RETRIES" => "8",
                "EARTHSCIIO_HTTP_LOW_SPEED_TIME" => "60") do
            EarthSciIO._reset_http_downloader!()
            @test_throws Exception EarthSciIO.fetch!(
                EarthSciIO.HttpTransport(), "http://127.0.0.1:$port/trickle.bin", dest)
        end
        # Budget is 8 s; without a whole-call deadline the extension ladder plus
        # 8 backed-off retries runs for ~1 minute (measured: 57 s).
        @test time() - t0 < 20
    finally
        close(server); rm(dest; force = true)
        EarthSciIO._reset_http_downloader!()
    end
end

# --- a store-backed (per-object) read gets its own, shorter budget -----------
# `fetch!` runs once per zarr chunk object, so the whole-blob budget (2 h, right
# for a single multi-GB file) is the wrong bound for one small chunk: a source
# that trickles above the bytes/s floor could sit on ONE chunk for two hours,
# times the object count of a scan. The store path therefore passes
# `store_read=true`, which selects `EARTHSCIIO_HTTP_TIMEOUT_MAX_STORE`.
function _start_trickle_server(; chunk::Int = 512, pause::Float64 = 0.25,
                               claimed::Int = 1048576, pieces::Int = 400)
    server = listen(Sockets.localhost, 0)
    port = Int(getsockname(server)[2])
    @async try
        while true
            conn = accept(server)
            @async try
                _read_http_request(conn)
                write(conn, "HTTP/1.1 200 OK\r\nContent-Length: $claimed\r\n",
                      "Connection: close\r\n\r\n")
                for _ in 1:pieces
                    write(conn, zeros(UInt8, chunk)); flush(conn); sleep(pause)
                end
            catch
            finally
                close(conn)
            end
        end
    catch
    end
    return server, port
end

@testset "http transport — a store-backed read is bounded by TIMEOUT_MAX_STORE" begin
    # Driven through the zarr object-fetch helper (the real seam), not through
    # `fetch!` directly, so the test measures BEHAVIOUR and cannot be satisfied
    # by a signature alone. The whole-blob ceiling here is 25 s and the store one
    # 2 s; a per-object read must honour the latter.
    server, port = _start_trickle_server()
    root = mktempdir()
    try
        c = Cache(LocalStore(root); offline = false)
        t0 = time()
        withenv("EARTHSCIIO_HTTP_TIMEOUT" => "1",
                "EARTHSCIIO_HTTP_RETRIES" => "2",
                "EARTHSCIIO_HTTP_TIMEOUT_MAX" => "25",
                "EARTHSCIIO_HTTP_TIMEOUT_MAX_STORE" => "2",
                "EARTHSCIIO_HTTP_LOW_SPEED_TIME" => "60") do
            EarthSciIO._reset_http_downloader!()
            @test_throws Exception EarthSciIO._fetch_bytes(c, "http://127.0.0.1:$port/c/0.0.0")
        end
        # 2 s budget (its own floor is RETRIES x TIMEOUT = 2 s). On the whole-blob
        # ceiling this same fetch walks the extension ladder for its full 25 s.
        @test time() - t0 < 15
    finally
        close(server)
        EarthSciIO._reset_http_downloader!()
        rm(root; recursive = true, force = true)
    end
end

@testset "http transport — the store budget never shortens a whole-blob fetch" begin
    # The mirror of the test above: with a 1 s store ceiling and a 6 s whole-blob
    # ceiling, a default (`store_read=false`) fetch must spend the SIX -- the
    # shorter per-object bound must not leak into a multi-GB single-file download.
    server, port = _start_trickle_server()
    dest = tempname()
    try
        t0 = time()
        withenv("EARTHSCIIO_HTTP_TIMEOUT" => "1",
                "EARTHSCIIO_HTTP_RETRIES" => "2",
                "EARTHSCIIO_HTTP_TIMEOUT_MAX" => "6",
                "EARTHSCIIO_HTTP_TIMEOUT_MAX_STORE" => "1",
                "EARTHSCIIO_HTTP_LOW_SPEED_TIME" => "60") do
            EarthSciIO._reset_http_downloader!()
            @test_throws Exception EarthSciIO.fetch!(
                EarthSciIO.HttpTransport(), "http://127.0.0.1:$port/blob.bin", dest)
        end
        el = time() - t0
        @test el >= 4     # spent its own 6 s budget (a lower bound: load only lengthens it)
        @test el < 30
    finally
        close(server); rm(dest; force = true)
        EarthSciIO._reset_http_downloader!()
    end
end
