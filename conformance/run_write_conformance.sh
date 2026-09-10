#!/usr/bin/env bash
#
# Cross-language WRITE-conformance harness driver (streaming-output-sinks Wave 5).
#
# The write-side mirror of run_conformance.sh. Drives each language's Zarr v3
# sharded WRITER over a shared input spec to produce a store, reads every produced
# store back with every available track's reader (decoding to
# earthsciio/write-native-dump/v1), and asserts with conformance/crosscheck_write.py
# that every writer's store agrees with the spec oracle and pairwise within
# tolerance (RFC §16.6 — decoded-array agreement, NOT byte identity) plus
# structural/CF-metadata agreement across the stores.
#
# CODEC PROFILES. The run is parameterized over output codec profiles; each is a
# full independent write/read/compare round over its own spec variant:
#
#   diagnostic  conformance/write_spec.json       inner codec Blosc(zstd,5)+shuffle
#   wasm        conformance/write_spec_wasm.json  inner codec plain v3 zstd(5), no
#               Blosc — so the store is loadable by a WebAssembly/browser Zarr
#               reader (`zarrs`' blosc support comes from `blosc-src`, whose
#               vendored C sources don't build for wasm32-unknown-unknown; the
#               standard v3 zstd codec is pure Rust there). Sharding/crc32c are
#               unchanged — only the inner compressor differs.
#
# The dataset (and therefore the decoded oracle) is identical across profiles, so
# a profile that silently fell back to the wrong codec still has to produce the
# right VALUES *and* match the other tracks' declared inner codec chain.
#
# Usage:
#   conformance/run_write_conformance.sh [OUT_DIR]
#     OUT_DIR  where the stores + readback dumps go (default: a temp dir, removed
#              on exit). Pass a path to keep them for inspection.
#
# Toolchain overrides (for the Python-3.11+/zarr-3.x requirement and pinned envs):
#   PYTHON   python interpreter with the `zarr` extra installed (default: python3)
#   JULIA    julia launcher                                       (default: julia)
#   CARGO    cargo launcher                                       (default: cargo)
#   PROFILES space-separated codec profiles to run   (default: "diagnostic wasm")
#
# A track whose toolchain is unavailable is skipped with a logged reason (never
# silently dropped); the comparator still gates on whatever ran, and requires at
# least one store cross-read by at least one reader.
#
# Exit status: 0 = every profile's stores agree, 1 = a divergence in ANY profile.
# Any driver failing (build/write/read) also aborts non-zero.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
JULIA="${JULIA:-julia}"
CARGO="${CARGO:-cargo}"
PROFILES="${PROFILES:-diagnostic wasm}"

if [ "$#" -ge 1 ]; then
  OUT="$1"; mkdir -p "$OUT"
else
  OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fi

echo "cross-language WRITE-conformance harness"
echo "  repo:     $REPO_ROOT"
echo "  out:      $OUT"
echo "  profiles: $PROFILES"
echo

# --- which tracks can we run here? ------------------------------------------
have_python_zarr=0
if "$PYTHON" -c 'import zarr' >/dev/null 2>&1; then have_python_zarr=1; fi
have_julia=0; if command -v "$JULIA" >/dev/null 2>&1; then have_julia=1; fi
# The Julia drivers encode/decode chunks through weakdep EXTENSIONS (Blosc,
# CodecZstd), which `--project=julia` cannot import. Resolve them WITH
# EarthSciIO's own deps into one environment and run the drivers there — a
# separately-resolved env stacked onto LOAD_PATH mixes manifests and breaks
# precompilation (conformance/dumpers/julia_env.jl). Warm env => no network.
JULIA_ENV=""
if [ "$have_julia" = 1 ]; then
  JULIA_ENV="$("$JULIA" conformance/dumpers/julia_env.jl)"
fi
have_cargo=0; if command -v "$CARGO" >/dev/null 2>&1; then have_cargo=1; fi

# Build the Rust example ONCE up front (rather than per profile) so a build
# failure is reported once and the per-profile runs are pure execution.
rust_ok=0
if [ "$have_cargo" = 1 ]; then
  if "$CARGO" build --quiet --manifest-path rust/Cargo.toml --example conformance_write; then
    rust_ok=1
  else
    echo "[build] rust SKIPPED: cargo build failed (e.g. the netcdf-reader path dep is absent in this checkout)"
  fi
else
  echo "[build] rust SKIPPED: '$CARGO' not found"
fi

spec_for() {
  case "$1" in
    diagnostic) echo "conformance/write_spec.json" ;;
    wasm)       echo "conformance/write_spec_wasm.json" ;;
    *)          echo "conformance/write_spec_$1.json" ;;
  esac
}

overall=0

for PROFILE in $PROFILES; do
  SPEC="$(spec_for "$PROFILE")"
  if [ ! -f "$SPEC" ]; then
    echo "ERROR: no input spec for profile '$PROFILE' (expected $SPEC)" >&2
    exit 2
  fi
  POUT="$OUT/$PROFILE"; mkdir -p "$POUT"

  echo "############################################################"
  echo "# profile: $PROFILE   spec: $SPEC"
  echo "############################################################"
  echo

  WRITERS=()   # labels of writers whose store was produced
  READBACKS=() # readback dump files
  STORE_ARGS=()

  # --- writers --------------------------------------------------------------
  if [ "$have_python_zarr" = 1 ]; then
    echo "[write] python -> $POUT/store_python"
    "$PYTHON" conformance/dumpers/write_python.py "$POUT/store_python" "$SPEC"
    WRITERS+=(python); STORE_ARGS+=(--store "python=$POUT/store_python")
  else
    echo "[write] python SKIPPED: '$PYTHON -c import zarr' failed (needs Python>=3.11 + the zarr extra)"
  fi

  if [ "$have_julia" = 1 ]; then
    echo "[write] julia -> $POUT/store_julia"
    "$JULIA" --project="$JULIA_ENV" conformance/dumpers/write_julia.jl "$POUT/store_julia" "$SPEC"
    WRITERS+=(julia); STORE_ARGS+=(--store "julia=$POUT/store_julia")
  else
    echo "[write] julia SKIPPED: '$JULIA' not found"
  fi

  if [ "$rust_ok" = 1 ]; then
    echo "[write] rust -> $POUT/store_rust"
    "$CARGO" run --quiet --manifest-path rust/Cargo.toml --example conformance_write -- \
      "$POUT/store_rust" "$SPEC"
    WRITERS+=(rust); STORE_ARGS+=(--store "rust=$POUT/store_rust")
  fi

  echo
  # --- readbacks: every available reader over every produced store ----------
  for w in "${WRITERS[@]}"; do
    if [ "$have_python_zarr" = 1 ]; then
      echo "[read ] python reader over $w store"
      "$PYTHON" conformance/dumpers/read_python.py \
        "$POUT/store_$w" "$w" "$POUT/rd_python_from_$w.json" "$SPEC"
      READBACKS+=("$POUT/rd_python_from_$w.json")
    fi
    if [ "$have_julia" = 1 ]; then
      echo "[read ] julia reader over $w store"
      "$JULIA" --project="$JULIA_ENV" conformance/dumpers/read_julia.jl \
        "$POUT/store_$w" "$w" "$POUT/rd_julia_from_$w.json" "$SPEC"
      READBACKS+=("$POUT/rd_julia_from_$w.json")
    fi
  done

  echo
  echo "[gate ] cross-language write comparison (profile=$PROFILE)"
  echo
  if "$PYTHON" conformance/crosscheck_write.py "${READBACKS[@]}" "${STORE_ARGS[@]}" --spec "$SPEC"; then
    echo "[gate ] profile $PROFILE: PASS"
  else
    echo "[gate ] profile $PROFILE: FAIL"
    overall=1
  fi
  echo
done

echo "############################################################"
if [ "$overall" = 0 ]; then
  echo "# ALL PROFILES PASS: $PROFILES"
else
  echo "# FAILURE in at least one profile (see above)"
fi
echo "############################################################"
exit "$overall"
