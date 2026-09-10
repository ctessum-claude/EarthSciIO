"""Active format readers (component (b)) — the decode half of the Provider.

A reader opens a cached blob and returns RAW native-grid arrays keyed by the
on-disk ``file_variable`` name. It applies ONLY the format/CF decode pinned by
``spec/conformance.md`` §3; it does NOT remap variable names or convert units
(those stay in ESS — Risk R3). Readers register into the shared
:data:`~earthsciio.registry.format_registry` by name (``netcdf``, ``csv``), so a
new format plugs in with a new :class:`~earthsciio.registry.Reader` + one
``register`` line — never a Provider edit (``spec/registries.md`` §4).

These are the **active** counterparts to the ``zarr`` stub
(:class:`earthsciio.backends.zarr.ZarrReader`). They mirror the Julia
``NetCDFReader``/``CSVReader`` (``julia/src/readers.jl``) and the Rust
``netcdf`` reader, and they decode **byte-identically** to the conformance
oracle (``conformance/verify.py``) so cross-language array equality holds
(``esio-9nb.9``).

The netcdf reader needs xarray + netCDF4 (the optional ``netcdf`` extra); it
imports them lazily so the cache/transport core stays lean.
"""

from __future__ import annotations

import csv as _csv
import datetime as _dt
import decimal as _decimal
import fnmatch as _fnmatch
import io as _io
import zipfile as _zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .native import NativeDataset, NativeField
from .registry import Registry, format_registry
from .selection import _parse_axis, _resolve_axis_indices, _select_axes

__all__ = [
    "NetCDFReader",
    "CSVReader",
    "GeoTIFFReader",
    "FF10Reader",
    "ShapefileReader",
    "ParquetReader",
    "register_format_readers",
]


# --------------------------------------------------------------------------- #
# Shared decode helper (spec/conformance.md §3, "Numeric dtype").
# --------------------------------------------------------------------------- #


def _finalize_numeric(values: Any) -> np.ndarray:
    """Normalize a decoded numeric array to its §3 logical dtype.

    Floats (incl. CF-unpacked / mask-and-scaled values, which carry ``NaN`` for
    fill cells) become ``float64``; an unpacked pure-integer field (e.g. a raw
    ``hours since`` time axis) keeps its integer dtype. This removes the
    float32-vs-float64 ambiguity between xarray / NCDatasets / netcdf-rs.
    """
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype("float64", copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        return arr  # raw integer read kept as-is (int32/int64)
    return arr


# --------------------------------------------------------------------------- #
# NetCDF reader (xarray / netCDF4).
# --------------------------------------------------------------------------- #


def _field_from_dataarray(da: Any) -> NativeField:
    """Build a :class:`NativeField` from an ``xarray.DataArray`` in file order.

    xarray reports ``dims`` and ``values`` in the on-disk dimension order (unlike
    NCDatasets, which the Julia track has to reverse), so no permute is needed.
    Only the decode-relevant ``units``/``calendar`` attributes are carried — the
    CF packing attrs (``scale_factor``/``add_offset``/``_FillValue``) are consumed
    by ``mask_and_scale`` and intentionally dropped.
    """
    # The declared shape is authoritative, and it must be read BEFORE `.values`:
    # xarray's lazy backend indexer materialises an EMPTY orthogonal selection to
    # the wrong shape (a `{"indices": []}` window on a (2,3,3) variable yields
    # values of shape (1,0,1), and reading `.values` replaces the lazy array so
    # `da.shape` then reports that too). Left alone, such a field's `shape`
    # contradicts its own `dims` and the coordinates beside it. Reshaping to the
    # pre-read shape fixes the empty case and turns any other mismatch into a loud
    # error rather than a silently wrong array.
    shape = tuple(int(n) for n in da.shape)
    values = np.asarray(da.values)
    if values.shape != shape:
        values = values.reshape(shape)
    data = _finalize_numeric(values)
    dims = tuple(str(d) for d in da.dims)
    attrs = {k: da.attrs[k] for k in ("units", "calendar") if k in da.attrs}
    return NativeField(data, dims, attrs)


def _netcdf_engine() -> Optional[str]:
    """The xarray engine used to decode a NetCDF blob.

    The shared cache stores content-addressed blobs *without* a file extension,
    so xarray's extension-based engine auto-detection fails with "cannot guess
    the engine". Pick the first installed engine explicitly: ``netcdf4`` and
    ``h5netcdf`` read NetCDF4/HDF5 (what the CDS/ERA5 transport downloads) as well
    as classic NetCDF3; ``scipy`` reads NetCDF3 only. ``None`` ⇒ fall back to
    xarray's guess (which raises a clear error when no engine is installed)."""
    import importlib.util

    for engine, module in (("netcdf4", "netCDF4"),
                           ("h5netcdf", "h5netcdf"),
                           ("scipy", "scipy")):
        if importlib.util.find_spec(module) is not None:
            return engine
    return None


def _is_cf_time(attrs: Any) -> bool:
    """A CF time axis: ``units`` of the form ``"<step> since <reference>"``.

    The test is a whitespace-separated token equal to ``since``, compared
    case-INSENSITIVELY — the same rule the Rust track applies, because "hours
    SINCE 1900-01-01" must not be a time axis in one track and a selectable
    spatial axis in another.
    """
    units = attrs.get("units")
    return bool(units) and "since" in str(units).lower().split()


def _netcdf_is_time_dim(ds: Any, dim: str) -> bool:
    """Is ``dim`` the file's time axis?

    A same-named coordinate whose ``units`` is CF ``"<step> since <ref>"`` settles
    it; a dimension literally named ``time`` with no coordinate (the GEOS-FP
    shape) is taken at its word. Only used to REFUSE a time selection — record
    selection is the Provider's job — so erring towards "yes" costs a clear
    error, never a wrong array.
    """
    if dim in ds.variables and _is_cf_time(ds[dim].attrs):
        return True
    return str(dim).lower() == "time"


def _netcdf_dim_selection(ds: Any, select: Optional[Any]) -> Optional[Dict[str, List[int]]]:
    """``select`` → ``{dimension: ordered 0-based index list}``, or ``None``.

    The axes are positional over the file-order dims of every array whose rank
    matches the axis count (the zarr rule); the induced map is what the read
    applies BY NAME, which is what keeps the coordinates in step with the data.
    NetCDF dimension lengths are file-global, so each axis resolves once.
    """
    axes_spec = _select_axes(select)
    if axes_spec is None:
        return None
    parsed = [_parse_axis(a) for a in axes_spec]
    naxes = len(parsed)

    bydim: Dict[str, Tuple] = {}
    matched = False
    for name, da in ds.variables.items():
        if len(da.dims) != naxes:
            continue
        matched = True
        for axis, dim in zip(parsed, da.dims):
            dim = str(dim)
            if dim in bydim and bydim[dim] != axis:
                raise ValueError(
                    f"select is ambiguous: dimension {dim!r} is asked for two "
                    "different selectors by two rank-"
                    f"{naxes} variables in this blob; a netcdf `select` is "
                    "positional over file-order dims and must agree"
                )
            bydim[dim] = axis
    if not matched:
        raise ValueError(
            f"select has {naxes} axes but no variable in the blob has rank "
            f"{naxes}; a netcdf `select` is positional over the file-order dims "
            "of the arrays it applies to"
        )

    out: Dict[str, List[int]] = {}
    for dim, axis in bydim.items():
        if axis[0] == "all":
            continue
        if _netcdf_is_time_dim(ds, dim):
            raise ValueError(
                f"select asks for a subset of the time dimension {dim!r}; record "
                "selection is the Provider's (it owns the cadence: "
                "records_per_sample), not the reader's — the time axis of a "
                'netcdf `select` must be "all"'
            )
        out[dim] = _resolve_axis_indices(axis, int(ds.sizes[dim]))
    return out or None


def _netcdf_window(da: Any, isel: Optional[Dict[str, List[int]]]) -> Any:
    """Apply the dimension selection to one ``DataArray`` (identity when empty).

    ``.isel`` with an index list is xarray's orthogonal indexing: it reads only
    the intersecting chunks and returns the indices in the order given.
    """
    if not isel:
        return da
    take = {d: isel[d] for d in map(str, da.dims) if d in isel}
    return da.isel(take) if take else da


class NetCDFReader:
    """The active ``netcdf`` reader, backed by xarray (netCDF4 engine).

    CF-decodes per ``spec/conformance.md`` §3, identically to the oracle
    (``conformance/verify.py``): opens with ``decode_times=False`` and
    ``mask_and_scale=True``, so ``scale_factor``/``add_offset`` are applied in
    float64, ``_FillValue``/``missing_value`` cells become ``NaN``, and the time
    axis is returned **raw** (its stored integers) with ``units``+``calendar``
    carried in ``attrs`` for ESS. Data variables land in ``variables``;
    dimension coordinates (latitude/longitude/time) land in ``coords``.

    ``variables`` is a projection: an unrequested variable is never read.
    ``select`` is a DECODE-TIME orthogonal selection — the reader still gets the
    whole blob (one URL, one cache key, nothing about the fetch changes) but
    materialises only the requested hyperslab. Hence ``supports_selection = True``
    while ``store_backed`` stays absent/``False``, which is how a caller tells a
    decode-scoped selection from the zarr reader's fetch-scoped one. See
    :meth:`read_native` for the axis rules.
    """

    #: Registry name + format key(s) + extension sniff hints.
    NAME = "netcdf"
    FORMATS = ("netcdf",)
    EXTENSIONS = ("nc", "nc4", "cdf")
    #: Third-party stacks this reader needs (see Registry.register).
    REQUIRES = (("xarray",), ("netCDF4",))
    #: Honours a per-axis ``select`` without materialising the whole array — at
    #: DECODE time (see :func:`earthsciio.registry.supports_selection`).
    supports_selection = True

    def formats(self) -> List[str]:
        return list(self.FORMATS)

    def extensions(self) -> List[str]:
        return list(self.EXTENSIONS)

    def open(self, blob_path: Any) -> Any:
        """Return the blob path as the handle; the dataset is opened in
        :meth:`read_native` under a ``with`` block so nothing leaks."""
        return blob_path

    def read_native(
        self,
        handle: Any,
        variables: Optional[Sequence[str]] = None,
        select: Optional[Any] = None,
        **_: Any,
    ) -> NativeDataset:
        """Decode ``handle`` into a :class:`NativeDataset`.

        ``variables`` (on-disk ``file_variable`` names) restricts the returned
        **data variables**; coordinates are always kept. ``None``/empty returns
        all data variables. A requested name absent from the blob is a
        :class:`KeyError`.

        ``select`` is an orthogonal selection in the shared 0-based vocabulary
        (:mod:`earthsciio.selection`): ``{"axes": [...]}`` with each axis
        ``"all"``, ``{"indices": [...]}`` or ``{"slice": [start, stop, step?]}``.
        It is honoured at DECODE time — the blob is already fetched, and
        ``.isel`` reads only the intersecting chunks. Applied thus:

        * the axes are **positional over file-order dims** of every array whose
          rank equals the axis count (the zarr rule);
        * that induces a **dimension → selector** map, applied by NAME to every
          other array **and to the coordinates**, so a windowed variable never
          comes back beside a full-length ``lon``/``lat``. Two same-rank arrays
          disagreeing about a dimension is a :class:`ValueError`;
        * a **time** axis that is not ``"all"`` is REFUSED: record selection
          belongs to the Provider, which owns the cadence;
        * an axis count matching no array is a :class:`ValueError`, never a
          silently ignored selection;
        * every resolved index is **bounds-checked** (``0 <= i < dim_len``) for a
          ``slice`` exactly as for an ``indices`` list: an over-long or negative
          ``[start, stop)`` is an :class:`IndexError`, never a silent clamp and
          never a negative wrap-around;
        * a **dimension is never dropped** — a one-index axis comes back at
          length 1 — and an axis may legally select NOTHING (``{"indices": []}``,
          or an empty half-open ``{"slice": [1, 1]}``), giving a **zero-length
          axis** kept in ``dims`` rather than an error.
        """
        import xarray as xr  # lazy: only the netcdf path needs the heavy stack

        want = {str(v) for v in variables} if variables else None
        out_vars: Dict[str, NativeField] = {}
        out_coords: Dict[str, NativeField] = {}
        with xr.open_dataset(handle, decode_times=False, mask_and_scale=True,
                             engine=_netcdf_engine()) as ds:
            if want is not None:
                missing = [v for v in want if v not in ds.data_vars]
                if missing:
                    raise KeyError(
                        f"requested variables not in blob: {sorted(missing)}; "
                        f"present data variables: {sorted(map(str, ds.data_vars))}"
                    )
            isel = _netcdf_dim_selection(ds, select)
            for name, da in ds.data_vars.items():
                if want is not None and str(name) not in want:
                    continue
                out_vars[str(name)] = _field_from_dataarray(_netcdf_window(da, isel))
            for name, da in ds.coords.items():
                out_coords[str(name)] = _field_from_dataarray(_netcdf_window(da, isel))
        return NativeDataset(out_vars, out_coords)


# --------------------------------------------------------------------------- #
# CSV reader — a second format proving the registry seam (spec/conformance.md).
# --------------------------------------------------------------------------- #


def _parses_float(s: str) -> bool:
    try:
        float(s.strip())
        return True
    except ValueError:
        return False


class CSVReader:
    """The active ``csv`` reader — a non-NetCDF format behind the same registry.

    Columns named in ``numeric_columns`` parse to ``float64`` 1-D arrays keyed by
    the column (``file_variable``) name; every other column is returned as a
    ``list`` of ``str``. All fields carry the single dimension ``index``; there
    are no coordinates.

    ``numeric_columns`` is REQUIRED by the loader spec and is not inferred: the
    corpus ``location_id`` column is digit-only text (``"1"``/``"2"``) yet must
    stay a string, so "parses as a number" is not a safe signal. When it is
    ``None`` the reader falls back to best-effort inference (every value parses as
    a float), which the loader/``.esm`` node should override. Quoted fields with
    embedded delimiters are handled by :mod:`csv`; matches the Julia ``CSVReader``.
    """

    #: Registry name + format key(s) + extension sniff hints.
    NAME = "csv"
    FORMATS = ("csv",)
    EXTENSIONS = ("csv", "txt")

    def formats(self) -> List[str]:
        return list(self.FORMATS)

    def extensions(self) -> List[str]:
        return list(self.EXTENSIONS)

    def open(self, blob_path: Any) -> Any:
        return blob_path

    def read_native(
        self,
        handle: Any,
        variables: Optional[Sequence[str]] = None,
        select: Optional[Any] = None,
        *,
        numeric_columns: Optional[Sequence[str]] = None,
        delimiter: str = ",",
        header_row: int = 0,
        **_: Any,
    ) -> NativeDataset:
        """Decode a delimited-text blob into a :class:`NativeDataset` of points."""
        with open(handle, newline="") as fh:
            rows = [r for r in _csv.reader(fh, delimiter=delimiter) if r]
        if not rows:
            return NativeDataset()
        header = rows[header_row]
        body = rows[header_row + 1 :]
        want = {str(v) for v in variables} if variables else None
        if want is not None:
            missing = [v for v in want if v not in header]
            if missing:
                raise KeyError(
                    f"requested variables not in CSV: {sorted(missing)}; "
                    f"present columns: {header}"
                )
        numset = {str(c) for c in numeric_columns} if numeric_columns is not None else None

        out_vars: Dict[str, NativeField] = {}
        for j, col in enumerate(header):
            name = str(col)
            if want is not None and name not in want:
                continue
            vals = [r[j] for r in body]
            is_numeric = (name in numset) if numset is not None else all(map(_parses_float, vals))
            if is_numeric:
                data: Any = np.array([float(v) for v in vals], dtype="float64")
            else:
                data = [str(v) for v in vals]
            out_vars[name] = NativeField(data, ("index",), {})
        return NativeDataset(out_vars, {})


# --------------------------------------------------------------------------- #
# GeoTIFF reader — raster bands + a domain-derived lon/lat (or x/y) grid.
#
# The decode half for the ArcGIS ImageServer ``exportImage`` rasters the ESS
# loaders fetch (LANDFIRE fuel model, USGS 3DEP elevation) and any other GeoTIFF.
# Prefers GDAL via ``rasterio`` (``spec/registries.md`` §registries: "raster
# bands via GDAL"); falls back to pure-Python ``tifffile`` so a lean install
# without the GDAL stack still reads the geo-referencing tags directly. Both
# yield the SAME :class:`NativeDataset`, so the Provider/ESS see one shape.
# --------------------------------------------------------------------------- #


class _Raster:
    """A decoded raster: band arrays + cell-center axes + georef flags."""

    __slots__ = ("bands", "x_centers", "y_centers", "geographic", "nodata")

    def __init__(
        self,
        bands: List[np.ndarray],
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        geographic: bool,
        nodata: Optional[float],
    ) -> None:
        self.bands = bands
        self.x_centers = x_centers
        self.y_centers = y_centers
        self.geographic = geographic
        self.nodata = nodata


def _geokey_value(geokeys: Optional[Sequence[int]], key_id: int) -> Optional[int]:
    """Read an *inline* GeoKey from a flat ``GeoKeyDirectoryTag`` (or ``None``).

    The directory is ``[version, keyRev, minorRev, nKeys, (KeyID, loc, count,
    value) * nKeys]``; only inline keys (``loc == 0``) carry their value in the
    4th slot. Used to detect ``GTModelTypeGeoKey`` (1024): 1=projected,
    2=geographic.
    """
    if not geokeys or len(geokeys) < 4:
        return None
    g = [int(v) for v in geokeys]
    n = g[3]
    for k in range(n):
        off = 4 + 4 * k
        if off + 3 >= len(g):
            break
        if g[off] == key_id and g[off + 1] == 0:
            return g[off + 3]
    return None


def _parse_nodata(tags: Dict[str, Any]) -> Optional[float]:
    """The GDAL_NODATA sentinel (an ASCII tag), parsed to ``float`` or ``None``."""
    raw = tags.get("GDAL_NoData", tags.get("GDAL_NODATA"))
    if raw is None:
        return None
    text = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
    text = text.strip().strip("\x00").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_with_rasterio(path: Any) -> _Raster:
    """Decode via GDAL (``rasterio``): bands, cell-center xy, CRS kind, nodata."""
    import rasterio  # lazy: only this path needs the GDAL stack

    with rasterio.open(path) as ds:
        bands = [np.asarray(ds.read(i + 1)) for i in range(ds.count)]
        height, width = ds.height, ds.width
        # ds.xy(row, col) is the CELL CENTER in the dataset CRS (handles any
        # north-up/affine transform); take the first row/col to get the axes.
        xs = np.array([ds.xy(0, c)[0] for c in range(width)], dtype="float64")
        ys = np.array([ds.xy(r, 0)[1] for r in range(height)], dtype="float64")
        nodata = None if ds.nodata is None else float(ds.nodata)
        crs = ds.crs
        geographic = bool(crs.is_geographic) if crs is not None else True
    return _Raster(bands, xs, ys, geographic, nodata)


def _read_with_tifffile(path: Any) -> _Raster:
    """Decode via pure-Python ``tifffile``, parsing the GeoTIFF georef tags.

    Reads ``ModelPixelScaleTag`` (cell size) + ``ModelTiepointTag`` (a raster→
    model anchor) to build north-up cell-center axes, ``GeoKeyDirectoryTag`` for
    the geographic/projected flag, and ``GDAL_NODATA`` for the fill sentinel.
    """
    import tifffile  # lazy

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        arr = np.asarray(page.asarray())
        spp = int(getattr(page, "samplesperpixel", 1) or 1)
        if arr.ndim == 2:
            bands = [arr]
        elif arr.ndim == 3:
            # contiguous (H, W, S) vs planar (S, H, W); pick the axis of length spp.
            if arr.shape[-1] == spp:
                bands = [arr[..., i] for i in range(arr.shape[-1])]
            elif arr.shape[0] == spp:
                bands = [arr[i] for i in range(arr.shape[0])]
            else:
                bands = [arr[..., i] for i in range(arr.shape[-1])]
        else:
            raise ValueError(f"unsupported GeoTIFF array ndim={arr.ndim}")
        tags = {tg.name: tg.value for tg in page.tags.values()}
        scale = tags.get("ModelPixelScaleTag")
        tie = tags.get("ModelTiepointTag")
        if scale is None or tie is None:
            raise ValueError(
                "GeoTIFF lacks ModelPixelScaleTag/ModelTiepointTag; cannot derive "
                "a grid (install rasterio for non-tiepoint georeferencing)."
            )
        sx, sy = float(scale[0]), float(scale[1])
        i0, j0 = float(tie[0]), float(tie[1])
        x0, y0 = float(tie[3]), float(tie[4])
        height, width = bands[0].shape
        # GeoTIFF model space is y-up; raster rows increase downward (north-up).
        xs = x0 + (np.arange(width, dtype="float64") - i0 + 0.5) * sx
        ys = y0 - (np.arange(height, dtype="float64") - j0 + 0.5) * sy
        geographic = _geokey_value(tags.get("GeoKeyDirectoryTag"), 1024) != 1
        nodata = _parse_nodata(tags)
    return _Raster(bands, xs, ys, geographic, nodata)


def _open_raster(path: Any) -> _Raster:
    """Decode a GeoTIFF, preferring GDAL/``rasterio`` then ``tifffile``."""
    try:
        import rasterio  # noqa: F401
    except Exception:
        rasterio = None  # type: ignore[assignment]
    if rasterio is not None:
        return _read_with_rasterio(path)
    try:
        import tifffile  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only with no backend
        raise ImportError(
            "the geotiff reader needs a raster backend: install rasterio "
            "(GDAL) or tifffile — e.g. `pip install earthsciio[geotiff]`."
        ) from exc
    return _read_with_tifffile(path)


class GeoTIFFReader:
    """The active ``geotiff`` reader — raster bands on a native lon/lat grid.

    Decodes a GeoTIFF blob into a :class:`NativeDataset`: one data variable per
    raster band keyed ``Band1``..``BandN`` (1-based, the GDAL convention; the
    LANDFIRE loader's ``file_variable: "Band1"`` matches), plus the cell-center
    coordinate fields. Geographic rasters (the ArcGIS ImageServer ``imageSR=4326``
    responses) get ``lon``/``lat`` axes; projected rasters get ``x``/``y``. Band
    arrays are ``float64`` with the ``GDAL_NODATA`` sentinel mapped to ``NaN``
    (``spec/conformance.md`` §3). Reader-only: no variable-name remap, no unit
    conversion, no reprojection — those stay in ESS/ESD.

    ``reader_kwargs``: pass ``band_names=[...]`` to rename the bands positionally
    (e.g. a single-band elevation raster → ``["elevation"]``).
    """

    #: Registry name + format key(s) + extension sniff hints.
    NAME = "geotiff"
    FORMATS = ("geotiff",)
    EXTENSIONS = ("tif", "tiff")
    #: rasterio (GDAL) is preferred; tifffile is the pure-Python fallback.
    REQUIRES = (("rasterio", "tifffile"),)

    def formats(self) -> List[str]:
        return list(self.FORMATS)

    def extensions(self) -> List[str]:
        return list(self.EXTENSIONS)

    def open(self, blob_path: Any) -> Any:
        return blob_path

    def read_native(
        self,
        handle: Any,
        variables: Optional[Sequence[str]] = None,
        select: Optional[Any] = None,
        *,
        band_names: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> NativeDataset:
        """Decode ``handle`` into a :class:`NativeDataset` of raster bands + grid.

        ``variables`` (band names) restricts the returned data variables; ``None``
        returns all. A requested-but-absent band name is a :class:`KeyError`.
        ``select`` is accepted for interface parity (the Provider owns slicing).
        """
        raster = _open_raster(handle)
        nbands = len(raster.bands)
        if band_names is not None:
            names = [str(n) for n in band_names]
            if len(names) != nbands:
                raise ValueError(
                    f"band_names has {len(names)} entries but the GeoTIFF has "
                    f"{nbands} band(s)"
                )
        else:
            names = [f"Band{i + 1}" for i in range(nbands)]

        ydim, xdim = ("lat", "lon") if raster.geographic else ("y", "x")
        want = {str(v) for v in variables} if variables else None
        if want is not None:
            missing = [v for v in want if v not in names]
            if missing:
                raise KeyError(
                    f"requested bands not in GeoTIFF: {sorted(missing)}; "
                    f"present bands: {names}"
                )

        out_vars: Dict[str, NativeField] = {}
        for name, band in zip(names, raster.bands):
            if want is not None and name not in want:
                continue
            data = np.asarray(band).astype("float64", copy=True)
            if raster.nodata is not None and not np.isnan(raster.nodata):
                data[data == raster.nodata] = np.nan
            out_vars[name] = NativeField(data, (ydim, xdim), {})
        out_coords: Dict[str, NativeField] = {
            xdim: NativeField(np.asarray(raster.x_centers, dtype="float64"), (xdim,), {}),
            ydim: NativeField(np.asarray(raster.y_centers, dtype="float64"), (ydim,), {}),
        }
        return NativeDataset(out_vars, out_coords)


# --------------------------------------------------------------------------- #
# FF10 point reader — the RAW long-format FF10 point table (SMOKE/Emissions.jl).
#
# A NEW reader: the CSVReader skips only empty lines (not ``#`` comments) and does
# not apply a fixed positional schema, so it cannot read FF10. The 77 column names
# are copied from Emissions.jl ``src/ff10.jl`` ``FF10_POINT_COLUMNS``, with the
# SMOKE FF10_POINT spec names COUNTRY_CD/REGION_CD for the first two (Emissions.jl:
# COUNTRY/FIPS — identical values, a positional alias).
# --------------------------------------------------------------------------- #

FF10_POINT_COLUMNS = [
    "COUNTRY_CD", "REGION_CD", "TRIBAL_CODE", "FACILITY_ID",
    "UNIT_ID", "REL_POINT_ID", "PROCESS_ID", "AGY_FACILITY_ID",
    "AGY_UNIT_ID", "AGY_REL_POINT_ID", "AGY_PROCESS_ID", "SCC",
    "POLID", "ANN_VALUE", "ANN_PCT_RED", "FACILITY_NAME",
    "ERPTYPE", "STKHGT", "STKDIAM", "STKTEMP",
    "STKFLOW", "STKVEL", "NAICS", "LONGITUDE",
    "LATITUDE", "LL_DATUM", "HORIZ_COLL_MTHD", "DESIGN_CAPACITY",
    "DESIGN_CAPACITY_UNITS", "REG_CODES", "FAC_SOURCE_TYPE", "UNIT_TYPE_CODE",
    "CONTROL_IDS", "CONTROL_MEASURES", "CURRENT_COST", "CUMULATIVE_COST",
    "PROJECTION_FACTOR", "SUBMITTER_FAC_ID", "CALC_METHOD", "DATA_SET_ID",
    "FACIL_CATEGORY_CODE", "ORIS_FACILITY_CODE", "ORIS_BOILER_ID", "IPM_YN",
    "CALC_YEAR", "DATE_UPDATED", "FUG_HEIGHT", "FUG_WIDTH_XDIM",
    "FUG_LENGTH_YDIM", "FUG_ANGLE", "ZIPCODE", "ANNUAL_AVG_HOURS_PER_YEAR",
    "JAN_VALUE", "FEB_VALUE", "MAR_VALUE", "APR_VALUE",
    "MAY_VALUE", "JUN_VALUE", "JUL_VALUE", "AUG_VALUE",
    "SEP_VALUE", "OCT_VALUE", "NOV_VALUE", "DEC_VALUE",
    "JAN_PCTRED", "FEB_PCTRED", "MAR_PCTRED", "APR_PCTRED",
    "MAY_PCTRED", "JUN_PCTRED", "JUL_PCTRED", "AUG_PCTRED",
    "SEP_PCTRED", "OCT_PCTRED", "NOV_PCTRED", "DEC_PCTRED",
    "COMMENT",
]

#: The 42 FF10 point columns decoded to ``float64`` (blank -> ``NaN``); every
#: other column (ids/codes/free-text/temporal tokens) stays ``str`` so leading-
#: zero codes (REGION_CD ``"01001"``, ZIPCODE ``"00000"``, SCC, POLID) and dates
#: (DATE_UPDATED/CALC_YEAR) never become floats. Overridable via ``numeric_columns``.
FF10_POINT_NUMERIC = frozenset({
    "ANN_VALUE", "ANN_PCT_RED", "STKHGT", "STKDIAM", "STKTEMP", "STKFLOW",
    "STKVEL", "LONGITUDE", "LATITUDE", "DESIGN_CAPACITY", "CURRENT_COST",
    "CUMULATIVE_COST", "PROJECTION_FACTOR", "FUG_HEIGHT", "FUG_WIDTH_XDIM",
    "FUG_LENGTH_YDIM", "FUG_ANGLE", "ANNUAL_AVG_HOURS_PER_YEAR",
    "JAN_VALUE", "FEB_VALUE", "MAR_VALUE", "APR_VALUE", "MAY_VALUE", "JUN_VALUE",
    "JUL_VALUE", "AUG_VALUE", "SEP_VALUE", "OCT_VALUE", "NOV_VALUE", "DEC_VALUE",
    "JAN_PCTRED", "FEB_PCTRED", "MAR_PCTRED", "APR_PCTRED", "MAY_PCTRED",
    "JUN_PCTRED", "JUL_PCTRED", "AUG_PCTRED", "SEP_PCTRED", "OCT_PCTRED",
    "NOV_PCTRED", "DEC_PCTRED",
})


def _ff10_select_members(
    names: Sequence[str],
    members: Optional[Sequence[str]],
    member_glob: Optional[str],
) -> List[str]:
    """Resolve ``members``/``member_glob`` against a zip's member list.

    The selection is the union of the explicit ``members`` list and the
    ``member_glob`` matches (fnmatch-style, case-sensitive, against the full
    member path), deduplicated, returned in ascending lexicographic (byte)
    order — the deterministic concatenation order. A ``members`` name absent
    from the archive, and a glob matching zero members, are errors. Selection
    considers only FILE members: directory placeholder entries (names ending in
    ``/``, e.g. the real 2016fd zip's ``…/ptegu/``) are ignored.
    """
    names = [n for n in names if not n.endswith("/")]
    selected = set()
    if members is not None:
        missing = [m for m in members if m not in names]
        if missing:
            raise KeyError(
                f"zip members {sorted(missing)!r} not found in archive; "
                f"members: {sorted(names)!r}"
            )
        selected.update(str(m) for m in members)
    if member_glob is not None:
        hits = [n for n in names if _fnmatch.fnmatchcase(n, member_glob)]
        if not hits:
            raise ValueError(
                f"member_glob {member_glob!r} matched no members in archive; "
                f"members: {sorted(names)!r}"
            )
        selected.update(hits)
    if not selected:
        raise ValueError("members/member_glob selected no zip members")
    return sorted(selected)


def _ff10_skip_header(data_lines: List[str], delimiter: str) -> List[str]:
    """Drop the asserted ``country_cd`` header line from one input's data lines.

    The first non-comment, non-empty line's first delimiter-separated field must
    equal ``country_cd`` (case-insensitive) — anything else is an error, so the
    option can never silently drop a data row.
    """
    if not data_lines:
        raise ValueError(
            "skip_header_row: no non-comment lines — the asserted header row "
            "is missing"
        )
    first_field = data_lines[0].split(delimiter, 1)[0].strip().lower()
    if first_field != "country_cd":
        raise ValueError(
            "skip_header_row: first non-comment line does not start with a "
            f"'country_cd' header field (got {first_field!r}); refusing to "
            "drop a data row"
        )
    return data_lines[1:]


class FF10Reader:
    """The active ``ff10`` reader — the RAW long-format FF10 **point** table.

    Unlike :class:`CSVReader` (which only skips empty lines and does not apply a
    fixed positional schema), this reader (a) skips the leading ``#`` comment
    header block (``#FORMAT=…``, ``#COUNTRY``, …), (b) applies the fixed 77-column
    :data:`FF10_POINT_COLUMNS` schema positionally — FF10 data rows carry no clean
    header row, so the names come from the schema constant exactly as Emissions.jl
    supplies them — and (c) does RFC-4180 quote handling (via :mod:`csv`) so a
    free-text ``FACILITY_NAME`` may embed the delimiter.

    Each of the 77 columns becomes one :class:`NativeField` on a single ``index``
    dim (one index per data row); there are no coordinates (``LONGITUDE`` /
    ``LATITUDE`` are ordinary variables — a points table has no gridded axis). The
    42 numeric columns parse to ``float64`` (blank -> ``NaN``); the other 35 stay
    ``str`` (blank -> ``""``).

    READER-ONLY (Risk R3): no pollutant pivot (POLID stays a data column, rows are
    not reshaped), no unit conversion (STKHGT/STKDIAM stay feet, STKTEMP °F,
    STKFLOW ft³/s, STKVEL ft/s, ANN_VALUE tons/yr), no FIPS/SCC normalization, no
    EGU/pollutant filter — those move downstream into the ``.esm``.

    ``reader_kwargs``: ``member="path/in/zip"`` extracts a named member from a
    ``.zip`` blob (the whole zip stays the content-addressed cached blob; the
    member is reader config so it never enters the cache key). ``members`` (an
    explicit list of member names) and/or ``member_glob`` (an fnmatch-style
    pattern — ``*``, ``?``, ``[...]``, case-sensitive, matched against the full
    member path, e.g. ``*egu*``) select MULTIPLE members: the selection is the
    union of the explicit list and the glob matches, deduplicated, and the
    members are read and their rows concatenated in ascending lexicographic
    (byte) order of member name. A ``members`` name absent from the archive, and
    a ``member_glob`` matching zero members, are errors; directory placeholder
    entries (names ending in ``/``) are never selected; ``member`` (singular)
    is mutually exclusive with ``members``/``member_glob``. Like ``member``,
    none of these enter the cache key. ``skip_header_row=True`` handles the EPA
    2016fd-style column-header line: after comment (``#``) and blank lines are
    dropped, the first remaining line of each selected input (each selected zip
    member, or the bare file) must be a header row — its first
    delimiter-separated field, compared case-insensitively, must be
    ``country_cd`` — and exactly that one line is skipped per member; if the
    first field is anything else the reader errors (the option asserts a header
    row and never silently drops a data row). ``kind="point"`` selects the
    schema (only point ships). ``numeric_columns``, ``delimiter``, ``comment``
    override the defaults.
    """

    #: Registry name + format key(s) + extension sniff hints.
    NAME = "ff10"
    FORMATS = ("ff10",)
    EXTENSIONS = ("ff10", "csv")

    def formats(self) -> List[str]:
        return list(self.FORMATS)

    def extensions(self) -> List[str]:
        return list(self.EXTENSIONS)

    def open(self, blob_path: Any) -> Any:
        return blob_path

    def read_native(
        self,
        handle: Any,
        variables: Optional[Sequence[str]] = None,
        select: Optional[Any] = None,
        *,
        member: Optional[str] = None,
        members: Optional[Sequence[str]] = None,
        member_glob: Optional[str] = None,
        skip_header_row: bool = False,
        kind: str = "point",
        numeric_columns: Optional[Sequence[str]] = None,
        delimiter: str = ",",
        comment: str = "#",
        **_: Any,
    ) -> NativeDataset:
        """Decode an FF10 point blob into a ``points`` :class:`NativeDataset`."""
        if kind != "point":
            raise ValueError(
                f"FF10Reader only supports kind='point' (got {kind!r}); the 45-col "
                "nonpoint/onroad/nonroad schemas are not implemented yet"
            )
        if members is not None or member_glob is not None:
            if member is not None:
                raise ValueError(
                    "`member` is mutually exclusive with `members`/`member_glob`"
                )
            with _zipfile.ZipFile(handle) as zf:
                selected = _ff10_select_members(zf.namelist(), members, member_glob)
                texts = [zf.read(n).decode("utf-8") for n in selected]
        elif member is not None:
            with _zipfile.ZipFile(handle) as zf:
                texts = [zf.read(member).decode("utf-8")]
        else:
            with open(handle, newline="") as fh:
                texts = [fh.read()]

        # Per selected input: skip empty + '#' comment lines, drop the asserted
        # `country_cd` header line (if skip_header_row), then RFC-4180 parse.
        ncol = len(FF10_POINT_COLUMNS)
        rows: List[List[str]] = []
        for text in texts:
            data_lines = [
                ln for ln in text.splitlines()
                if ln.strip() and not ln.lstrip().startswith(comment)
            ]
            if skip_header_row:
                data_lines = _ff10_skip_header(data_lines, delimiter)
            member_rows = list(_csv.reader(data_lines, delimiter=delimiter))
            for r in member_rows:
                if len(r) != ncol:
                    raise ValueError(
                        f"FF10 point row has {len(r)} fields, expected {ncol}: {r!r}"
                    )
            rows.extend(member_rows)

        numset = (
            set(numeric_columns) if numeric_columns is not None
            else set(FF10_POINT_NUMERIC)
        )
        want = {str(v) for v in variables} if variables else None
        if want is not None:
            missing = [v for v in want if v not in FF10_POINT_COLUMNS]
            if missing:
                raise KeyError(f"requested FF10 columns not in schema: {sorted(missing)}")

        out_vars: Dict[str, NativeField] = {}
        for j, name in enumerate(FF10_POINT_COLUMNS):
            if want is not None and name not in want:
                continue
            vals = [r[j] for r in rows]
            if name in numset:
                data: Any = np.array(
                    [np.nan if v.strip() == "" else float(v) for v in vals],
                    dtype="float64",
                )
            else:
                data = [str(v) for v in vals]
            out_vars[name] = NativeField(data, ("index",), {})
        return NativeDataset(out_vars, {})


# --------------------------------------------------------------------------- #
# Shapefile reader (pyshp) — the ESRI shapefile feature table.
# --------------------------------------------------------------------------- #


#: Shape-type code -> name (ESRI Shapefile Technical Description, page 4). The
#: name is carried in the geometry field's ``attrs`` so ESS can tell a polygon
#: layer from a polyline one without re-sniffing the blob.
SHAPE_TYPE_NAMES: Dict[int, str] = {
    0: "Null", 1: "Point", 3: "PolyLine", 5: "Polygon", 8: "MultiPoint",
    11: "PointZ", 13: "PolyLineZ", 15: "PolygonZ", 18: "MultiPointZ",
    21: "PointM", 23: "PolyLineM", 25: "PolygonM", 28: "MultiPointM",
    31: "MultiPatch",
}

#: Field names the reader itself produces. A `.dbf` column of the same name is a
#: collision the reader refuses rather than silently shadowing either side.
SHAPEFILE_RESERVED = (
    "geometry", "n_vertices", "shape_index", "part_index", "n_parts",
    "xmin", "ymin", "xmax", "ymax", "shape_type", "crs_wkt",
)


def _shp_sidecar_bytes(handle: Any, member: Optional[str]) -> Dict[str, bytes]:
    """The `.shp` + sidecar byte blobs of one shapefile, keyed by extension.

    A shapefile is a **file set** but the content-addressed cache holds ONE blob,
    so the fetchable form is a `.zip`; a bare `.shp` blob decodes too, with
    geometry only (no `.dbf` attributes, no `.prj`). `member` names the `.shp`
    inside a zip; when omitted the archive must contain exactly one.
    """
    with open(handle, "rb") as fh:
        magic = fh.read(4)
    if magic[:2] != b"PK":
        with open(handle, "rb") as fh:
            return {"shp": fh.read()}
    out: Dict[str, bytes] = {}
    with _zipfile.ZipFile(handle) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        shps = sorted(n for n in names if n.lower().endswith(".shp"))
        if member is not None:
            if member not in names:
                raise KeyError(
                    f"zip member {member!r} not in the archive; .shp members present: {shps}"
                )
            target = member
        elif len(shps) == 1:
            target = shps[0]
        elif not shps:
            raise KeyError("the zip contains no .shp member")
        else:
            raise KeyError(
                f"the zip contains {len(shps)} .shp members; name one with "
                f"reader_options.member: {shps}"
            )
        out["shp"] = zf.read(target)
        stem = target[: -len(".shp")]
        lower = {n.lower(): n for n in names}
        for ext in ("dbf", "shx", "prj"):
            hit = lower.get(f"{stem.lower()}.{ext}")
            if hit is not None:
                out[ext] = zf.read(hit)
    return out


def _dbf_deletion_flags(raw: bytes) -> Tuple[bytes, List[bool]]:
    """Normalize the DBF deletion flags and report which rows are deleted.

    A dBASE record's first byte marks deletion. GDAL — and the shapefiles the
    world actually holds — use ``*`` (0x2A) for a deleted row and a space for a
    live one, but writers exist (InMAP's Go writer among them) that leave the
    byte ``NUL``. pyshp treats **any** non-space flag as deleted and drops the
    row, while the Julia (DBFTables) and Rust (``dbase``) libraries both use the
    ``*``-only rule. Rewriting every other flag byte to a space pins that ONE
    cross-language rule (``spec/conformance.md`` §3): *a row is deleted iff its
    flag byte is* ``*``. The returned mask is what realigns the `.shp` shapes
    with the LIVE `.dbf` rows, since pyshp's record iterator skips deleted rows
    silently. A malformed/short header is returned untouched with an empty mask
    — the dbf library is left to raise its own error.
    """
    if len(raw) < 32:
        return raw, []
    nrec = int.from_bytes(raw[4:8], "little")
    hdr = int.from_bytes(raw[8:10], "little")
    rec = int.from_bytes(raw[10:12], "little")
    if rec < 1 or hdr < 32 or hdr + nrec * rec > len(raw):
        return raw, []
    buf = bytearray(raw)
    deleted: List[bool] = []
    for i in range(nrec):
        off = hdr + i * rec
        deleted.append(buf[off] == 0x2A)
        if not deleted[-1]:
            buf[off] = 0x20
    return bytes(buf), deleted


def _shp_parts(shape: Any) -> List[List[Any]]:
    """One shape's vertex rings/parts, in file order, as lists of ``(x, y)``.

    A Polygon/PolyLine carries explicit part offsets; a Point or MultiPoint has
    none and is one part (a MultiPoint's points stay together). A Null shape is
    one EMPTY part, so a null row still occupies its slot in the record axis.
    """
    pts = list(getattr(shape, "points", ()) or ())
    if int(getattr(shape, "shapeType", 0)) == 0:
        return [[]]
    offs = [int(p) for p in (getattr(shape, "parts", ()) or ())]
    if not offs:
        return [pts]
    bounds = offs + [len(pts)]
    return [pts[bounds[k]:bounds[k + 1]] for k in range(len(offs))]


def _shp_bbox(shape: Any) -> Tuple[float, float, float, float]:
    """A shape's bounding box: the record's own stored ``Box`` where the format
    has one, else (Point) the point itself. ``NaN``s for a Null shape."""
    try:
        bb = shape.bbox
    except Exception:
        bb = None
    if bb is None or len(bb) < 4:
        pts = list(getattr(shape, "points", ()) or ())
        if not pts:
            return (np.nan, np.nan, np.nan, np.nan)
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    return (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))


class ShapefileReader:
    """The active ``shapefile`` reader — an ESRI shapefile as a feature table.

    Decode is delegated to **pyshp** (:mod:`shapefile`), imported lazily so a
    base install stays lean, mirroring the Julia ``Shapefile.jl`` extension and
    the Rust ``shapefile`` crate. Nothing about the format is re-implemented
    here; what this reader owns is the mapping onto the NATIVE-ARRAY contract.

    **One row per PART.** A shapefile record may carry several parts — a
    polygon's outer ring plus its holes, a county's mainland plus its islands, a
    multi-part route. The op that consumes this geometry
    (``polygon_intersection_area``, ``intersect_polygon``) takes ONE ring, so a
    reader that surfaced only the first part would silently drop the islands.
    Each part therefore becomes one row of the ``index`` axis, with the record's
    `.dbf` attributes REPLICATED across its parts and ``shape_index`` /
    ``part_index`` / ``n_parts`` naming where the row came from. A layer whose
    records are all single-part (the common case) decodes 1:1.

    Variables:

    ``geometry``
        ``float64[index, vertex, xy]`` — the vertex rings, right-PADDED to the
        longest part by REPEATING the final vertex. That padding is the
        rectangular-vertex-storage convention `esm-spec` §8.6.1 pins: a binding
        evaluates such a ring as its deduplicated form, so a short ring padded
        this way has the same area as the ring itself. A Null shape's row is all
        ``NaN``. Vertices are kept exactly as stored — the explicit closing
        vertex of a shapefile ring is NOT dropped (§8.6.1 makes dropping it the
        kernel's job) and no winding is normalized.
    ``shape_type``, ``crs_wkt``
        ``str[meta]`` — one-element string fields: the layer's shape type, and
        the `.prj` WKT verbatim when the archive carries one (``crs_wkt`` is
        absent otherwise). The native CRS is DECLARED, not acted on —
        **reprojection is ESD's job, never the reader's**. They are fields rather
        than field ``attrs`` because the Rust ``NativeField`` has no ``attrs``,
        and the cross-language native-array equality check compares fields.
    ``n_vertices``
        ``int64[index]`` — real vertices in the part, before padding.
    ``shape_index``, ``part_index``, ``n_parts``
        ``int64[index]`` — the 0-based record index in the `.shp`, the 0-based
        part index inside it, and its part count.
    ``xmin``, ``ymin``, ``xmax``, ``ymax``
        ``float64[index]`` — the parent RECORD's stored bounding box (a Point's
        own coordinate where the format stores no box), replicated to its parts.
        On disk, not computed: it is the shapefile's own broad-phase envelope.
    one field per `.dbf` column
        ``[index]``, replicated to parts. Numeric (``N``/``F``) columns become
        ``float64`` with a blank as ``NaN``; ``L`` becomes ``bool``; ``D`` and
        ``C`` stay ``str``. A row whose deletion flag is ``*`` is dropped, and
        no other flag byte means deleted (``spec/conformance.md`` §3).

    READER-ONLY (Risk R3): no reprojection, no unit conversion, no ring
    orientation fix, no polygon/hole classification, no name remap.

    ``reader_kwargs``: ``member="path/in/zip"`` names the `.shp` inside a zip
    blob (sidecars are the same stem with `.dbf`/`.shx`/`.prj`); it never enters
    the cache key. ``numeric_columns=[...]`` parses the named ``C`` columns as
    ``float64`` — the ``CSVReader``/``FF10Reader`` spelling, for a text-typed
    code column (a FIPS ``GEOID``) a model wants as a number.
    """

    #: Registry name + format key(s) + extension sniff hints.
    NAME = "shapefile"
    FORMATS = ("shapefile",)
    EXTENSIONS = ("shp", "zip")
    #: pyshp, whose import name is `shapefile`.
    REQUIRES = (("shapefile",),)

    def formats(self) -> List[str]:
        return list(self.FORMATS)

    def extensions(self) -> List[str]:
        return list(self.EXTENSIONS)

    def open(self, blob_path: Any) -> Any:
        return blob_path

    def read_native(
        self,
        handle: Any,
        variables: Optional[Sequence[str]] = None,
        select: Optional[Any] = None,
        *,
        member: Optional[str] = None,
        numeric_columns: Optional[Sequence[str]] = None,
        nvert_max: Optional[int] = None,
        **_: Any,
    ) -> NativeDataset:
        """Decode a shapefile blob into a one-row-per-part feature table."""
        try:
            import shapefile as _pyshp  # lazy: only this path needs pyshp
        except ImportError as exc:  # pragma: no cover - exercised with no backend
            raise ImportError(
                "the shapefile reader needs the pyshp backend: install it — "
                "e.g. `pip install earthsciio[shapefile]`."
            ) from exc

        blobs = _shp_sidecar_bytes(handle, member)
        kw: Dict[str, Any] = {"shp": _io.BytesIO(blobs["shp"])}
        if "shx" in blobs:
            kw["shx"] = _io.BytesIO(blobs["shx"])
        deleted: List[bool] = []
        if "dbf" in blobs:
            normalized, deleted = _dbf_deletion_flags(blobs["dbf"])
            kw["dbf"] = _io.BytesIO(normalized)

        columns: List[str] = []
        types: Dict[str, Any] = {}
        records: List[List[Any]] = []
        with _pyshp.Reader(**kw) as rdr:
            shapes = list(rdr.iterShapes())
            shape_type = int(rdr.shapeType)
            if "dbf" in blobs:
                fields = [f for f in rdr.fields if f[0] != "DeletionFlag"]
                columns = [f[0] for f in fields]
                types = {f[0]: f[1] for f in fields}
                records = [list(r) for r in rdr.iterRecords()]
        if "dbf" in blobs:
            if len(deleted) != len(shapes):
                raise ValueError(
                    f"shapefile has {len(shapes)} shapes but {len(deleted)} .dbf rows"
                )
            live = len(shapes) - sum(deleted)
            if len(records) != live:
                raise ValueError(
                    f"shapefile has {live} live shapes but {len(records)} live .dbf rows"
                )
        else:
            deleted = [False] * len(shapes)

        clash = sorted(set(columns) & set(SHAPEFILE_RESERVED))
        if clash:
            raise ValueError(
                f".dbf column name(s) {clash} collide with the reader's own fields "
                f"{list(SHAPEFILE_RESERVED)}"
            )

        # Explode to one row per part, dropping `*`-deleted rows whole.
        rings: List[List[Any]] = []
        shape_ix: List[int] = []
        part_ix: List[int] = []
        nparts: List[int] = []
        boxes: List[Tuple[float, float, float, float]] = []
        rows: List[Optional[List[Any]]] = []
        live_ix = 0
        for si, shp in enumerate(shapes):
            if deleted[si]:
                continue
            parts = _shp_parts(shp)
            box = _shp_bbox(shp)
            for pi, ring in enumerate(parts):
                rings.append(ring)
                shape_ix.append(si)
                part_ix.append(pi)
                nparts.append(len(parts))
                boxes.append(box)
                rows.append(records[live_ix] if records else None)
            live_ix += 1

        n = len(rings)
        nvert = max((len(r) for r in rings), default=0)
        if nvert_max is not None:
            if nvert > int(nvert_max):
                worst = max(range(len(rings)), key=lambda i: len(rings[i]))
                raise ValueError(
                    f"declared nvert_max={int(nvert_max)} but row {worst} (shape "
                    f"{shape_ix[worst]}, part {part_ix[worst]}) has {nvert} vertices"
                )
            nvert = int(nvert_max)
        geom = np.full((n, max(nvert, 1), 2), np.nan, dtype="float64")
        for i, ring in enumerate(rings):
            for v, pt in enumerate(ring):
                geom[i, v, 0] = float(pt[0])
                geom[i, v, 1] = float(pt[1])
            if ring:  # right-pad by repeating the final vertex (esm-spec §8.6.1)
                geom[i, len(ring):, 0] = float(ring[-1][0])
                geom[i, len(ring):, 1] = float(ring[-1][1])

        out: Dict[str, NativeField] = {
            "geometry": NativeField(geom, ("index", "vertex", "xy"), {}),
            "shape_type": NativeField(
                [SHAPE_TYPE_NAMES.get(shape_type, str(shape_type))], ("meta",), {}),
            "n_vertices": NativeField(
                np.array([len(r) for r in rings], dtype="int64"), ("index",), {}),
            "shape_index": NativeField(np.array(shape_ix, dtype="int64"), ("index",), {}),
            "part_index": NativeField(np.array(part_ix, dtype="int64"), ("index",), {}),
            "n_parts": NativeField(np.array(nparts, dtype="int64"), ("index",), {}),
        }
        if "prj" in blobs:
            out["crs_wkt"] = NativeField(
                [blobs["prj"].decode("utf-8", "replace").strip()], ("meta",), {})
        for k, name in enumerate(("xmin", "ymin", "xmax", "ymax")):
            out[name] = NativeField(
                np.array([b[k] for b in boxes], dtype="float64"), ("index",), {})

        numset = {str(c) for c in numeric_columns} if numeric_columns else set()
        unknown = sorted(numset - set(columns))
        if unknown:
            raise KeyError(f"numeric_columns names no such .dbf column: {unknown}")
        for j, name in enumerate(columns):
            vals = [None if r is None else r[j] for r in rows]
            kind = str(types[name])
            if name in numset or kind in ("N", "F", "O", "I", "+"):
                out[name] = NativeField(
                    np.array([_shp_float(v) for v in vals], dtype="float64"),
                    ("index",), {})
            elif kind == "L":
                out[name] = NativeField(
                    np.array([bool(v) for v in vals], dtype="bool"), ("index",), {})
            else:
                out[name] = NativeField([_shp_text(v) for v in vals], ("index",), {})

        want = {str(v) for v in variables} if variables else None
        if want is not None:
            missing = sorted(w for w in want if w not in out)
            if missing:
                raise KeyError(
                    f"requested variables not in the shapefile: {missing}; "
                    f"present: {sorted(out)}"
                )
            out = {k: v for k, v in out.items() if k in want}
        return NativeDataset(out, {})


def _shp_float(value: Any) -> float:
    """A `.dbf` cell as ``float64``: blank / missing / unparseable → ``NaN``."""
    if value is None:
        return np.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _shp_text(value: Any) -> str:
    """A `.dbf` cell as ``str``: missing → ``""``; a ``D`` date → ``YYYYMMDD``."""
    if value is None:
        return ""
    if isinstance(value, _dt.date):
        return f"{value.year:04d}{value.month:02d}{value.day:02d}"
    return str(value).strip() if isinstance(value, str) else str(value)


# --------------------------------------------------------------------------- #
# Parquet reader (pyarrow) — `spec/conformance.md` §3, "Parquet decode notes".
# --------------------------------------------------------------------------- #

#: Working context for the decimal → unscaled-integer shift below. Wide enough
#: for a `Decimal256` (76 digits), so the shift never rounds.
_PQ_DECIMAL_CTX = _decimal.Context(prec=100)


def _pq_target_dtype(typ: Any, forced_float: bool) -> Optional[str]:
    """The §3 native dtype an Arrow column maps to, or ``None`` when it has none.

    A total function of the Arrow type — which is what makes cross-language
    parity cheap here, and why a track that deviates is *wrong* rather than
    merely different. pyarrow is the same Apache Arrow project as the Rust
    ``parquet`` crate, so this is literally the same enum as
    ``rust/src/format/parquet.rs``'s ``target_dtype``.

    ``forced_float`` is the column being named in ``float_columns``: a statement
    about the SOURCE ("this column is a float64 measurement"), so it applies to
    anything with a rank-1 numeric reading, decimal *text* included. A column
    with no rank-1 reading at all (list/struct/map/union/binary/interval) has
    none whatever the option says.
    """
    import pyarrow.types as pt

    if pt.is_dictionary(typ):
        # A categorical reads as its VALUE type, expanded to one value per row;
        # the key encoding is storage and never reaches the native array.
        return _pq_target_dtype(typ.value_type, forced_float)
    if pt.is_boolean(typ):
        return "float64" if forced_float else "bool"
    # The narrow/wide integer split is the NetCDF reader's, VERBATIM — restated
    # so a MOVES `int32` ID column and a CF `int32` time axis cannot drift apart.
    if (
        pt.is_int8(typ) or pt.is_int16(typ) or pt.is_int32(typ)
        or pt.is_uint8(typ) or pt.is_uint16(typ)
    ):
        return "float64" if forced_float else "int32"
    if pt.is_int64(typ) or pt.is_uint32(typ) or pt.is_uint64(typ):
        return "float64" if forced_float else "int64"
    if pt.is_floating(typ) or pt.is_decimal128(typ) or pt.is_decimal256(typ):
        return "float64"
    # An all-null column has no type of its own; float64 is the one logical type
    # that can represent every cell of it (as NaN).
    if pt.is_null(typ):
        return "float64"
    if pt.is_string(typ) or pt.is_large_string(typ) or pt.is_string_view(typ):
        return "float64" if forced_float else "string"
    # Temporal columns ride as their RAW stored integer (see `_pq_cells`).
    if pt.is_date32(typ) or pt.is_time32(typ):
        return "float64" if forced_float else "int32"
    if (
        pt.is_date64(typ) or pt.is_time64(typ)
        or pt.is_timestamp(typ) or pt.is_duration(typ)
    ):
        return "float64" if forced_float else "int64"
    return None


def _pq_cells(col: Any) -> Tuple[List[Any], Any]:
    """A column's cells as plain Python values (``None`` for a null) + its type.

    The returned type is the DICTIONARY-DECODED one, so the caller sees the
    value type a categorical really carries.

    Two decodes happen here rather than in the caller:

    * a ``Dictionary(K, V)`` is expanded to one ``V`` per row — a null key and a
      key pointing at a null value are both nulls, which ``cast`` already folds;
    * a temporal column is cast to its **raw stored integer**. The Arrow unit
      (``s``/``ms``/``us``/``ns``) and any timezone are NOT applied and NOT
      reported, the same rule a CF time axis gets: turning an epoch offset into
      a wall-clock instant is ESS's job (Risk R3). Casting is what keeps the raw
      value — ``to_pylist`` alone would hand back ``datetime`` objects.

    ``to_pylist`` is used rather than ``to_pandas``/``to_numpy`` on purpose:
    pandas promotes a nullable int64 column to ``float64``, which is exactly the
    silent substitution the null policy below exists to forbid.
    """
    import pyarrow as pa
    import pyarrow.types as pt

    typ = col.type
    if pt.is_dictionary(typ):
        col = col.cast(typ.value_type)
        typ = col.type
    if pt.is_date32(typ) or pt.is_time32(typ):
        return col.cast(pa.int32()).to_pylist(), typ
    if (
        pt.is_date64(typ) or pt.is_time64(typ)
        or pt.is_timestamp(typ) or pt.is_duration(typ)
    ):
        return col.cast(pa.int64()).to_pylist(), typ
    return col.to_pylist(), typ


def _pq_null_error(name: str, row: int, kind: str, option: str) -> ValueError:
    """The refusal a null in a type with no missing value gets.

    It names the way out, because "declare a sentinel" is a decision only the
    document can make.
    """
    return ValueError(
        f"parquet column {name!r} row {row} is null, and a {kind} native field has no "
        f"missing value; declare the `{option}` reader option to substitute one, or "
        f"list the column in `float_columns` to read it as float64 with NaN"
    )


def _pq_float_text(name: str, row: int, text: str) -> float:
    """A decimal-TEXT cell as ``float64`` under ``float_columns``.

    Trimmed and parsed; blank → ``NaN`` (the FF10/shapefile rule); anything else
    unparseable is an error naming the column, the row and the text. The
    underscore guard keeps Python's numeric-literal spelling (``"1_0"`` parses as
    10.0) from accepting text no other track would.
    """
    t = text.strip()
    if not t:
        return np.nan
    if "_" not in t:
        try:
            return float(t)
        except ValueError:
            pass
    raise ValueError(
        f"parquet column {name!r} row {row}: {text!r} is not a float64 "
        f"(the column is declared in `float_columns`)"
    )


def _pq_field(
    name: str,
    col: Any,
    dtype: str,
    null_int: Optional[int],
    null_string: Optional[str],
) -> NativeField:
    """One Parquet column as a rank-1 :class:`NativeField` over ``index``.

    Applies the §3 null policy: a null in a **float** column is ``NaN`` (the same
    fold a CF ``_FillValue`` gets); a null in an **integer**, **string** or
    **boolean** column is an error naming the column and the row, because those
    types have no NaN and any default would be a real value silently standing in
    for a missing one. ``null_int`` / ``null_string`` open that gate only when a
    document declares them, and ``null_int`` is reported back in
    ``attrs["fill_value"]`` (an integer sentinel cannot be NaN, so it survives
    into the array exactly as a CF integer fill does).
    """
    import pyarrow.types as pt

    cells, typ = _pq_cells(col)

    if dtype == "float64":
        if pt.is_decimal128(typ) or pt.is_decimal256(typ):
            # Unscaled integer ÷ 10^scale, in double — the arrow-rs reader's own
            # arithmetic, so the two tracks round identically.
            scale = typ.scale
            div = 10.0 ** scale
            values = [
                np.nan if c is None else float(int(c.scaleb(scale, _PQ_DECIMAL_CTX))) / div
                for c in cells
            ]
        elif pt.is_string(typ) or pt.is_large_string(typ) or pt.is_string_view(typ):
            values = [
                np.nan if c is None else _pq_float_text(name, i, c)
                for i, c in enumerate(cells)
            ]
        elif pt.is_boolean(typ):
            for i, c in enumerate(cells):
                if c is not None:
                    raise ValueError(
                        f"parquet column {name!r} row {i}: a boolean cell cannot be "
                        f"read as float64"
                    )
            values = [np.nan] * len(cells)
        else:
            values = [np.nan if c is None else float(c) for c in cells]
        return NativeField(np.array(values, dtype="float64"), ("index",), {})

    if dtype in ("int32", "int64"):
        lo, hi = (-(2 ** 31), 2 ** 31 - 1) if dtype == "int32" else (-(2 ** 63), 2 ** 63 - 1)
        values = []
        for i, c in enumerate(cells):
            if c is None:
                if null_int is None:
                    raise _pq_null_error(name, i, "integer", "null_int")
                v = int(null_int)
            else:
                v = int(c)
            if not lo <= v <= hi:
                # The one integer width that does not fit: refuse rather than
                # wrap a uint64 into a negative ID.
                raise ValueError(
                    f"parquet column {name!r} row {i}: value {v} does not fit the "
                    f"{dtype} native dtype"
                )
            values.append(v)
        attrs = {"fill_value": int(null_int)} if null_int is not None else {}
        return NativeField(np.array(values, dtype=dtype), ("index",), attrs)

    if dtype == "string":
        text: List[str] = []
        for i, c in enumerate(cells):
            if c is None:
                if null_string is None:
                    raise _pq_null_error(name, i, "string", "null_string")
                text.append(str(null_string))
            else:
                # Utf8 → str, and NOTHING else: an `SCC` or any other
                # leading-zero code must not be helpfully turned into a number.
                text.append(c)
        return NativeField(text, ("index",), {})

    for i, c in enumerate(cells):
        if c is None:
            # No sentinel option: a third boolean state is a float64 column.
            raise ValueError(
                f"parquet column {name!r} row {i} is null, and a boolean native field "
                f"has no missing value; declare the column in the `float_columns` "
                f"reader option if a third state is meant"
            )
    return NativeField(np.array(cells, dtype="bool"), ("index",), {})


class ParquetReader:
    """The active ``parquet`` reader — an Apache Parquet file as a flat table.

    Decode is delegated to **pyarrow** (:mod:`pyarrow.parquet`), imported lazily
    so a base install stays lean. pyarrow is the reference Apache Arrow/Parquet
    implementation and the peer of the Rust track's ``parquet`` crate, so the
    Arrow type surface below is literally the same enum on both sides — the
    reason cross-language parity is cheap here. Nothing about the format is
    re-implemented; what this reader owns is the mapping onto the NATIVE-ARRAY
    contract (``spec/conformance.md`` §3, "Parquet decode notes").

    **A table, not a grid.** Like the ``csv``/``ff10``/``shapefile`` readers,
    every column becomes a rank-1 field over ``index``, keyed by its **on-disk
    column name**, and the dataset carries **no coordinates**. ``index`` has
    length ``num_rows``. A zero-row file is **typed, not absent**: the schema
    lives in the footer, so every column comes back empty with its declared
    dtype — which matters because most of a MOVES fixture's ~770 tables are
    empty and a document binding one must still see the array it named.

    **Type mapping** (a total function of the Arrow type):

    ==========================================  =========
    Arrow type                                  dtype
    ==========================================  =========
    ``Boolean``                                 ``bool``
    ``Int8``/``Int16``/``Int32``/``UInt8``/``UInt16``  ``int32``
    ``Int64``/``UInt32``/``UInt64``             ``int64``
    ``Float16``/``Float32``/``Float64``         ``float64``
    ``Decimal128``/``Decimal256``               ``float64`` (unscaled ÷ 10^scale)
    ``Utf8``/``LargeUtf8``/``Utf8View``         ``string``
    ``Date32``/``Time32``                       ``int32`` (raw, **undecoded**)
    ``Date64``/``Time64``/``Timestamp``/``Duration``  ``int64`` (raw, **undecoded**)
    ``Dictionary(_, V)``                        as ``V``, expanded
    ``Null``                                    ``float64``, all ``NaN``
    ==========================================  =========

    The narrow/wide integer split is the :class:`NetCDFReader`'s, **verbatim**,
    so a MOVES ``int32`` ID column and a CF ``int32`` time axis cannot drift
    apart. A ``uint64`` above ``int64`` max is an error naming the column and
    row, never a wraparound into a negative ID. Nested and binary columns have
    no rank-1 reading: naming one in ``variables`` is an error, unrequested it is
    simply not a field (as the NetCDF reader skips its non-numeric variables).

    **Null policy.** Nearly every Parquet column is nullable in its schema
    whether or not it holds a null — a table exported from a relational database
    usually marks every column nullable — so nullability cannot pick the dtype.
    A null in a float column is ``NaN``; a null in an integer/string/boolean
    column is an error, opened only by a declared ``null_int`` / ``null_string``.

    **Column projection pushes down.** ``variables`` becomes a pyarrow column
    projection, so only those column chunks are read off disk — not
    read-then-discarded. These tables are wide (a MOVES table runs to dozens of
    columns) and a document typically wants three. Empty ``variables`` reads
    every column; a requested name absent from the file is an error listing what
    is present, never a silently missing array.

    READER-ONLY (Risk R3): no row selection, no filtering, no code mapping, no
    name remap, no unit conversion. esm-spec §8.9 puts ``codes``,
    ``record_filter``, ``select`` and ``extent`` downstream of the decode, and
    ``select`` never reaches a whole-file reader at all (the Provider hands one
    whole-file selection unconditionally).

    ``reader_kwargs``: ``float_columns=[...]`` forces the named columns to
    ``float64`` whatever their on-disk type — the Parquet twin of the
    ``ShapefileReader``'s ``numeric_columns`` — and does two jobs, because they
    are the same statement about the source: an integer column that is really a
    measurement (missing cells → ``NaN`` rather than a sentinel), **and** a
    column of fixed-decimal **text**. A corpus that needs byte-reproducible
    floats stores them as decimal strings rather than IEEE doubles — the MOVES
    snapshots write ``meanBaseRate`` as ``"261.000000000000"`` — and this is how
    a document says so. ``null_int=n`` / ``null_string="…"`` declare the null
    substitutes above. None of the three enters the cache key.
    """

    #: Registry name + format key(s) + extension sniff hints.
    NAME = "parquet"
    FORMATS = ("parquet",)
    EXTENSIONS = ("parquet", "parq", "pq")

    def formats(self) -> List[str]:
        return list(self.FORMATS)

    def extensions(self) -> List[str]:
        return list(self.EXTENSIONS)

    def open(self, blob_path: Any) -> Any:
        return blob_path

    def read_native(
        self,
        handle: Any,
        variables: Optional[Sequence[str]] = None,
        select: Optional[Any] = None,
        *,
        float_columns: Optional[Sequence[str]] = None,
        null_int: Optional[int] = None,
        null_string: Optional[str] = None,
        **_: Any,
    ) -> NativeDataset:
        """Decode a Parquet blob into a flat table of rank-1 ``index`` fields.

        ``select`` is accepted and ignored: row selection is esm-spec §8.9.2 work
        downstream of the decode, and the Provider hands a whole-file reader one
        whole-file selection unconditionally. The whole table is read.
        """
        try:
            import pyarrow.parquet as _pq  # lazy: only this path needs pyarrow
        except ImportError as exc:  # pragma: no cover - exercised with no backend
            raise ImportError(
                "the parquet reader needs the pyarrow backend: install it — "
                "e.g. `pip install earthsciio[parquet]`."
            ) from exc

        pf = _pq.ParquetFile(handle)
        schema = pf.schema_arrow
        present = list(schema.names)

        # Resolve the requested names against the file's OWN schema first, so an
        # unknown one is this reader's error listing what is present rather than
        # pyarrow's "field not found".
        wanted = [str(v) for v in variables] if variables else []
        if wanted:
            missing = sorted({w for w in wanted if w not in present})
            if missing:
                raise KeyError(
                    f"requested variables not in the parquet file: {missing}; "
                    f"present: {present}"
                )
        want = set(wanted)
        forced = {str(c) for c in float_columns} if float_columns else set()

        # Decide every column's dtype from the SCHEMA, before a single data byte
        # is read: an unreadable column named in `variables` fails without
        # touching it, and an unrequested one is simply never projected. Building
        # the accumulators from the schema is also what makes a zero-row file
        # typed rather than absent.
        columns: List[str] = []
        dtypes: List[str] = []
        for field in schema:
            if want and field.name not in want:
                continue
            dtype = _pq_target_dtype(field.type, field.name in forced)
            if dtype is None:
                if field.name in want:
                    raise ValueError(
                        f"parquet column {field.name!r} has arrow type {field.type}, "
                        f"which has no rank-1 native reading (nested and binary "
                        f"columns are not supported)"
                    )
                continue
            columns.append(field.name)
            dtypes.append(dtype)

        # PROJECTION PUSHDOWN. `columns` reaches pyarrow as the read projection,
        # so only these column chunks come off disk and get decoded.
        out: Dict[str, NativeField] = {}
        if columns:
            table = pf.read(columns=columns)
            for i, (name, dtype) in enumerate(zip(columns, dtypes)):
                out[name] = _pq_field(name, table.column(i), dtype, null_int, null_string)
        return NativeDataset(out, {})


# --------------------------------------------------------------------------- #
# Registration (idempotent) — called from earthsciio/__init__.py on import.
# --------------------------------------------------------------------------- #


def register_format_readers(registry: Optional[Registry] = None) -> None:
    """Register the active ``netcdf``/``csv``/``geotiff``/``ff10``/``shapefile``/``parquet`` readers.

    Idempotent: the underlying :meth:`Registry.register` is a no-op when the same
    factory is re-registered, so importing the package twice is safe. Orthogonal
    to the ``zarr`` stub — distinct names/keys never collide.

    Registration is UNCONDITIONAL, including for readers whose decode stack is an
    optional extra: asking for ``geotiff`` without rasterio/tifffile should raise
    that reader's "install the extra" error, not
    :class:`~earthsciio.errors.BackendNotRegistered`. What each reader needs is
    declared as ``requires`` metadata instead, so a caller that must know what
    this environment can actually decode — the conformance dumpers, a diagnostic
    — asks :meth:`~earthsciio.registry.Registry.available` rather than assuming
    ``status == "active"`` means "works here".
    """
    reg = registry if registry is not None else format_registry
    reg.register(
        NetCDFReader.NAME,
        NetCDFReader,
        keys=list(NetCDFReader.FORMATS),
        status="active",
        extensions=list(NetCDFReader.EXTENSIONS),
        requires=NetCDFReader.REQUIRES,
        notes="CF-decode via xarray; scale/offset + _FillValue->NaN; time axis raw.",
    )
    reg.register(
        CSVReader.NAME,
        CSVReader,
        keys=list(CSVReader.FORMATS),
        status="active",
        extensions=list(CSVReader.EXTENSIONS),
        notes="Delimited text; numeric_columns->float64, other columns->string.",
    )
    reg.register(
        GeoTIFFReader.NAME,
        GeoTIFFReader,
        keys=list(GeoTIFFReader.FORMATS),
        status="active",
        extensions=list(GeoTIFFReader.EXTENSIONS),
        requires=GeoTIFFReader.REQUIRES,
        notes="Raster bands via GDAL/rasterio (tifffile fallback); GDAL_NODATA->NaN.",
    )
    reg.register(
        ShapefileReader.NAME,
        ShapefileReader,
        keys=list(ShapefileReader.FORMATS),
        status="active",
        extensions=list(ShapefileReader.EXTENSIONS),
        requires=ShapefileReader.REQUIRES,
        notes="ESRI shapefile via pyshp; one row per PART with .dbf attributes "
              "replicated; `geometry` [index, vertex, xy] right-padded by repeating "
              "the final vertex; the record's stored bbox as xmin/ymin/xmax/ymax; "
              ".prj WKT carried as attrs.crs_wkt; `.shp` inside a zip via `member`.",
    )
    reg.register(
        ParquetReader.NAME,
        ParquetReader,
        keys=list(ParquetReader.FORMATS),
        status="active",
        extensions=list(ParquetReader.EXTENSIONS),
        notes="Apache Parquet as a FLAT TABLE via pyarrow; every column a rank-1 field "
              "over `index`, no coords; dtype a total function of the Arrow type (the "
              "netcdf reader's narrow/wide integer split, verbatim); temporal columns "
              "carried as their RAW integer (unit/timezone unapplied); Dictionary(_,V) "
              "expanded to V; nested/binary columns are not fields (an error when named); "
              "a zero-row file is typed, not absent. Null policy: float->NaN, "
              "integer/string/boolean an error unless `null_int` (reported back as "
              "attrs.fill_value) / `null_string` is declared. `float_columns` forces "
              "float64 and parses fixed-decimal TEXT (blank->NaN). `variables` pushes "
              "down as a column projection.",
    )
    reg.register(
        FF10Reader.NAME,
        FF10Reader,
        keys=list(FF10Reader.FORMATS),
        status="active",
        extensions=list(FF10Reader.EXTENSIONS),
        notes="FF10 point long-format; '#' header skipped; fixed 77-col schema; "
              "numeric->float64 (empty->NaN), ids/codes/text->string; zip member via "
              "`member`; multi-member via `members`/`member_glob` (sorted-name concat); "
              "`skip_header_row` drops one asserted `country_cd` header line per member.",
    )
