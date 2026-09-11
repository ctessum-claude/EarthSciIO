//! `netcdf` format reader — decodes **NetCDF-3 classic** (CDF-1/CDF-2) **and**
//! **NetCDF-4 / HDF5-backed** files through the pure-Rust `netcdf-reader` crate.
//! Decode parity is `spec/conformance.md` §3.
//!
//! # Why `netcdf-reader`
//!
//! This reader once was a hand-rolled NetCDF-3-classic-only parser; NetCDF-4 /
//! HDF5 (what the Copernicus CDS API returns for ERA5) was rejected as out of
//! scope. The `netcdf-reader` fork reads **both** data models purely in Rust —
//! no system `libnetcdf`/HDF5 C dependency, same "a C compiler alone, no
//! clang/bindgen" build story as the rustls transport and the geotiff reader —
//! so the one built-in `netcdf` reader now covers the whole classic corpus plus
//! live HDF5 CDS blobs, emitting the same [`NativeDataset`] the rest of the
//! pipeline folds. (`netcdf-reader` uses `unsafe` internally for its memory map;
//! that is the dependency's code, not this crate's `#![forbid(unsafe_code)]`.)
//!
//! The CF decode contract is applied here, keyed by the **on-disk** variable
//! name (remap/regrid/`unit_conversion` stay upstream/downstream, Risk R3):
//!
//! - a **packed** variable (`scale_factor`/`add_offset`) or an on-disk **float**
//!   → `float64`, with `_FillValue`/`missing_value` folded to `NaN`;
//! - an unpacked **integer** keeps an integer logical type (`int32`/`int64`); an
//!   integer fill sentinel cannot be `NaN`, so it survives and is reported via
//!   `fill_value`;
//! - a **coordinate variable** (name == a dimension) is always returned, on the
//!   native grid, carrying its `units`/`calendar` verbatim;
//! - a **text** variable (`char`, or a NetCDF-4 `NC_STRING`) is a `string` field,
//!   under the classic character-array convention spelled out on
//!   [`stacks_last_dimension`].

use std::collections::HashSet;
use std::path::Path;

use netcdf_reader::{NcFile, NcType, NcVariable};

use crate::error::{Error, Result};

use super::{ArrayData, Coord, DType, NativeDataset, NativeField, Reader, Selection};

/// The active `netcdf` reader: pure-Rust NetCDF-3 + NetCDF-4/HDF5 decode.
#[derive(Debug, Default, Clone, Copy)]
pub struct NetcdfReader;

impl NetcdfReader {
    /// Construct the reader.
    pub fn new() -> Self {
        Self
    }
}

impl Reader for NetcdfReader {
    fn formats(&self) -> &'static [&'static str] {
        &["netcdf"]
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["nc", "nc4", "cdf"]
    }

    fn read_native(
        &self,
        blob_path: &Path,
        variables: &[String],
        select: &Selection,
    ) -> Result<NativeDataset> {
        let file = NcFile::open(blob_path).map_err(fmt_err)?;
        let sel = dim_selection(&file, select)?;
        decode(&file, variables, sel.as_ref())
    }

    /// Honours a per-axis `Selection` at DECODE time: the blob is already
    /// fetched, and only the requested hyperslab is read out of it (the
    /// `netcdf-reader` slice API reads just the intersecting chunks). NOT
    /// `store_backed` — the download is unchanged, and that pair is how a caller
    /// tells the two kinds of selection apart.
    ///
    /// A text field is the one exception to the hyperslab, and only to it: the
    /// VALUES are the same either way, but they are gathered after a whole read
    /// rather than sliced during one ([`select_text`] says why).
    fn supports_selection(&self) -> bool {
        true
    }
}

/// One dimension's selection: the ordered, 0-based global indices to return.
type DimSelection = std::collections::HashMap<String, Vec<usize>>;

/// Is `dim` the file's TIME axis?
///
/// A same-named coordinate variable whose `units` is CF `"<step> since <ref>"`
/// settles it; a dimension literally named `time` with no coordinate variable
/// (the GEOS-FP shape) is taken at its word. Only used to REFUSE a time
/// selection — record selection is the Provider's job — so erring towards "yes"
/// costs a clear error, never a wrong array.
fn is_time_dim(vars: &[NcVariable], dim: &str) -> bool {
    for v in vars {
        if v.name() == dim {
            if let Some(units) = att_text(v, "units") {
                if units
                    .split_whitespace()
                    .any(|w| w.eq_ignore_ascii_case("since"))
                {
                    return true;
                }
            }
        }
    }
    dim.eq_ignore_ascii_case("time")
}

/// Resolve a `Selection` into `{dimension: ordered 0-based indices}`.
///
/// The axes are positional over the file-order dims of every array whose rank
/// equals the axis count (the zarr rule); the induced map is what the decode
/// applies BY NAME, which is what keeps the coordinate variables in step with
/// the data they index. NetCDF dimension lengths are file-global, so each axis
/// resolves once.
///
/// "The arrays" means the FIELDS this blob decodes to, so the rank a variable
/// offers is [`field_dims`]'s, not `var.dimensions()`'s, and a variable that
/// decodes to no field at all (compound/opaque/enum/vlen) offers none. The two
/// differ for exactly one shape and it matters enormously: a `char label(n,
/// strlen)` is a rank-1 field on two on-disk dimensions, so counting its
/// dimensions would let a 2-axis `select` "match" a variable nobody can index
/// that way, bind axis 0 to `n` and axis 1 to a string LENGTH, and then apply
/// those selectors by name to every other array in the file — silently moving
/// which dimension a selector means for the numeric variables the caller was
/// actually windowing.
fn dim_selection(file: &NcFile, select: &Selection) -> Result<Option<DimSelection>> {
    let axes = match select {
        Selection::All => return Ok(None),
        Selection::Orthogonal(axes) => axes,
    };
    let vars: Vec<NcVariable> = file.variables().map_err(fmt_err)?.to_vec();

    let mut bydim: Vec<(String, &super::AxisSelect, usize)> = Vec::new();
    let mut matched = false;
    for var in &vars {
        if classify(var).is_none() {
            continue;
        }
        let (dims, shape) = field_dims(&vars, var);
        if dims.len() != axes.len() {
            continue;
        }
        matched = true;
        for ((axis, name), len) in axes.iter().zip(dims.iter()).zip(shape.iter()) {
            match bydim.iter().find(|(seen_name, _, _)| seen_name == name) {
                Some((_, seen, _)) if *seen != axis => {
                    return Err(Error::Format {
                        format: "netcdf".to_string(),
                        detail: format!(
                            "select is ambiguous: dimension '{}' is asked for two different \
                             selectors by two rank-{} variables in this blob; a netcdf \
                             `select` is positional over file-order dims and must agree",
                            name,
                            axes.len()
                        ),
                    })
                }
                Some(_) => {}
                None => bydim.push((name.clone(), axis, *len)),
            }
        }
    }
    if !matched {
        return Err(Error::Format {
            format: "netcdf".to_string(),
            detail: format!(
                "select has {} axes but no variable in the blob has rank {}; a netcdf \
                 `select` is positional over the file-order dims of the arrays it applies to",
                axes.len(),
                axes.len()
            ),
        });
    }

    let mut out = DimSelection::new();
    for (name, axis, len) in bydim {
        if matches!(axis, super::AxisSelect::All) {
            continue;
        }
        if is_time_dim(&vars, &name) {
            return Err(Error::Format {
                format: "netcdf".to_string(),
                detail: format!(
                    "select asks for a subset of the time dimension '{name}'; record \
                     selection is the Provider's (it owns the cadence: \
                     records_per_sample), not the reader's — the time axis of a netcdf \
                     `select` must be \"all\""
                ),
            });
        }
        out.insert(name, axis.resolve_in(len, "netcdf")?);
    }
    Ok(if out.is_empty() { None } else { Some(out) })
}

/// Decode an opened NetCDF file into native arrays, honoring the `variables`
/// filter (empty = all data variables; coordinate variables are always kept).
///
/// A requested name absent from the blob is an error listing what is present —
/// the rule the `parquet`/`shapefile` readers and the Python/Julia netcdf
/// readers already follow, so a typo'd `file_variable` cannot read back as a
/// silently missing array in this track alone.
///
/// `sel` (when present) is the decode-time hyperslab: every array — coordinate
/// variables included, so a windowed variable never comes back beside a
/// full-length axis — is read through it.
fn decode(
    file: &NcFile,
    variables: &[String],
    sel: Option<&DimSelection>,
) -> Result<NativeDataset> {
    let vars: Vec<NcVariable> = file.variables().map_err(fmt_err)?.to_vec();
    let want: HashSet<&str> = variables.iter().map(String::as_str).collect();

    if !want.is_empty() {
        let mut present: Vec<&str> = vars
            .iter()
            .filter(|v| !v.is_coordinate_variable())
            .map(NcVariable::name)
            .collect();
        present.sort_unstable();
        let mut missing: Vec<&str> = want
            .iter()
            .copied()
            .filter(|n| !present.contains(n))
            .collect();
        if !missing.is_empty() {
            missing.sort_unstable();
            return Err(Error::Format {
                format: "netcdf".to_string(),
                detail: format!(
                    "requested variables not in blob: {missing:?}; \
                     present data variables: {present:?}"
                ),
            });
        }
    }

    let mut out = NativeDataset::default();
    for var in &vars {
        let is_coord = var.is_coordinate_variable();
        // A coordinate variable is always returned — it is the native grid the
        // data lives on. Data variables honor the filter.
        if !is_coord && !want.is_empty() && !want.contains(var.name()) {
            continue;
        }
        // A TEXT variable — a `char` array (an ERA5 `expver`, a `char label(n)`)
        // or a NetCDF-4 `NC_STRING` — is a `string` field here, exactly as the
        // Python (xarray) and Julia (NCDatasets) tracks return one. This reader
        // used to skip it, which made the same bytes decode to a different SET of
        // fields in each track; a variable one track drops is not a permitted
        // divergence, so it is decoded, not skipped.
        //
        // What is left over is the genuinely unreadable: a compound/opaque/enum/
        // vlen variable, which has no array reading in ANY track. Skipping one the
        // document EXPLICITLY NAMED would hand back a dataset missing an array it
        // asked for — the "silently missing array" the absent-name check above
        // exists to prevent, reached by the other door, since such a variable IS
        // present and so passes that check. So it is an error naming its type;
        // unrequested, it is simply not a native field. (Verbatim the `parquet`
        // reader's rule for a column with no rank-1 reading.)
        let Some(class) = classify(var) else {
            if want.contains(var.name()) {
                return Err(Error::Format {
                    format: "netcdf".to_string(),
                    detail: format!(
                        "requested variable {:?} has netcdf type {:?}, which has no \
                         native array reading (compound/opaque/enum/vlen variables \
                         are not decoded)",
                        var.name(),
                        var.dtype()
                    ),
                });
            }
            continue;
        };
        let field = decode_field(file, &vars, var, class, sel)?;

        if is_coord {
            out.coords.insert(
                var.name().to_string(),
                Coord {
                    field,
                    units: att_text(var, "units"),
                    calendar: att_text(var, "calendar"),
                },
            );
        } else {
            out.variables.insert(var.name().to_string(), field);
        }
    }
    Ok(out)
}

/// The logical field a variable maps to under the CF decode contract.
enum FieldClass {
    /// `float64`: a packed (scale/offset) variable or an on-disk float; a fill
    /// cell becomes `NaN`.
    Float,
    /// `int32`: a narrow unpacked integer; an integer fill sentinel survives.
    Int32,
    /// `int64`: a wide unpacked integer; an integer fill sentinel survives.
    Int64,
    /// `string`: a `char` array, or a NetCDF-4 `NC_STRING`.
    Text,
}

/// Classify a variable, or `None` if it has no native array reading in ANY
/// track (a compound/opaque/enum/vlen variable).
fn classify(var: &NcVariable) -> Option<FieldClass> {
    // Text is matched FIRST. A `char` variable is never CF-packed, and a stray
    // `add_offset`/`scale_factor` attribute on one would otherwise route its
    // bytes into the float reader, which cannot read them.
    if matches!(var.dtype(), NcType::Char | NcType::String) {
        return Some(FieldClass::Text);
    }
    // Packing forces float64 regardless of the on-disk integer width.
    if var.attribute("scale_factor").is_some() || var.attribute("add_offset").is_some() {
        return Some(FieldClass::Float);
    }
    match var.dtype() {
        NcType::Float | NcType::Double => Some(FieldClass::Float),
        NcType::Byte | NcType::Short | NcType::Int | NcType::UByte | NcType::UShort => {
            Some(FieldClass::Int32)
        }
        NcType::UInt | NcType::Int64 | NcType::UInt64 => Some(FieldClass::Int64),
        _ => None,
    }
}

/// Does `var`'s LAST dimension measure a **string length** rather than count
/// elements? This is the classic-NetCDF character-array convention, and it is a
/// property of the whole file, not of the variable in isolation.
///
/// In classic NetCDF a "string" is conventionally a `char` array whose last
/// dimension is the string length: `char label(n, strlen)` is `n` strings of up
/// to `strlen` bytes, NUL-padded — **not** `n * strlen` single characters. But
/// the very same spelling `char label(n)` next to a `float value(n)` is `n`
/// one-byte strings, because there `n` is a real axis a numeric variable lives
/// on. The dimension name alone cannot tell the two apart.
///
/// So the last dimension is a string length exactly when **nothing else claims
/// it as an axis** — which is the rule xarray applies (`conventions.stackable`,
/// gating `CharacterArrayCoder`) and therefore the rule the Python track already
/// produces for these bytes:
///
/// - the dimension has no coordinate variable of its own, and
/// - every variable that uses it is a `char` variable that uses it LAST.
///
/// Two consequences, both verified against xarray in this module's tests:
/// a 1-D `char label(strlen)` on a dimension nothing else uses is a **scalar**
/// string (`dims == []`, one value); a `char label(n)` sharing `n` with a
/// numeric variable stays `n` one-character strings on `dims == ["n"]`.
fn stacks_last_dimension(vars: &[NcVariable], var: &NcVariable) -> bool {
    let var_dims = var.dimensions();
    let Some(last) = var_dims.last().map(|d| d.name.as_str()) else {
        // A scalar `char` is one one-byte string under either reading.
        return false;
    };
    // A dimension carrying a coordinate variable is an axis, never a length.
    if vars.iter().any(|v| v.name() == last) {
        return false;
    }
    vars.iter().all(|v| {
        let dims = v.dimensions();
        !dims.iter().any(|d| d.name == last)
            || (matches!(v.dtype(), NcType::Char)
                && dims.last().map(|d| d.name.as_str()) == Some(last))
    })
}

/// The string-length dimension `var` CONSUMES, if any: the on-disk dimension
/// that [`stacks_last_dimension`] folds away, so the decoded field never carries
/// it in `dims`.
///
/// This is the one place the two features in this file meet. A consumed
/// dimension is NOT an axis of the field, so it must not be selectable and must
/// not be counted when a `select`'s axes are matched positionally against a
/// variable's rank — `char label(n, strlen)` is a RANK-1 field, whatever its
/// two on-disk dimensions say. An `NC_STRING` consumes nothing: its elements are
/// already whole strings, so its last dimension stays a real axis.
fn consumed_length_dim<'a>(vars: &[NcVariable], var: &'a NcVariable) -> Option<&'a str> {
    if !matches!(var.dtype(), NcType::Char) || !stacks_last_dimension(vars, var) {
        return None;
    }
    var.dimensions().last().map(|d| d.name.as_str())
}

/// The `(dims, shape)` of the FIELD `var` decodes to — its on-disk dimensions
/// minus a consumed string length ([`consumed_length_dim`]).
///
/// Every part of this file that reasons about a variable's AXES must go through
/// here rather than through `var.dimensions()` directly, because for a `char`
/// array the two disagree and the on-disk answer is the wrong one.
fn field_dims(vars: &[NcVariable], var: &NcVariable) -> (Vec<String>, Vec<usize>) {
    let mut dims: Vec<String> = var.dimensions().iter().map(|d| d.name.clone()).collect();
    let mut shape: Vec<usize> = var.dimensions().iter().map(|d| d.size as usize).collect();
    if consumed_length_dim(vars, var).is_some() {
        dims.pop();
        shape.pop();
    }
    (dims, shape)
}

/// How to read one variable under a decode-time selection: the hyperslab to ask
/// the file for, its shape, and the per-axis positions to gather out of it.
///
/// An ordered index list that is an arithmetic progression (`all`, any `Range`,
/// a contiguous list) reads as ONE strided hyperslab and needs no gather;
/// anything else — a permuted or irregular list — reads its BOUNDING slab and
/// gathers from that. Either way the whole array is never materialised.
///
/// A dimension is never DROPPED: a one-index axis stays in `dims` at length 1,
/// and an axis that selects nothing (`Indices([])`, an empty half-open `Range`)
/// is a legal ZERO-LENGTH axis — `read_values` short-circuits it and reads
/// nothing, since the bounding-slab plan would otherwise return cells beside a
/// shape declaring `0`.
///
/// Numeric variables only: a slab is planned over the ON-DISK dimensions, which
/// are the field's axes for every class but [`FieldClass::Text`]. A `char`
/// array's selection is applied by [`select_text`] instead, over the axes the
/// decoded field actually has — including that same zero-length rule.
struct Slab {
    info: netcdf_reader::NcSliceInfo,
    slab_shape: Vec<usize>,
    take: Vec<Vec<usize>>,
    needs_gather: bool,
}

/// The (step, needs_gather) of an ordered index list, per the rule above.
fn progression(idxs: &[usize]) -> (u64, bool) {
    if idxs.len() <= 1 {
        return (1, false);
    }
    if idxs[1] <= idxs[0] {
        return (1, true); // descending or repeated: gather
    }
    let step = idxs[1] - idxs[0];
    for w in idxs.windows(2) {
        if w[1] <= w[0] || w[1] - w[0] != step {
            return (1, true);
        }
    }
    (step as u64, false)
}

/// Plan the hyperslab read for `var` under `sel`, or `None` when the selection
/// touches none of its dimensions (read it whole).
fn plan_slab(var: &NcVariable, sel: Option<&DimSelection>) -> Option<Slab> {
    let sel = sel?;
    let dims = var.dimensions();
    if !dims.iter().any(|d| sel.contains_key(&d.name)) {
        return None;
    }
    let mut selections = Vec::with_capacity(dims.len());
    let mut slab_shape = Vec::with_capacity(dims.len());
    let mut take = Vec::with_capacity(dims.len());
    let mut needs_gather = false;
    for d in dims.iter() {
        let len = d.size as usize;
        match sel.get(&d.name) {
            None => {
                selections.push(netcdf_reader::NcSliceInfoElem::Slice {
                    start: 0,
                    end: len as u64,
                    step: 1,
                });
                slab_shape.push(len);
                take.push((0..len).collect());
            }
            Some(idxs) if idxs.is_empty() => {
                // A legal zero-length axis: an empty slab, no gather positions.
                // `read_values` short-circuits on it and reads nothing at all.
                selections.push(netcdf_reader::NcSliceInfoElem::Slice {
                    start: 0,
                    end: 0,
                    step: 1,
                });
                slab_shape.push(0);
                take.push(Vec::new());
            }
            Some(idxs) => {
                let lo = *idxs.iter().min().unwrap_or(&0);
                let hi = *idxs.iter().max().unwrap_or(&0);
                let (step, gather) = progression(idxs);
                let step = if gather { 1 } else { step };
                selections.push(netcdf_reader::NcSliceInfoElem::Slice {
                    start: lo as u64,
                    end: (hi + 1) as u64,
                    step,
                });
                let count = (hi - lo) / (step as usize) + 1;
                slab_shape.push(count);
                take.push(idxs.iter().map(|g| (g - lo) / (step as usize)).collect());
                needs_gather |= gather;
            }
        }
    }
    Some(Slab {
        info: netcdf_reader::NcSliceInfo { selections },
        slab_shape,
        take,
        needs_gather,
    })
}

/// Gather `take` out of a row-major slab of `slab_shape`, preserving the
/// requested index ORDER on every axis (a reader that sorted them fails the
/// permuted corpus case).
///
/// `Clone` rather than `Copy` so the one gather serves both the numeric slabs
/// and [`select_text`]'s `String` cells — two gathers would be two chances to
/// get the row-major arithmetic subtly different in one of them.
fn gather<T: Clone>(flat: &[T], slab_shape: &[usize], take: &[Vec<usize>]) -> Vec<T> {
    let out_shape: Vec<usize> = take.iter().map(Vec::len).collect();
    let n: usize = out_shape.iter().product();
    let mut src_strides = vec![1usize; slab_shape.len()];
    for i in (0..slab_shape.len().saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * slab_shape[i + 1];
    }
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; out_shape.len()];
    for _ in 0..n {
        let mut off = 0;
        for (ax, &k) in idx.iter().enumerate() {
            off += take[ax][k] * src_strides[ax];
        }
        out.push(flat[off].clone());
        for ax in (0..idx.len()).rev() {
            idx[ax] += 1;
            if idx[ax] < out_shape[ax] {
                break;
            }
            idx[ax] = 0;
        }
    }
    out
}

/// Read one variable's values as f64, whole or through the selection's slab.
/// `unpacked` applies CF scale/offset + `_FillValue`->NaN; the raw path does not.
fn read_values(
    file: &NcFile,
    var: &NcVariable,
    slab: Option<&Slab>,
    unpacked: bool,
) -> Result<(Vec<f64>, Vec<usize>)> {
    let name = var.name();
    // An axis may legally resolve to NOTHING (`Indices([])`, or an empty half-open
    // `Range { start: 1, stop: 1 }`): a zero-length axis, KEPT in `dims`, never an
    // error and never a dropped dimension. Nothing is read — and it MUST be
    // short-circuited, because the slab planner's bounding-slab read would return
    // a non-empty buffer beside a shape declaring 0, i.e. a field whose `shape`
    // contradicts its `data`.
    if let Some(slab) = slab {
        let out_shape: Vec<usize> = slab.take.iter().map(Vec::len).collect();
        if out_shape.contains(&0) {
            return Ok((Vec::new(), out_shape));
        }
    }
    let Some(slab) = slab else {
        let arr = if unpacked {
            file.read_variable_unpacked_masked(name).map_err(fmt_err)?
        } else {
            file.read_variable_as_f64(name).map_err(fmt_err)?
        };
        let shape: Vec<usize> = var.dimensions().iter().map(|d| d.size as usize).collect();
        return Ok((arr.iter().copied().collect(), shape));
    };
    let arr = if unpacked {
        file.read_variable_slice_unpacked_masked(name, &slab.info)
            .map_err(fmt_err)?
    } else {
        file.read_variable_slice_as_f64(name, &slab.info)
            .map_err(fmt_err)?
    };
    let flat: Vec<f64> = arr.iter().copied().collect();
    let out_shape: Vec<usize> = slab.take.iter().map(Vec::len).collect();
    if slab.needs_gather {
        Ok((gather(&flat, &slab.slab_shape, &slab.take), out_shape))
    } else {
        Ok((flat, out_shape))
    }
}

/// Decode one variable's values into a [`NativeField`] under `class`, honoring
/// the decode-time selection `sel` (`None` ⇒ the whole array). `vars` is every
/// variable in the file — [`stacks_last_dimension`] needs the whole set.
///
/// `dims`/`shape` come from [`field_dims`], not from `var.dimensions()`: for a
/// `char` array whose last dimension is a string length the two differ, and a
/// field reported on its on-disk dims would claim an axis it does not have.
fn decode_field(
    file: &NcFile,
    vars: &[NcVariable],
    var: &NcVariable,
    class: FieldClass,
    sel: Option<&DimSelection>,
) -> Result<NativeField> {
    let (dims, shape) = field_dims(vars, var);

    match class {
        // Text is NOT read through a hyperslab; `sel` is applied to the decoded
        // strings instead. See [`select_text`] for why, and for what the
        // consumed string-length dimension does with a selection.
        FieldClass::Text => decode_text_field(file, vars, var, dims, shape, sel),
        FieldClass::Float => {
            // scale_factor/add_offset applied in double; _FillValue/missing_value
            // folded to NaN. Values are row-major (C order) per `shape`.
            let (data, shape) = read_values(file, var, plan_slab(var, sel).as_ref(), true)?;
            Ok(NativeField {
                dtype: DType::Float64,
                dims,
                shape,
                data: ArrayData::F64(data),
                fill_value: None, // folded into NaN
            })
        }
        FieldClass::Int32 => {
            let (raw, shape) = read_values(file, var, plan_slab(var, sel).as_ref(), false)?;
            Ok(NativeField {
                dtype: DType::Int32,
                dims,
                shape,
                data: ArrayData::I32(raw.iter().map(|&v| v as i32).collect()),
                fill_value: int_fill(var),
            })
        }
        FieldClass::Int64 => {
            let (raw, shape) = read_values(file, var, plan_slab(var, sel).as_ref(), false)?;
            Ok(NativeField {
                dtype: DType::Int64,
                dims,
                shape,
                data: ArrayData::I64(raw.iter().map(|&v| v as i64).collect()),
                fill_value: int_fill(var),
            })
        }
    }
}

/// Decode a `char`/`NC_STRING` variable into a `string` [`NativeField`].
///
/// Three shapes, all of them what the Python (xarray) track returns for the same
/// bytes — this module's tests pin the values against it:
///
/// - **`NC_STRING`** (NetCDF-4 only): already one string per element. `dims` and
///   `shape` are the variable's own.
/// - **`char` whose last dimension is a string length**
///   ([`stacks_last_dimension`]): that dimension is consumed, so `dims`/`shape`
///   lose their last entry — `char label(n, strlen)` is `n` strings, and a 1-D
///   `char label(strlen)` is a scalar string on `dims == []`. `dims`/`shape`
///   arrive already trimmed, from [`field_dims`].
/// - **`char` whose last dimension is a real axis**: one one-character string
///   per byte, `dims`/`shape` unchanged.
///
/// Trailing NUL padding is stripped from every string, and only trailing NULs —
/// a trailing SPACE is data. That is `netcdf-reader`'s `decode_char_string` and
/// it is byte-for-byte numpy's `|S` semantics, which is what makes the Python
/// track agree: a lone NUL byte decodes to the EMPTY string, as numpy's `|S1`
/// does, not to a `"\0"`.
///
/// The decode-time `sel` is applied afterwards, by [`select_text`].
fn decode_text_field(
    file: &NcFile,
    vars: &[NcVariable],
    var: &NcVariable,
    dims: Vec<String>,
    shape: Vec<usize>,
    sel: Option<&DimSelection>,
) -> Result<NativeField> {
    let name = var.name();
    // Both backends flatten a char array by its last dimension and NUL-strip each
    // group; for an `NC_STRING` each element is already its own string.
    let groups = file.read_variable_as_strings(name).map_err(fmt_err)?;

    let values = if consumed_length_dim(vars, var).is_some() {
        // The last dimension was the string length; `field_dims` already dropped
        // it, and `netcdf-reader` grouped by exactly that dimension.
        groups
    } else if matches!(var.dtype(), NcType::String) {
        groups
    } else {
        // A real axis: every byte is its own one-character string. Rebuild that
        // from the grouped read — the group is NUL-stripped only at its END, so
        // the characters it dropped are exactly the trailing cells, and each of
        // those is the empty string (numpy `|S1` of a NUL byte is `b""`).
        let group_len = if shape.len() >= 2 {
            shape[shape.len() - 1]
        } else {
            shape.iter().product::<usize>().max(1)
        };
        let mut out: Vec<String> = Vec::with_capacity(group_len * groups.len());
        for group in &groups {
            let mut n = 0usize;
            for ch in group.chars() {
                // A NUL byte is an EMPTY cell, never a `"\0"` — numpy's `|S1`
                // strips trailing NULs from each one-byte cell, so an interior
                // NUL in the run reads back as `b""`. Only trailing NULs of the
                // whole run were already dropped above; these are the rest.
                out.push(if ch == '\0' {
                    String::new()
                } else {
                    ch.to_string()
                });
                n += 1;
            }
            if n > group_len {
                // Only reachable for genuinely multi-byte text, where "one cell
                // per byte" and "one cell per character" disagree. Refuse rather
                // than hand back a differently-sized array than Python's.
                return Err(Error::Format {
                    format: "netcdf".to_string(),
                    detail: format!(
                        "char variable {name:?} is not one byte per character \
                         ({n} characters in a {group_len}-cell run); a non-ASCII \
                         char array on a shared dimension has no single native \
                         reading"
                    ),
                });
            }
            for _ in n..group_len {
                out.push(String::new());
            }
        }
        out
    };

    let expected: usize = shape.iter().product();
    if values.len() != expected {
        return Err(Error::Format {
            format: "netcdf".to_string(),
            detail: format!(
                "text variable {name:?} decoded to {} strings, but its shape \
                 {shape:?} holds {expected}",
                values.len()
            ),
        });
    }

    let (values, shape) = select_text(vars, var, &dims, shape, values, sel)?;

    Ok(NativeField {
        dtype: DType::Str,
        dims,
        shape,
        data: ArrayData::Str(values),
        fill_value: None,
    })
}

/// Apply the decode-time selection to an already-decoded text field, over the
/// axes the FIELD has.
///
/// A `select` is applied by dimension NAME to every array and every coordinate,
/// so a `char` variable is not exempt: `char label(n)` beside `float value(n)`
/// must lose the same rows `value` loses, or the two come back describing
/// different cells. What differs from the numeric path is only HOW:
///
/// - **No hyperslab.** `netcdf-reader` has no slicing twin of
///   `read_variable_as_strings`, and the NUL-stripping/numpy-`|S` semantics that
///   make this track agree with xarray byte for byte live inside that call.
///   Slicing raw `char` bytes would mean re-deriving them here, risking a wrong
///   STRING to save reading a label array; the gridded arrays the hyperslab
///   exists for are never text. So the strings are decoded whole and gathered.
/// - **A consumed string-length dimension is NOT selectable.** It is not an axis
///   of the field ([`consumed_length_dim`]), so a selection naming it is an
///   ERROR, not a no-op: honouring it would slice CHARACTERS off every string
///   (`"efgh"` handed back as `"ef"`), and ignoring it would hand back the full
///   array while the caller believes it asked for a window — the silent kind of
///   wrong this reader's rules exist to prevent. [`dim_selection`]'s rank
///   matching cannot produce such a name (that is what [`field_dims`] is for),
///   so this is the guard on the invariant, not a reachable user path.
/// - **A zero-length axis is legal**, exactly as for a numeric field: it stays in
///   `dims` at length 0 and selects no strings.
fn select_text(
    vars: &[NcVariable],
    var: &NcVariable,
    dims: &[String],
    shape: Vec<usize>,
    values: Vec<String>,
    sel: Option<&DimSelection>,
) -> Result<(Vec<String>, Vec<usize>)> {
    let Some(sel) = sel else {
        return Ok((values, shape));
    };
    if let Some(strlen) = consumed_length_dim(vars, var) {
        if sel.contains_key(strlen) {
            return Err(Error::Format {
                format: "netcdf".to_string(),
                detail: format!(
                    "select asks for a subset of '{strlen}', which is the string \
                     LENGTH of the char variable {:?}, not an axis of it; the \
                     decoded field has dims {dims:?}, and selecting along a string \
                     length would truncate every string rather than choose cells",
                    var.name()
                ),
            });
        }
    }
    if !dims.iter().any(|d| sel.contains_key(d)) {
        return Ok((values, shape));
    }
    let take: Vec<Vec<usize>> = dims
        .iter()
        .zip(shape.iter())
        .map(|(d, &len)| match sel.get(d) {
            Some(idxs) => idxs.clone(),
            None => (0..len).collect(),
        })
        .collect();
    let out_shape: Vec<usize> = take.iter().map(Vec::len).collect();
    // A legal zero-length axis reads NOTHING; `gather` would otherwise be asked
    // for a product of zero cells out of a full array, which is harmless but
    // says less about the intent than short-circuiting does.
    if out_shape.contains(&0) {
        return Ok((Vec::new(), out_shape));
    }
    Ok((gather(&values, &shape, &take), out_shape))
}

/// A surviving integer fill sentinel (`_FillValue`, else `missing_value`).
fn int_fill(var: &NcVariable) -> Option<f64> {
    att_f64(var, "_FillValue").or_else(|| att_f64(var, "missing_value"))
}

/// The first value of attribute `name`, widened to f64.
fn att_f64(var: &NcVariable, name: &str) -> Option<f64> {
    var.attribute(name).and_then(|a| a.value.as_f64())
}

/// The text of attribute `name` (CF `units`/`calendar`), if it is a string.
fn att_text(var: &NcVariable, name: &str) -> Option<String> {
    var.attribute(name).and_then(|a| a.value.as_string())
}

/// Wrap a `netcdf-reader` error as the registry's `netcdf` format error.
fn fmt_err(e: netcdf_reader::Error) -> Error {
    Error::Format {
        format: "netcdf".to_string(),
        detail: e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::AxisSelect;
    use std::io::Write;

    /// Decode an in-memory blob by staging it to a temp file (the reader opens a
    /// path). Returns the decode result for assertion.
    fn read_bytes(bytes: &[u8]) -> Result<NativeDataset> {
        read_bytes_projected(bytes, &[])
    }

    /// As [`read_bytes`], with a `variables` projection.
    fn read_bytes_projected(bytes: &[u8], variables: &[String]) -> Result<NativeDataset> {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        NetcdfReader::new().read_native(f.path(), variables, &Selection::All)
    }

    /// As [`read_bytes`], under a decode-time `select` of `axes`.
    fn read_bytes_select(bytes: &[u8], axes: Vec<AxisSelect>) -> Result<NativeDataset> {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        NetcdfReader::new().read_native(f.path(), &[], &Selection::Orthogonal(axes))
    }

    /// Stage a blob and open it. The `NamedTempFile` is returned because the
    /// reader memory-maps the path: dropping it would unlink the file underneath
    /// the open `NcFile`.
    fn open_blob(bytes: &[u8]) -> (tempfile::NamedTempFile, NcFile) {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        let file = NcFile::open(f.path()).expect("a readable blob");
        (f, file)
    }

    /// Decode with a HAND-BUILT `{dimension: indices}` map, bypassing the
    /// positional axis matching. [`dim_selection`] cannot produce a consumed
    /// string-length dimension — that is what [`field_dims`] is for — so the
    /// guard against one is unreachable through `read_native`; this reaches it,
    /// so the guard is tested rather than merely asserted in a comment.
    fn decode_with_dim_selection(
        bytes: &[u8],
        sel: &[(&str, Vec<usize>)],
    ) -> Result<NativeDataset> {
        let (_keep, file) = open_blob(bytes);
        let map: DimSelection = sel
            .iter()
            .map(|(d, idx)| ((*d).to_string(), idx.clone()))
            .collect();
        decode(&file, &[], Some(&map))
    }

    /// The `value` field of a decode, as `(dims, shape, floats)`.
    fn value_of(ds: &NativeDataset) -> (Vec<String>, Vec<usize>, Vec<f64>) {
        let f = ds.variables.get("value").expect("a `value` field");
        let ArrayData::F64(vals) = &f.data else {
            panic!(
                "a float variable must carry ArrayData::F64, got {:?}",
                f.data
            )
        };
        (f.dims.clone(), f.shape.clone(), vals.clone())
    }

    /// The `label` field of a decode, as `(dims, shape, strings)`.
    fn label_of(ds: &NativeDataset) -> (Vec<String>, Vec<usize>, Vec<String>) {
        let f = ds.variables.get("label").expect("a `label` field");
        assert_eq!(f.dtype, DType::Str, "a text variable is a `string` field");
        let ArrayData::Str(vals) = &f.data else {
            panic!(
                "a text variable must carry ArrayData::Str, got {:?}",
                f.data
            )
        };
        (f.dims.clone(), f.shape.clone(), vals.clone())
    }

    /// A minimal CDF-1 classic file: dim `n=3`, `float value(n) = [1,2,3]` and
    /// `char label(n) = "abc"`. `n` is a real axis (`value` lives on it), so
    /// `label` is THREE one-character strings, not one string `"abc"`.
    const CHAR_VAR_CDF1: &[u8] = b"\
        \x43\x44\x46\x01\x00\x00\x00\x00\x00\x00\x00\x0a\x00\x00\x00\x01\
        \x00\x00\x00\x01\x6e\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\
        \x00\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x02\x00\x00\x00\x05\
        \x76\x61\x6c\x75\x65\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\
        \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x0c\
        \x00\x00\x00\x7c\x00\x00\x00\x05\x6c\x61\x62\x65\x6c\x00\x00\x00\
        \x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
        \x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x88\x3f\x80\x00\x00\
        \x40\x00\x00\x00\x40\x40\x00\x00\x61\x62\x63\x00";

    /// The same 176-byte CDF-1 file with dims `n=4` and a PRIVATE `strlen=4`:
    /// `float value(n)` and `char label(n, strlen)` holding, NUL-padded, the four
    /// rows `"ab"`, `"cd  "`, `"efgh"`, `""`. Nothing but `label` uses `strlen`,
    /// so `strlen` is the string length: four strings on `dims == ["n"]`.
    const STRING_ROWS_CDF1: &[u8] = b"\
        \x43\x44\x46\x01\x00\x00\x00\x00\x00\x00\x00\x0a\x00\x00\x00\x02\
        \x00\x00\x00\x01\x6e\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x06\
        \x73\x74\x72\x6c\x65\x6e\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\
        \x00\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x02\x00\x00\x00\x05\
        \x76\x61\x6c\x75\x65\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\
        \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x10\
        \x00\x00\x00\x90\x00\x00\x00\x05\x6c\x61\x62\x65\x6c\x00\x00\x00\
        \x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\
        \x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x10\x00\x00\x00\xa0\
        \x3f\x80\x00\x00\x40\x00\x00\x00\x40\x40\x00\x00\x40\x80\x00\x00\
        \x61\x62\x00\x00\x63\x64\x20\x20\x65\x66\x67\x68\x00\x00\x00\x00";

    /// A 160-byte CDF-1 file whose `char label(strlen)` is ONE-dimensional on a
    /// private `strlen=5`, holding `"hi"` NUL-padded. The single dimension is the
    /// string length, so the field is a SCALAR string: `dims == []`.
    const SCALAR_STRING_CDF1: &[u8] = b"\
        \x43\x44\x46\x01\x00\x00\x00\x00\x00\x00\x00\x0a\x00\x00\x00\x02\
        \x00\x00\x00\x01\x6e\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x06\
        \x73\x74\x72\x6c\x65\x6e\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\
        \x00\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x02\x00\x00\x00\x05\
        \x76\x61\x6c\x75\x65\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\
        \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x0c\
        \x00\x00\x00\x8c\x00\x00\x00\x05\x6c\x61\x62\x65\x6c\x00\x00\x00\
        \x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\
        \x00\x00\x00\x02\x00\x00\x00\x08\x00\x00\x00\x98\x3f\x80\x00\x00\
        \x40\x00\x00\x00\x40\x40\x00\x00\x68\x69\x00\x00\x00\x00\x00\x00";

    /// [`CHAR_VAR_CDF1`] with an INTERIOR NUL: `char label(n) = "a\0c"` beside
    /// `float value(n)`. Pins that a NUL cell is the empty string, not `"\0"`.
    const CHAR_HOLES_CDF1: &[u8] = b"\
        \x43\x44\x46\x01\x00\x00\x00\x00\x00\x00\x00\x0a\x00\x00\x00\x01\
        \x00\x00\x00\x01\x6e\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\
        \x00\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x02\x00\x00\x00\x05\
        \x76\x61\x6c\x75\x65\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\
        \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x0c\
        \x00\x00\x00\x7c\x00\x00\x00\x05\x6c\x61\x62\x65\x6c\x00\x00\x00\
        \x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
        \x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x88\x3f\x80\x00\x00\
        \x40\x00\x00\x00\x40\x40\x00\x00\x61\x00\x63\x00";

    /// A `char` variable whose last dimension is a REAL AXIS — `n`, which
    /// `float value(n)` also lives on — is one one-character string per cell, on
    /// the variable's own `dims`/`shape`. It is NOT one string `"abc"`: `n`
    /// counts elements here, it does not measure a length.
    ///
    /// Cross-track ground truth (`xr.open_dataset(decode_times=False,
    /// mask_and_scale=True)` on these exact bytes, run against xarray 2024.7.0):
    /// `label` is `dims=('n',) shape=(3,) dtype=|S1` with values
    /// `[b'a', b'b', b'c']` — three cells, matching the three below.
    #[test]
    fn a_char_variable_on_a_shared_axis_is_one_string_per_cell() {
        let all = read_bytes(CHAR_VAR_CDF1).expect("plain decode");
        let (dims, shape, vals) = label_of(&all);
        assert_eq!(dims, ["n"]);
        assert_eq!(shape, [3]);
        assert_eq!(vals, ["a", "b", "c"]);
    }

    /// A NUL cell in such a variable is the EMPTY string — numpy renders a `|S1`
    /// NUL byte as `b""`, so `"a\0c"` is `["a", "", "c"]` and never `["a","\0","c"]`.
    /// Ground truth: xarray gives `[b'a', b'', b'c']` for these bytes.
    #[test]
    fn a_nul_cell_in_a_char_axis_is_the_empty_string() {
        let all = read_bytes(CHAR_HOLES_CDF1).expect("plain decode");
        let (dims, shape, vals) = label_of(&all);
        assert_eq!(dims, ["n"]);
        assert_eq!(shape, [3]);
        assert_eq!(vals, ["a", "", "c"]);
    }

    /// A `char label(n, strlen)` on a PRIVATE `strlen` is `n` strings of up to
    /// `strlen` bytes, NUL-padded — never `n * strlen` single characters. The
    /// length dimension is consumed, so `dims == ["n"]` and `shape == [4]`.
    ///
    /// Padding: trailing NULs are stripped and trailing SPACES are not (a space
    /// is data), so `"cd  "` survives whole while `"ab\0\0"` is `"ab"` and an
    /// all-NUL row is `""`. Ground truth: xarray gives
    /// `dims=('n',) shape=(4,) dtype=|S4` with `[b'ab', b'cd  ', b'efgh', b'']`.
    #[test]
    fn a_private_last_dimension_is_the_string_length_not_an_axis() {
        let all = read_bytes(STRING_ROWS_CDF1).expect("plain decode");
        let (dims, shape, vals) = label_of(&all);
        assert_eq!(dims, ["n"], "the strlen dimension is consumed");
        assert_eq!(shape, [4]);
        assert_eq!(vals, ["ab", "cd  ", "efgh", ""]);
    }

    /// The 1-D case of the same rule: a `char label(strlen)` whose only dimension
    /// is a private length is ONE string, and therefore a SCALAR field — `dims`
    /// and `shape` are both empty, exactly as xarray reports `dims=() shape=()
    /// dtype=|S5` holding `b'hi'` for these bytes.
    #[test]
    fn a_one_dimensional_char_on_a_private_dimension_is_a_scalar_string() {
        let all = read_bytes(SCALAR_STRING_CDF1).expect("plain decode");
        let (dims, shape, vals) = label_of(&all);
        assert!(dims.is_empty(), "a scalar string has no dims, got {dims:?}");
        assert!(
            shape.is_empty(),
            "a scalar string has no shape, got {shape:?}"
        );
        assert_eq!(vals, ["hi"]);
    }

    /// Read-everything mode RETURNS the text variable rather than skipping it —
    /// the whole point of the change. A reader that dropped it handed back a
    /// different set of fields from the Python and Julia tracks for the same
    /// bytes, which is a divergence, not a gap.
    #[test]
    fn read_everything_returns_the_text_variable_alongside_the_numeric_one() {
        for blob in [CHAR_VAR_CDF1, STRING_ROWS_CDF1, SCALAR_STRING_CDF1] {
            let all = read_bytes(blob).expect("plain decode");
            let mut names: Vec<&str> = all.variables.keys().map(String::as_str).collect();
            names.sort_unstable();
            assert_eq!(names, ["label", "value"], "read-everything must keep both");
        }
    }

    /// ...and the projection reaches it: naming the text variable projects to it
    /// (it is no longer an error), naming the numeric one still excludes it, and
    /// a name absent from the blob is still the error listing what is present.
    #[test]
    fn the_projection_selects_a_text_variable_like_any_other() {
        let one = read_bytes_projected(STRING_ROWS_CDF1, &["label".to_string()])
            .expect("a requested text variable now decodes");
        assert_eq!(one.variables.keys().collect::<Vec<_>>(), ["label"]);
        assert_eq!(label_of(&one).2, ["ab", "cd  ", "efgh", ""]);

        let other = read_bytes_projected(STRING_ROWS_CDF1, &["value".to_string()])
            .expect("projected decode");
        assert_eq!(other.variables.keys().collect::<Vec<_>>(), ["value"]);

        // The absent-name rule is untouched: a name that is NOT in the blob is
        // still an error listing what is.
        let err = read_bytes_projected(STRING_ROWS_CDF1, &["nope".to_string()])
            .expect_err("an absent name must still be an error");
        let msg = err.to_string();
        assert!(
            msg.contains("nope"),
            "error must name the absent one: {msg}"
        );
        assert!(
            msg.contains("label"),
            "error must list what IS present: {msg}"
        );
    }

    // ---- text × selection -------------------------------------------------
    //
    // A `select` is applied by dimension NAME to every array and every
    // coordinate, so a `char` variable is not exempt from it. The two features
    // meet at the string-length dimension a `char` array CONSUMES: it is not an
    // axis of the decoded field, so it must neither be selectable nor count
    // towards the positional rank match. Both halves are pinned below, because
    // getting the second one wrong would quietly rebind the selectors of every
    // NUMERIC variable in the same file.

    /// A `char label(n)` sharing its axis with `float value(n)` is windowed
    /// along `n` like any other array: the two must lose the SAME cells, in the
    /// same order, or they come back describing different rows of the file.
    #[test]
    fn a_select_windows_a_char_variable_on_its_shared_axis() {
        let ds = read_bytes_select(CHAR_VAR_CDF1, vec![AxisSelect::Indices(vec![2, 0])])
            .expect("a select over the shared axis");

        let (dims, shape, vals) = label_of(&ds);
        assert_eq!(dims, ["n"], "a windowed axis is never dropped");
        assert_eq!(shape, [2]);
        assert_eq!(vals, ["c", "a"], "in the order asked for, not sorted");

        let (vdims, vshape, vvals) = value_of(&ds);
        assert_eq!(vdims, ["n"]);
        assert_eq!(vshape, [2]);
        assert_eq!(
            vvals,
            [3.0, 1.0],
            "the numeric neighbour loses the same rows"
        );
    }

    /// A `char label(n, strlen)` on a PRIVATE `strlen` is windowed along `n` —
    /// its STRINGS are selected, whole. The failure this pins is truncation: a
    /// reader that planned the window over the on-disk dims would slice the
    /// string-length axis too and hand back `"ef"` for `"efgh"`, a wrong string
    /// rather than a wrong count, which no assertion on `shape` would catch.
    #[test]
    fn a_select_windows_the_strings_not_their_characters() {
        let ds = read_bytes_select(STRING_ROWS_CDF1, vec![AxisSelect::Indices(vec![2, 0])])
            .expect("a select over `n`");

        let (dims, shape, vals) = label_of(&ds);
        assert_eq!(dims, ["n"], "`strlen` is still consumed under a select");
        assert_eq!(shape, [2]);
        assert_eq!(
            vals,
            ["efgh", "ab"],
            "whole strings, in the order asked for"
        );

        assert_eq!(value_of(&ds).2, [3.0, 1.0]);
    }

    /// The rank a variable offers the positional match is its FIELD's, not its
    /// on-disk dimension count. `STRING_ROWS_CDF1` holds a rank-1 `value(n)` and
    /// a `char label(n, strlen)` that is a rank-1 FIELD on two on-disk
    /// dimensions, so nothing in the blob has rank 2 and a 2-axis `select` is
    /// the "matches no array" error — not a match against `label` that would
    /// bind axis 1 to a string length.
    #[test]
    fn a_consumed_string_length_never_makes_an_axis_count_match() {
        let err = read_bytes_select(
            STRING_ROWS_CDF1,
            vec![AxisSelect::All, AxisSelect::Indices(vec![0])],
        )
        .expect_err("a char variable's consumed dimension must not answer for rank 2");
        let msg = err.to_string();
        assert!(
            msg.contains("no variable in the blob has rank 2"),
            "the error must say the axis count matched nothing: {msg}"
        );
    }

    /// ...and the same rule stops a selector BINDING to a string length. In
    /// `SCALAR_STRING_CDF1` the `char label(strlen)` is a rank-1 array on disk
    /// and a rank-0 FIELD, so a 1-axis `select` may match `value(n)` only. Were
    /// the on-disk rank counted, `strlen` would bind the very same selector as
    /// `n` and the scalar string `"hi"` would come back sliced to `"h"` — a
    /// wrong string produced by a selection the caller aimed at `n`.
    #[test]
    fn a_consumed_string_length_is_not_an_axis_a_selector_can_bind_to() {
        let ds = read_bytes_select(SCALAR_STRING_CDF1, vec![AxisSelect::Indices(vec![0, 2])])
            .expect("a 1-axis select binds `n`, the only rank-1 field's dim");

        let (dims, shape, vals) = label_of(&ds);
        assert!(dims.is_empty(), "a scalar string stays scalar: {dims:?}");
        assert!(shape.is_empty(), "…and shapeless: {shape:?}");
        assert_eq!(vals, ["hi"], "the whole string, not a selected character");

        assert_eq!(value_of(&ds).2, [1.0, 3.0], "`n` was windowed as asked");
    }

    /// The guard behind that invariant. `dim_selection` cannot name a consumed
    /// string length (the test above is why), so this reaches the decode with a
    /// hand-built map instead: naming one is an ERROR. Silently ignoring it
    /// would hand back the full array while the caller believes it asked for a
    /// window, and honouring it would truncate every string — this repo's rules
    /// exist to prevent exactly that pair of outcomes.
    #[test]
    fn a_selection_naming_a_consumed_string_length_is_an_error() {
        let err = decode_with_dim_selection(STRING_ROWS_CDF1, &[("strlen", vec![0, 1])])
            .expect_err("a string length is not a selectable axis");
        let msg = err.to_string();
        assert!(msg.contains("strlen"), "the error must name it: {msg}");
        assert!(msg.contains("label"), "…and the variable: {msg}");
        assert!(
            msg.contains("string") && msg.contains("truncate"),
            "…and say why it is refused rather than applied: {msg}"
        );
    }

    /// A zero-length axis is legal for a text field too: an empty half-open
    /// `[1, 1)` keeps `n` in `dims` at length 0 and selects no strings. The
    /// failure mode is a field whose `shape` says `0` while its `data` still
    /// carries cells, which is what an unguarded gather would produce.
    #[test]
    fn an_empty_axis_is_legal_for_a_text_field_too() {
        let ds = read_bytes_select(
            STRING_ROWS_CDF1,
            vec![AxisSelect::Range {
                start: 1,
                stop: 1,
                step: 1,
            }],
        )
        .expect("an empty half-open range is a legal zero-length axis");

        let (dims, shape, vals) = label_of(&ds);
        assert_eq!(dims, ["n"], "a zero-length axis is KEPT in dims");
        assert_eq!(shape, [0]);
        assert!(vals.is_empty(), "shape 0 must mean no cells, got {vals:?}");

        let (vdims, vshape, vvals) = value_of(&ds);
        assert_eq!(vdims, ["n"]);
        assert_eq!(vshape, [0]);
        assert!(vvals.is_empty(), "the numeric field agrees: {vvals:?}");
    }

    /// The highest-severity thing this merge could have broken: teaching
    /// `dim_selection` to match on FIELD rank must not move which dimension a
    /// selector binds to for a NON-text variable. It cannot, and this is the
    /// proof rather than the claim — [`field_dims`] is the identity on every
    /// variable that is not a `char` array with a consumed last dimension, so
    /// for every other variable in every fixture here the rank it offers, the
    /// dim names it offers and their lengths are `var.dimensions()` verbatim.
    #[test]
    fn field_dims_is_the_identity_for_a_non_text_variable() {
        for blob in [
            CHAR_VAR_CDF1,
            CHAR_HOLES_CDF1,
            STRING_ROWS_CDF1,
            SCALAR_STRING_CDF1,
        ] {
            let (_keep, file) = open_blob(blob);
            let vars: Vec<NcVariable> = file.variables().unwrap().to_vec();
            let mut checked = 0;
            for var in &vars {
                if matches!(var.dtype(), NcType::Char | NcType::String) {
                    continue;
                }
                let (dims, shape) = field_dims(&vars, var);
                let on_disk: Vec<String> =
                    var.dimensions().iter().map(|d| d.name.clone()).collect();
                let sizes: Vec<usize> = var.dimensions().iter().map(|d| d.size as usize).collect();
                assert_eq!(dims, on_disk, "{} lost or gained a dim", var.name());
                assert_eq!(shape, sizes, "{} lost or gained a length", var.name());
                assert!(
                    consumed_length_dim(&vars, var).is_none(),
                    "{} is not a char array and consumes nothing",
                    var.name()
                );
                checked += 1;
            }
            assert!(checked > 0, "every fixture has a numeric variable to check");
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let err = read_bytes(b"NOPE, not a netcdf file at all").unwrap_err();
        assert!(matches!(err, Error::Format { .. }));
    }

    #[test]
    fn truncated_classic_file_is_an_error_not_a_panic() {
        // Valid classic magic + version byte, then nothing — must error cleanly.
        let err = read_bytes(b"CDF\x01\x00\x00").unwrap_err();
        assert!(matches!(err, Error::Format { .. }));
    }
}
