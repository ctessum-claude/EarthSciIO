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
//!   native grid, carrying its `units`/`calendar` verbatim.

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
        _select: &Selection,
    ) -> Result<NativeDataset> {
        // Selection::All is the only variant today; the whole blob is read.
        let file = NcFile::open(blob_path).map_err(fmt_err)?;
        decode(&file, variables)
    }
}

/// Decode an opened NetCDF file into native arrays, honoring the `variables`
/// filter (empty = all data variables; coordinate variables are always kept).
///
/// A requested name absent from the blob is an error listing what is present —
/// the rule the `parquet`/`shapefile` readers and the Python/Julia netcdf
/// readers already follow, so a typo'd `file_variable` cannot read back as a
/// silently missing array in this track alone.
fn decode(file: &NcFile, variables: &[String]) -> Result<NativeDataset> {
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
        // Only numeric variables become native fields; text/opaque/compound/…
        // variables (e.g. an ERA5 `expver` string) carry no array and are skipped.
        let Some(class) = classify(var) else { continue };
        let field = decode_field(file, var, class)?;

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
}

/// Classify a variable, or `None` if it is not a numeric native field.
fn classify(var: &NcVariable) -> Option<FieldClass> {
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

/// Decode one variable's values into a [`NativeField`] under `class`.
fn decode_field(file: &NcFile, var: &NcVariable, class: FieldClass) -> Result<NativeField> {
    let dims: Vec<String> = var.dimensions().iter().map(|d| d.name.clone()).collect();
    let shape: Vec<usize> = var.dimensions().iter().map(|d| d.size as usize).collect();
    let name = var.name();

    match class {
        FieldClass::Float => {
            // scale_factor/add_offset applied in double; _FillValue/missing_value
            // folded to NaN. Values are row-major (C order) per `shape`.
            let arr = file.read_variable_unpacked_masked(name).map_err(fmt_err)?;
            Ok(NativeField {
                dtype: DType::Float64,
                dims,
                shape,
                data: ArrayData::F64(arr.iter().copied().collect()),
                fill_value: None, // folded into NaN
            })
        }
        FieldClass::Int32 => {
            let raw = file.read_variable_as_f64(name).map_err(fmt_err)?;
            Ok(NativeField {
                dtype: DType::Int32,
                dims,
                shape,
                data: ArrayData::I32(raw.iter().map(|&v| v as i32).collect()),
                fill_value: int_fill(var),
            })
        }
        FieldClass::Int64 => {
            let raw = file.read_variable_as_f64(name).map_err(fmt_err)?;
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
    use std::io::Write;

    /// Decode an in-memory blob by staging it to a temp file (the reader opens a
    /// path). Returns the decode result for assertion.
    fn read_bytes(bytes: &[u8]) -> Result<NativeDataset> {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        NetcdfReader::new().read_native(f.path(), &[], &Selection::All)
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
