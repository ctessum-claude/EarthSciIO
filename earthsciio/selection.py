"""The ``select`` vocabulary — ONE spelling, shared by every reader.

A ``select`` is ``{"axes": [<axis>, ...]}`` (a bare list is accepted as
shorthand), one entry per array dimension in the array's own dimension order.
Each ``<axis>`` is ``"all"``, ``{"indices": [...]}`` (an explicit, possibly
non-contiguous, ordered index list, returned in the order given) or
``{"slice": [start, stop, step?]}`` (half-open, ``step`` defaults to 1).
**Indices are 0-based**, in every reader and every track.

These helpers live here rather than in one backend because two readers now parse
the same vocabulary and must not drift apart: the store-backed
:class:`~earthsciio.backends.zarr.ZarrReader`, whose selection decides which chunk
OBJECTS are fetched, and the whole-file
:class:`~earthsciio.readers.NetCDFReader`, which fetches the same blob under the
same cache key and materialises only the requested hyperslab. What a selection
SAVES differs; what it MEANS does not.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

__all__ = ["_ALL", "_parse_axis", "_select_axes", "_resolve_axis_indices"]

_ALL = ("all",)


def _parse_axis(spec: Any) -> Tuple:
    """Normalize one axis selector to a tagged tuple.

    Accepts ``"all"``/``None`` (whole axis), ``{"indices": [...]}`` (an explicit,
    possibly non-contiguous, ordered index list), ``{"slice": [start, stop,
    step?]}`` (a strided range, ``step`` defaults to 1), or a bare list of ints
    (shorthand for ``indices``).
    """
    if spec is None or spec == "all":
        return _ALL
    if isinstance(spec, dict):
        if "indices" in spec:
            return ("indices", [int(i) for i in spec["indices"]])
        if "slice" in spec:
            s = list(spec["slice"])
            start = int(s[0])
            stop = int(s[1])
            step = int(s[2]) if len(s) > 2 else 1
            return ("slice", start, stop, step)
        raise ValueError(f"unrecognized axis selector: {spec!r}")
    if isinstance(spec, (list, tuple)):
        return ("indices", [int(i) for i in spec])
    raise ValueError(f"unrecognized axis selector: {spec!r}")


def _select_axes(select: Any) -> Optional[List[Any]]:
    """Extract the ordered per-axis selector list from a ``select`` argument.

    ``None`` ⇒ ``None`` (all). A ``{"axes": [...]}`` mapping or a bare list both
    yield the axis list. Anything else ⇒ ``None`` (all).
    """
    if select is None:
        return None
    if isinstance(select, dict) and "axes" in select:
        return list(select["axes"])
    if isinstance(select, (list, tuple)):
        return list(select)
    return None


def _resolve_axis_indices(axis: Tuple, dim_len: int) -> List[int]:
    """Resolve a tagged axis selector to its ordered list of global indices."""
    if axis[0] == "all":
        return list(range(dim_len))
    if axis[0] == "indices":
        idxs = axis[1]
        for g in idxs:
            if g < 0 or g >= dim_len:
                raise IndexError(f"index {g} out of range for dimension length {dim_len}")
        return list(idxs)
    if axis[0] == "slice":
        _, start, stop, step = axis
        if step < 1:
            raise ValueError(f"slice step must be >= 1, got {step}")
        return list(range(start, stop, step))
    raise ValueError(f"unrecognized axis selector: {axis!r}")


