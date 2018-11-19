## ND Ducktypes

A set of numpy `ndarray` ducktypes testing the new `__array_function__` and `__array_ufunc__` functionality in numpy.

Currently three ducktypes are included:
 * `MaskedArray`, which provides missing-value support, mirroring numpy's MaskedArray
 * `ArrayCollection`, which mimics numpy's structured arrays but with better memory layout
 * `LazyArray`, which evaluates ufuncs in a lazy way, optimizing out any intermediate arrays

This repository also implements a helper module, `duckprint`, which provides ready-made printing routines to help implement the str/repr of ducktypes.

## Current Status

 * `MaskedArray`: In a working state. Implements both `__array_ufunc__` and `__array_function__`. Supports all ufuncs with domain masking just like numpy's MaskedArray.
 * `ArrayCollection`: In a working state. Implements `__array_function__`, though only a few api functions.
 * `LazyArray`: placeholder only, not in a working state.
 * `duckprint`: Working. This is a reworking of numpy's `arrayprint.py`, removing all the legacy stuff and reorganizing to make it easier for ducktypes to modify the printing dtype-dispatch process.

## Design Notes

### For MaskedArray:

 * No support for masking individual fields of structured dtypes. Use an `ArrayCollection` of `MaskedArray`s instead, or use the `MaskedArrayCollection` implemented in `test.py`.
 * No more `nomask`: the mask is always stored as a full ndarray.
 * No more returned `masked` singleton: Instead scalar values are returned as MaskedScalar instances. See comments in MaskedArray.py for discussion. A special masked singleton with no dtype exists for array construction purposes only.
 * No more `fill_value` stored with instances: You must always supply the desired fill as an argument to `filled()`
 * No more attempt to preserve the hidden data behind masked values. MaskedArray is free to modify these elements arbitrarily.
 * Features preserved from numpy's maskedarray: 1. The mask is not "sticky", it behaves as the "ignore" or "skipna" style described in the MaskedArray NEP. "na" style will not be supported. 2. Ufuncs replace out-of-domain inputs with mask.

### For duckprint

Ducktype implementors who wish to use the ready-made duckprint functionality can implement `__repr__` and `__str__` as follows, optionally also implementing a class method `__nd_duckprint_dispatch__` in their ducktype for further customization:
```python
    @classmethod
    def __nd_duckprint_dispatcher__(cls):
        return my_dispatcher

    def __str__(self):
        return duckprint.duck_str(self)

    def __repr__(self):
        return duckprint.duck_repr(self)
```
The optional `__nd_duckprint_dispatcher__` method should return a `FormatDispatcher` object, whose class is defined in the duckprint module. If `__nd_ducktype_dispatcher__` is not implemented, the default dispatcher `duckprint.default_duckprint_dispatcher` will be used. The dispatcher is used to determine how to print individual array elements in an array context, dispatching based on dtype: given a set of array elements to print, it returns a formatting function for those elements. Further documentation of `FormatDispatcher` is forthcoming.
