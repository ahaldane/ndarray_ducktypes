## ND Ducktypes

A set of numpy `ndarray` ducktypes testing the new `__array_function__` and `__array_ufunc__` functionality in numpy.

Currently three ducktypes are included:
 * `MaskedArray`, which provides missing-value support, mirroring numpy's MaskedArray
 * `ArrayCollection`, which mimics numpy's structured arrays but with better memory layout and avoids binary representation issues.
 * `LazyArray`, which evaluates ufuncs in a lazy way, optimizing out any intermediate arrays

This repository also implements a helper module, `duckprint`, which provides ready-made printing routines to help implement the str/repr of ducktypes.

## Current Status

 * `MaskedArray`: In an advanced state, and passes a large test suite derived from numpy's MaskedArray test suite. Is also well documented (see docs folder)
 * `ArrayCollection`: In a working but incomplete state. Only implements a few `__array_function__` API functions.
 * `LazyArray`: placeholder only, not in a working state.
 * `duckprint`: Working. This is a reworking of numpy's `arrayprint.py`, removing all the legacy stuff and reorganizing to make it easier for ducktypes to modify the printing dtype-dispatch process.

## Design Notes

### For MaskedArray:

 * No support for masking individual fields of structured dtypes. Use an `ArrayCollection` of `MaskedArray`s instead, or (better) use `MaskedArrayCollection`.
 * No more `nomask`: the mask is always stored as a full ndarray.
 * No more returned `masked` singleton: Instead scalar values are returned as MaskedScalar instances. See comments in MaskedArray.py for discussion. A special masked singleton called `X` with no dtype exists for array construction purposes only.
 * No more `fill_value` stored with instances: You must always supply the desired fill as an argument to `filled()`
 * No more attempt to preserve the hidden data behind masked values. MaskedArray is free to modify these elements arbitrarily.
 * Features preserved from numpy's maskedarray: 1. The mask is not "sticky", it behaves as the "ignore" or "skipna" style described in the MaskedArray NEP. "na" style will not be supported. 2. Ufuncs replace out-of-domain inputs with mask.
 * out arguments to ufuncs/methods must be MaskedArrays too
 * more careful preservation of dtype (old MA would often cast to float64)
 * np.sort now doesn't mix maxval and masked vals up.

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
