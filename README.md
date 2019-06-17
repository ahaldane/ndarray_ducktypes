## NDarray Ducktypes

This module provides a set of numpy `ndarray` ducktypes which use the new `__array_function__` and `__array_ufunc__` functionality in numpy. Currently, this includes a more modern `MaskedArray` replacement, and a structured-array-like ducktype called `ArrayCollection` meant to allow simple tabular data analysis. It also provides ready-made printing functionality useful for anyone implementing a new ndarray ducktype, in the `duckprint` submodule.

### MaskedArray

`MaskedArray` provides missing-value support for numpy arrays, and mirrors and improves numpy's `np.ma.MaskedArray`. It is meant as a new iteration of numpy's MaskedArray which uses Numpy's new and powerful "ducktyped" array functionality, so that the `MaskedArray`s from this module can be used as drop-in replacements in arbitrary numpy expressions. 

See [the MakedArray documentation](doc/MaskedArray.md)

### ArrayCollection

`ArrayCollection` mimics numpy's structured arrays but avoids binary representation issues of structured arrays and can have more optimized memory layout. `ArrayCollection` is meant for use as a "pandas-lite" for working with tabular data, which avoids the difficulties of using structured arrays for this purpose. 

This module also contains a `MaskedArrayCollection` type which allows you to mask individual fields in elements of an `ArrayCollection`.

See [the ArrayCollection documentation](doc/ArrayCollection.md) (WIP)

### Duckprint

This repository implements a helper module, `duckprint`, which provides ready-made printing routines to help implement the str/repr of ducktypes. `duckprint` is used internally by the `MaskedArray` and `ArrayCollection` classes defined in this repository.


See [the duckprint documentation](doc/duckprint.md)


