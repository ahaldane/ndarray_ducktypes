# NDarray Ducktypes

This module provides:

 1. A set of new numpy `ndarray` ducktypes, called `MaskedArray`, `ArrayCollection`, and `UnitArray` using the [new ducktype numpy api](https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html). See below for descriptions.
 2. Helper tools to help you define your own numpy ducktypes. This including helper methods, mixins, and most importantly printing functionality in the `duckprint` submodule to help implement your ducktype's `str` and `repr`.

This project is currently in "alpha" stage.

## New Ndarray Ducktypes

### MaskedArray

`MaskedArray` provides missing-value support for numpy arrays, and is meant as a new iteration of numpy's `np.ma.MaskedArray`. `MaskedArray`s from this module can be used as drop-in replacements in arbitrary numpy expressions. (95% complete.)

See [the MakedArray documentation](doc/MaskedArray.md).

### ArrayCollection

`ArrayCollection` mimics numpy's structured arrays but avoids binary representation issues of structured arrays and can have more optimized memory layout. `ArrayCollection` is meant for use as a "pandas-lite" for working with tabular data, which avoids the difficulties of using structured arrays for this purpose. (5% complete)

This module also contains a `MaskedArrayCollection` type which allows you to mask individual fields in elements of an `ArrayCollection`.

See [the ArrayCollection documentation](doc/ArrayCollection.md)

### UnitArray

Will implement a ducktype which stores unit information in addition to the dtype. 0% complete.

## Ducktype Helper Tools

This module implements helper classes and methods and mixins to help simplify defining new ducktypes, with some recommendations for implementation which are reuired by some of the helper methods, see docs linked below.

To make ducktypes quicker to define, the `duckprint` provides ready-made printing routines to help implement the str/repr of ducktypes. `duckprint` is used internally by the `MaskedArray` and `ArrayCollection` classes defined in this repository.

See [Main Documentation](doc/NDarray_Ducktypes.md)
