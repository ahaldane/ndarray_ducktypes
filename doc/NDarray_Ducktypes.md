# NDarray Ducktypes

This module provides:

 1. A set of new numpy `ndarray` ducktypes, including `MaskedArray`, `ArrayCollection`, and `UnitArray` using the [new ducktype numpy api]. See below for descriptions.
 2. Helper tools to help you define your own numpy ducktypes. This including helper methods, mixins, and most importantly printing functionality in the `duckprint` submodule to help implement your ducktype's `str` and `repr`.

## Insallation and Usage

This package supports standard [python package installation](https://packaging.python.org/tutorials/installing-packages/) using setuptools. The simplest way is to use `pip` to install from the cloned source directory:

    $ pip install /path/to/ndarray_ducktypes

and uninstall by "pip uninstall ndarray_ducktypes".

To use these ducktypes, numpy's `__array_function__` must be enabled, and it is currently disabled by default. To enable, in your shell do:
```
export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1
```
## Overview of Functionality

 * [MakedArray documentation](MaskedArray.md).
 * [ArrayCollection documentation](ArrayCollection.md)
 * UnitArray documentation (TODO)
 * LazyArray documentation (TODO)
 * [Making Ducktypes documentaiton](MakingDucktypes.md)

### MaskedArray

`MaskedArray` provides missing-value support for numpy arrays, and is meant as a new iteration of numpy's `np.ma.MaskedArray`. `MaskedArray`s from this module can be used as drop-in replacements in arbitrary numpy expressions. (95% complete.)

See [the MakedArray documentation](doc/MaskedArray.md).

### ArrayCollection

`ArrayCollection` mimics numpy's structured arrays but avoids binary representation issues of structured arrays and can have more optimized memory layout. `ArrayCollection` is meant for use as a "pandas-lite" for working with tabular data, which avoids the difficulties of using structured arrays for this purpose. (5% complete)

See [the ArrayCollection documentation](doc/ArrayCollection.md)

This module also defines a composite `MaskedArrayCollection` type which allows you to mask individual fields in elements of an `ArrayCollection`.

### UnitArray (Tentative)

Will implement a ducktype which stores unit information in addition to the dtype. 0% complete.

### LazyArray (Tentative)

Will implement a ducktype which delays evaluation of numpy operations until `.eval()` is called. Doing `np.sin(a + b)` will return an unevaluated LazyArray on which you can call `.eval()`. The default `.eval` will avoid creation of intermediate arrays using numpy's `out=` method arguments for a small speedup, but it should be possible to override `.eval` for bigger gains, eg numba compilation.  0% complete.

## Helper Tools for Making Your Own Ducktype

This module implements helper classes and methods and mixins to help simplify defining new ducktypes, with some recommendations for implementation which are reuired by some of the helper methods, see docs linked below.

To make ducktypes quicker to define, the `duckprint` provides ready-made printing routines to help implement the str/repr of ducktypes. `duckprint` is used internally by the `MaskedArray` and `ArrayCollection` classes defined in this repository.

See [Defining Your Own Ducktype](MakingDucktypes.md).
