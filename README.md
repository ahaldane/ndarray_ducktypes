# NDarray Ducktypes

This module provides a set of numpy ndarray ducktypes which use the [new `__array_function__`](https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html) and `__array_ufunc__` functionality in numpy. Currently, this includes a more modern MaskedArray replacement, and a structured-array-like ducktype called ArrayCollection meant to allow simple tabular data analysis.

This module also provides helper tools for defining your own new ndarray-ducktypes, including various boilerplate methods and ready-made ducktype printing functionality, in the duckprint submodule. Getting the `str` and `repr` of ndarray ducktypes to mirror that of numpy's ndarrays is not trivial, and the `duckprint` module's goal is to make it easy.

This project is currently in "alpha" state, and may still go through significant reorganization. 

## Documentation

The [main documentation](doc/NDarray_Ducktypes.md) has an overview of this modules functionality, including installation instructions. Particular ducktypes and the ducktype-helper methods are documented at:

 * [Defining Your Own Ducktype](doc/MakingDucktypes.md)
 * [MakedArray documentation](doc/MaskedArray.md).
 * [ArrayCollection documentation](doc/ArrayCollection.md)

## Contributing

PRs and issues are welcome. If you would like to help, right now a good way is to write tests for the MaskedArray numpy-api implementations in `tests/test_MaskedArray.py`, and fixing problems with the api implementations you find. As of writing this, over 150 api functions are implemented, but only a small fraction have a dedicated test. There are already many tests ported from `numpy.ma`, but I want to go through each api method one-by-one and add tests for it in the `TEST_API` class.

