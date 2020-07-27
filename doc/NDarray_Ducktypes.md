NDarray Ducktypes
=================

This module provides a set of numpy ndarray ducktypes which use the new `__array_function__` and `__array_ufunc__` functionality in numpy. Currently, this includes a more modern MaskedArray replacement, and a structured-array-like ducktype called ArrayCollection meant to allow simple tabular data analysis.

This module also provides helper tools for defining your own new ndarray-ducktypes, including various boilerplate methods and ready-made ducktype printing functionality, in the duckprint submodule. Getting the `str` and `repr` of ndarray ducktypes to mirror that of numpy's ndarrays is not trivial, and the `duckprint` module's goal is to make it easy.

Insallation and Usage
---------------------

This package supports standard [python package installation](https://packaging.python.org/tutorials/installing-packages/) using setuptools. The simplest way is to use `pip` to install from the cloned source directory:

    $ pip install /path/to/ndarray_ducktypes

and uninstall by "pip uninstall ndarray_ducktypes".

To use these ducktypes, numpy's `__array_function__` must be enabled, and it is currently disabled by default. To enable, in your shell do:
```
export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1
```

New Ndarray Ducktypes
---------------------

These are documented in the following links:

 * [MakedArray documentation](MaskedArray.md).
 * [ArrayCollection documentation](ArrayCollection.md)
 * UnitArray documentation (TODO)

Defining Your Own Ducktypes
---------------------------

This module provides code to help you define your own ducktypes, but expects the ducktypes to be implemented in a specific way which is described here.

Before implementing a ducktype, you should familiarize yourself with the basics of numpy's `__array_function__`, documented [HERE](),  and `__array_ufunc__`, documented [HERE]().

Mixins
------

Implementing a new ducktype can be sped up by using mixins provided by this module as well as by numpy.

Numpy provides a mixin, `NDArrayOperatorsMixin`, from `numpy.lib.mixins`, which defines almost all python operators incliuding arithmetic and comparisons on your ducktype. It implements them by calling the corresponding numpy ufunc. See its [documentation](https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html). If you then define an appropriate `__array_ufunc__` method as in the documentation linked above, your ducktype will automatically support Python arithmetic and comparison.

This module provides another mixin, `NDArrayAPIMixin`, in `ndarray_ducktypes.ndarray_api_mixin`, which defines many of the numpy-api methods as attributes on your ducktype by calling the corresponding Numpy-api function. For instance, this allows you to call `arr.sum()` on your ducktype, and this mixin implements this method to call `np.sum(arr)`. If you then define an appropriate `__array_function__` your ducktype will automatically support all of these attributes. Some numpy attribute method operate inplace, such as sort. This mixin also implements other ndarray attributes which are not part of the numpy-api such as `.fill()`, `.imag` and `.flatten()`.

There are some attributes which `NDArrayAPIMixin` does not attempt to implement because they typically depend on how your ducktype stores its data internally. These are the attributes `dtype`, `shape`, `strides`, `flags`, `base`, the methods `astype`, `view`, `item`, `resize` (which needs to be inplace), as well as the less commonly used attributes, `byteswap`, `ctypes`, `data`, `dump`, `dumps`, `flat`, `getfield`,  `itemset`, `itemsize`.

The `implements` decorator
--------------------------

In order to use `__array_function__`, you will need to define implementations for all the api-functions you wish to support, arrange to have these called in your `__array_function__` implementation, and correctly dispatch based on types of the input arguments. The numpy docs on `__array_function__` give some guidance on how to do this, but this module provides an extra helper function, `new_ducktype_implementation`, provided from `ndarray_ducktypes.common`.

This will return a decorator to decorate each of your api-implementations with, which will record all decorated api functions and which also will automate ducktype dispatch for you. It allows for a simple `__array_function__` implementation, as in this example:
```python
class MyDuckType:
    def __array_function__(self, func, types, arg, kwarg):
        known_types = (MyDuckType,)

        impl, check_args = implements.handled_functions.get(func, (None, None))
        if impl is None or not check_args(arg, kwarg, types, known_types):
            return NotImplemented

        return impl(*arg, **kwarg)

implements = new_ducktype_implementation()

@implements(np.sum)
def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    ... your implementation here ...


@implements(np.concatenate)
def concatenate(arrays, axis=0, out=None):
    ... your implementation here ...
```

To use it, you decorate each of your implementations giving the target
numpy-api function as argument.  Then in `__array_function__`, the `implements`
decorator is used to get your implementation, accessible through the
`.handled_functions` dict, as well as an automatically defined check function
(`check_args` above) which is used to implement the `__array_function__`
dispatch logic.

By default, `check_args(arg, kwarg, types, known_types)` will check that all of the elements in the `types` argument are of one of the `known_types` you provide or an ndarray, and will ignore `arg` and `kwarg`. In many cases you can simply pass on the `types`  argument of `__array_function__` to `check_args`.

However, in other cases you will want to only check certain args or will require a more complicated type check. For this, the decorator provides a `checked_args` optional argument to customize behavior. This may be either a tuple of strings which are argument names of the function being decorated, or a function with signature `(args, kwds, types, known_types)`. For example:

```python
@implements(np.split, checked_args=('ary',))
def split(ary, indices_or_sections, axis=0):
    ... your implementation here ...
```
Here, only the `ary` argument will be checked to have your ducktype, while the argument `indices_or_sections` will not be checked and may be a plain python list, allowing calls like `np.split(duckarr, [3, 5, 6, 10])`. If an optional keyword arg name is given in the `checked_args` tuple, it is only checked if explicitly supplied.

The most flexible way of controlling which args are checked is to provide a function, as in:
```python
@implements(np.select, checked_args=lambda a,k,t,n: [type(x) for x in a[1]])
def select(condlist, choicelist, default=0):
    ... your implementation here ...
```
In this example, the elements of the `choicelist` argument are checked for type, and not condlist. The function returns a list of types, which the implements decorator will then automatically check to be a known type. 

Lastly, the `checked_args` function you provide may raise a `NotImplementedError` to signify that type check failed. Since you have access to all of the `__array_function__` arguments you can implement any dispatch behavior desired.

Duck Scalars
------------

Numpy distinguishes between ndarrays and numpy scalars, which are subtypes of `np.generic`. For instance, when indexing an ndarray to get a single element, the returned value is a numpy scalar, and not an ndarray. It is desirable to mimic this behavior in your ducktype, since ndarrays and scalars behave differently in a number of situations.

While for some ducktypes it might be appropriate to simply return numpy scalars when a scalar should be returned, in most cases you want the returned scalar to retain information related to your ducktype, and therefore you will want to define duck-scalar types. How to do so is not obvious, and there are technical challengees caused by the fact that numpy scalar types cannot be subclassed.

The strategy used in this module is to define a single duck-scalar type associated to each ducktype, whose instance's datatype is determined by the `.dtype` attribute. You should do this too for your ducktype in order to fully use this module's helper functions, in particular its printing functionality,

Your duck-scalar should support `__array_function__` as well as all the ndarray attributes (just as numpy scalars do). Numpy scalars support indexing, and indexing with an empty tuple `()` should return a copy of the scalar.

Furthermore, in order to use some of the helper functions, this module expects two additional attributes to be implemented: Both the ducktype and the duck-scalar need a record of each other, so they should both have an `ArrayType` attribute equal to the ducktype type, and a `ScalarType` attribute equal to the scalar ducktype. As a convenience, you can call `ndarray_ducktypes.common.ducktype_linklink(arraytype, scalartype, known_types=None)` to automatically add these attributes.

This module also provides methods `ndarray_ducktypes.common.is_ndducktype(val)`, which tests whether val is either an ndarray, ducktype, or duck-scalar (and not a numpy scalar), and `ndarray_ducktypes.common.is_duckscalar(val)` which tests
whether val is a duck-scalar or a numpy scalar. These can be useful when implementing the numpy api.

Subclassing Ducktypes
---------------------

In order to combine different ndarray-ducktypes together, or provide additional behavior, it can sometimes be useful to subclass your ducktype or someone else's ducktype. It is reasonable to try to implement your ducktype's api to allow others to subclass it and still use your api function implementations.

To help with this, the function `get_duck_cls(*args)` is defined in `ndarray_ducktypes.common`. Given a set of input arguments, it returns the most derived ducktype class, or the ducktype which contains all others in its `known_types`. In ambiguous cases, for instance multiple ducktypes which are not related by inheritance and are both (or neither) in each other's `known_types`, it will raise a TypeError. Scalar inputs are treated as the corresponding array type.

Often, by calling this at the start of your api method implementation on all the inputs and converting all inputs to the returned class, you will allow downstream users to subclass your ducktype such that your api methods preserve their subtype.


Duckprint
=========

The duckprint module implements functionality for printing ndarray-ducktypes in
the numpy output style, which is intended for you to re-use when implementing your ducktype.

You use this module by implementing `__repr__` and `__str__` for your ducktype
by using `duck_str` and `duck_repr` from the `ndarray_ducktypes.duckprint`
submodule as follows, and optionally also implementing a class method
`__nd_duckprint_dispatch__` in your ducktype for further customization:
```python
    def __str__(self):
        return duckprint.duck_str(self)

    def __repr__(self):
        return duckprint.duck_repr(self)

    @classmethod
    def __nd_duckprint_dispatcher__(cls):
        return my_dispatcher
```

The main requirements for using `duck_str` and `duck_repr` are that your
ducktype allows numpy-style indexing. That is, your ducktype should be
indexable along multiple axes at once and support tuple-indexing, as well as
boolean and integer fancy-indexing. It should also have basic ndarray
attributes such as `.dtype`, `.shape`, and `.ndim`. When "fully" indexed at all
axes it should return a scalar value, and not a 0d array. See the discussion of scalar ducktypes above. If your ducktype's
scalars are the same as numpy's built-in scalars (eg, `np.uint32`, `np.float64`
etc) then nothing more should be needed, and `duckprint`'s default scalar
formatters should work without modification and `__nd_duckprint_dispatcher__`
can be ommitted above.

Defining formatters for custom scalar types
===========================================

If you have created new scalar types associated to your ducktype (eg, a
`MaskedScalar` type to accompany `MaskedArray`) you will also need to supply
printing dispatch routines for your scalar and should define the optional
`__nd_duckprint_dispatcher__` method in your ducktype as shown above, and
create scalar `ElementFormatter` subclasses corresponding to your scalars.

Defining new `ElementFormatter`s
--------------------------------

The `ElementFormatter` types are used to determine which formatting function
to use to print scalar values as part of the string representation of an array. This is typically different from how the scalar is printed on its own, for instance because array elements have whitespace padding so that they align well when printed.  An `ElementFormatter` needs to define three methods:

  * `will_dispatch(elem)` is used to determine whether this formatter can
    print the values in `elem`. It should be passed an array of elements to be
    printed, and return True or False. If True, this means that
    `get_format_func` can be called on those elements.
  * `get_format_func(elem, **options)` takes an array of elements and the
    dictionary of user-specified print-options, and returns a string-formatting
    function. This function should have signature `f(e)` where `e` is a single
    scalar value, and returns the string representation of that value.
  * `check_options` validates the user-specified printing-options (eg,
    `precision`, `sign` etc). This will validate any relevant options which the
    user has changed, and return a list of options it didn't handle.

In many cases it will be possible to re-use or subclass the `ElementFormatter`s
already defined in `duckprint`, which are `BoolFormatter`, `IntegerFormatter`,
`FloatingFormatter`, `ComplexFloatingFormatter`, `DatetimeFormatter`,
`TimedeltaFormatter`, `SubArrayFormatter`, `StructuredFormatter`,
`ObjectFormatter`, `StringFormatter`, and `VoidFormatter`. These handle the
builtin-numpy types, and can be subclassed to handle modified types.

Using your custom `ElementFormatter`s in your ducktype
------------------------------------------------------

If defined in your ndarray ducktype, `__nd_duckprint_dispatcher__` should
return a `duckprint.FormatDispatcher` object. If
`__nd_ducktype_dispatcher__` is not implemented, the default dispatcher
`duckprint.default_duckprint_dispatcher` will be used. The dispatcher is used
to determine how to print individual array elements in an array context,
dispatching based on dtype. Most of the time you will want to create a dispatcher
as:
```
my_dispatcher = FormatDispatcher([Ef1, Ef2, Ef3], default_duckprint_options)
```
where `Ef1`, `Ef2` and so on are your `ElementFormatter` instances. When given a set
of elements to print, the `FormatDispatcher` will try the `will_dispatch` method
of the ElementFormatters one-by-one until one returns True, and then use that
`ElementFormatter` to print the elements. The dispatcher created here should
be returned by `__nd_ducktype_dispatcher__` in your ducktype.

The default formatter list is available in the `duckprint` module
variable `default_duckprint_formatters`, the default printing options in
`default_duckprint_options`, and the default dispatcher is
`default_duckprint_dispatcher`.
