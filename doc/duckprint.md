Duckprint
=========

The duckprint module implements functionality for printing ndarray-ducktypes in
the numpy output style, which is meant to be re-used by ducktype implementors.

Implementors of new ndarray ducktypes can use this module by implementing 
`__repr__` and `__str__` for their ducktype as follows, and optionally also
implementing a class method `__nd_duckprint_dispatch__` in their ducktype for
further customization:
```python
    def __str__(self):
        return duckprint.duck_str(self)

    def __repr__(self):
        return duckprint.duck_repr(self)

    @classmethod
    def __nd_duckprint_dispatcher__(cls):
        return my_dispatcher
```

Using `duck_str` and `duck_repr`
================================

The main requirements for using `duck_str` and `duck_repr` are that your
ducktype allows numpy-style indexing. That is, your ducktype should be 
indexable along multiple axes at once and support tuple-indexing, as well as
boolean and integer fancy-indexing. It should also have basic ndarray
attributes such as `.dtype`, `.shape`, and `.ndim`. When "fully" indexed at all
axes it should return a scalar value, and not a 0d array. If your ducktype's
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
