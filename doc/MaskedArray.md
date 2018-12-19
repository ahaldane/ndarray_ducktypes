MaskedArray
===========

This `MaskedArray` module provides "missing value" support on top of NumPy's `ndarray` type. You can construct a `MaskedArray` from any `ndarray` and then mask selected elements, which will be ignored in subsequent NumPy operations:

```python
>>> import numpy as np
>>> from MaskedArray import MaskedArray, MaskedScalar, X
>>> a = np.arange(4)
>>> m = MaskedArray(a)
>>> m[1:3] = X  # X denotes mask
>>> m
MaskedArray([0, X, X, 3])
>>> np.sum(m)
MaskedScalar(3)
```

`MaskedArray` is implemented as an `ndarray` ducktype, and so `MaskedArrays` can be substituted into almost any NumPy expression, for instance involving `np.sum`, `np.mean`, `np.concatenate`, and many others. `MaskedArray` supports most of NumPy's API methods and vectorized mathematical operations.

Historical Relationship to NumPy's `MaskedArray`s
-------------------------------------------------

This module should not be confused with NumPy's builtin `np.ma.MaskedArray` class, which has the same name and similar functionality but is otherwise independent. Historically, the present module was developed by forking NumPy's `np.ma.MaskedArray` codebase and updating it to take advantage of NumPy's new and powerful `__array_function__` interface for creating `ndarray` ducktypes. It is intended as a more modern replacement for NumPy's `np.ma.MaskedArray`.

Quickstart
==========

This module introduces three main new types on top of `NumPy`'s types:
 * `MaskedArray`, which behaves like NumPy's `ndarray` type but with masked elements
 * `MaskedScalar`, which behaves like NumPy's scalar types but which can be masked
 * `X`, a special variable representing a masked value for use in `MaskedArray` construction and assignment.

`MaskedArray` and `MaskedScalar` are implemented by storing two NumPy `ndarrays`(or scalars) of the same shape: The first stores the data, which can be any NumPy datatype, and the other stores a mask, and is of boolean datatype. These are normally hidden from the user, but it is useful to know this to understand `MaskedArray` behavior.

MaskedArray construction
------------------------

`MaskedArray`s can be created in multiple ways. One way is using Python lists,
in which case `X` can be used to signify masked elements:

```python
>>> a = MaskedArray([[1, X, 3], [X, X, 2], [X, 4, 1]])
>>> a
MaskedArray([[1, X, 3],
             [X, X, 2],
             [X, 4, 1]])
```

Another way is by specifying separate `data` and `mask` objects as the first two arguments of the constructor. The mask can be any object which numpy can cast to a boolean array and broadcast to the same shape as the data:

```python
>>> MaskedArray(np.ones(4), [0, 1, 0, 1])
MaskedArray([1., X , 1., X ])
```

`MaskedArray`s can also be constructed from other `MaskedArray`s, resulting in a view of the original:

```python
>>> MaskedArray(a)
MaskedArray([[1, X, 3],
             [X, X, 2],
             [X, 4, 1]])
```

`MaskedArray` Basic Attributes
-----------------------------

`MaskedArray` and `MaskedScalar` support two new attributes relative to `ndarray`:

The `.filled` method  returns a copy of the `MaskedArray`s data as a plain `ndarray`, with all masked elements replaced by a fill value, which is 0 by default. The `.filled()` method is the primary way of converting a `MaskedArray` back to an `ndarray`.

```python
>>> a.filled()
array([[1, 0, 3],
       [0, 0, 2],
       [0, 4, 1]])
```

The `.mask` attribute gives a readonly view of the boolean ndarray representing which elements are masked.

```python
>>> a.mask
array([[False,  True, False],
       [ True,  True, False],
       [ True, False, False]])
```

Otherwise `MaskedArray` supports most of the basic attributes of `ndarrays` like `.shape`, `.dtype`, `.strides` and others, with a few exceptions such as `.base`.

MaskedScalars and the masked input marker `X`
---------------------------------------------

When operating on a `MaskedArray`, NumPy will return a `MaskedScalar` object in  situations which ordinarily return a NumPy scalar object, for instance when indexing an array to get a single element. While `MaskedScalar`s support almost all the methods and attributes of NumPy scalars there are some important differences. Notably, `MaskedScalar`s are not part of the NumPy scalar type hierarchy, and so for instance code of the form `isinstance(var, np.inexact)` will fail, and `np.isscalar` will return `False`. In such situations you should use `.filled` first to obtain a NumPy scalar.

Like NumPy scalars, `MaskedScalar`s are immutable (except for structured scalars) and can be used as dict keys. When a `MaskedScalar` is tested for truth, for instance in if-statements, it will return `False` if masked, or the same as the corresponding numpy scalar if not masked. To get masked values to test as `True`, use `scalar.filled(True)` first.

The special `X` variable is not a `MaskedScalar`, and does not share attributes with NumPy types. Significantly, it does not have a `dtype`. It is a singleton of a special `MaskedX` class which acts as a general masked-value signifier, such that when it is used in `MaskedArray` operations and assignments it is effectively promoted to a masked `MaskedScalar` of the appropriate `dtype`. It is never returned by any NumPy operation or indexing. As a convenience the `X` variable can be promoted to a masked `MaskedScalar` of a specified `dtype` by calling it with the `dtype` as argument. Note that the `repr` of `MaskedScalar`s whose mask is `True` is represented using this `X(dtype)` construction:

```python
>>> X, type(X)
masked_input, MaskedX
>>> MaskedScalar(1), type(MaskedScalar(1))
MaskedScalar(1), MaskedScalar
>>> MaskedScalar(1, mask=True), type(MaskedScalar(1, mask=True))
X(int64), MaskedScalar
>>> X(np.float64), type(X(np.float64))
X(float64), MaskedScalar
```

General Principles of Behavior
------------------------------

This module follows some general principles which help give some intuition for its behaviors in detail.

For simple unary and binary operations, such as addition of two arrays, output elements will be masked at positions where either input element is masked:

```python
>>> a = MaskedArray([1, X, 3])
>>> b = MaskedArray([1, 2, X])
>>> a + b
MaskedArray([2, X, X])
>>> a == b
MaskedArray([True, X, X])
```

Additionally, some operations will mask output values for which an invalid operation occurred, such as division by zero:

```python
>>> 1.0 / MaskedArray([2, 0, 4])
MaskedArray([0.5 , X   , 0.25])
```

For "reduction"-like operations involving many elements, such as summing an array, the masked elements are effectively skipped and do not contribute to the reduction. The result will be masked only if there were no unmasked input elements:

```python
>>> np.sum(MaskedArray([1, X, 2])
MaskedScalar(3)
>>> np.sum(MaskedArray([X, X, X], dtype=int))
X(int64)
```

 * out arguments of ufuncs must be maskedarrays. This forces users to use safe code which doesn't lose the mask or risk using uninitialized data under mask.
 * methods which return `int` arrays meant for indexing (eg, argsort,
   nonzero) will return plain ndarrays. Otherwise API methods return maskedarrays.
 * masks sort as greater than all other values
 * np.any treats mask as False. np.all treats mask as True. This affects many other methods which depend on these. Discuss how to use `filled` on boolean masked arrays to choose other behaviors.
 * masked scalar with mask=True evaluates to false (important for if-statements)
   (not implemented yet... should this raise?)

Views and Copies
----------------

It is important to understand when a NumPy operation returns a view or a copy, and this is doubly true for `MaskedArray`s as either the mask or data may be views. `MaskedArray` frequently uses views for performance reasons, but this means that with improper use the user can sometimes get access to the nonsense values at masked positions.

During `MaskedArray` construction, when supplying the data and mask as `ndarrays`,  in most cases these are viewed by the `MaskedArray`and will be modified by operations on the `MaskedArray`. This can lead to confusing or undesirable behavior if you subsequently use the data array, particularly as `MaskedArray` may fill the data `ndarray` with nonsense values at masked positions. To avoid this you can use the `copy=True` argument to the `MaskedArray` constructor.

The `.mask` attribute of a `MaskedArray` is a readonly view of the internal mask data, and will be updated if the `MaskedArray` is assigned to. The `.filled()` method on the other hand returns a copy of the data by default, but if the `view` argument is set to `True` it will return a readonly view. Use caution with this view, because subsequent operations with the `MaskedArray` may put nonsense values at masked positions of its internal data array.

Unlike `ndarray`s, `MaskedArray`s do not support a `.base` attribute which can be used to tell if an array is a view. However, it is possible to check the `.base` attribute of the `ndarray`s returned by `.mask`, or `.filled` with `view=True`.

Implementation Details and Cautionary Notes
-------------------------------------------

 * we store a separate boolean array for the mask (compare to R (sentinels) and Julia (similar)).
 * .filled and .mask return views, and Constructor views the inputs. Be careful!
 * MaskedScalars behave differently from numpy scalars in quite a few cases -- use filled a lot!
 * for complex datatypes, the masks of .real and .imag behave in a certain way. Sugest using nanmethod if heavy use of complex types.


`MaskedArray` Reference
=======================

Construction
------------

Indexing
--------

Ufuncs
------
