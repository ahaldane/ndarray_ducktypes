MaskedArray
===========

This `MaskedArray` module provides "missing value" support on top of NumPy's `ndarray` type. You can construct a `MaskedArray` from any `ndarray` and mask selected elements, which will then be ignored in subsequent NumPy operations::

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

`MaskedArray` is implemented as an `ndarray` ducktype, and so `MaskedArrays` can be substituted into almost any NumPy expression, for instance involving `np.sum`, `np.mean`, `np.concatenate`, and many others. `MaskedArray` supports the vast majority of NumPy's API methods and NumPy's vectorized arithmetic.

Historical Relationship to Numpy's `MaskedArray`s
-------------------------------------------------

This module should not be confused with NumPy's builtin `np.ma.MaskedArray` class, which has the same name and similar functionality but is otherwise independent. Historically, this module was developed by forking NumPy's `np.ma.MaskedArray` codebase and updating it to take advantage of NumPy's new and powerful `__array_function__` interface for creating `ndarray` ducktypes.

Quickstart
==========

This module introduces three main new types: 
 * `MaskedArray`, which behaves like NumPy's `ndarray` type but with masked elements
 * `MaskedScalar`, which behaves like NumPy's scalar types but which can be masked
 * `X`, a special variable representing a masked value for use in `MaskedArray` construction and assignment.

`MaskedArray` and `MaskedScalar` are implemented by storing two Numpy `ndarrays`(or scalars) with the same shape: One stores the data, which can be any NumPy datatype, and the other stores the mask, and is of boolean type. These are normally hidden from the user, but it is useful to know this to understand `MaskedArray` behavior.

MaskedArray construction
------------------------

`MaskedArray`s can be created in multiple ways. One way is using python lists,
in which case `X` can be used to signify masked elements::

```python
>>> a = MaskedArray([[1, X, 3], [X, X, 2], [X, 4, 1]])
>>> a
MaskedArray([[1, X, 3],
             [X, X, 2],
             [X, 4, 1]])
```

Another way is by specifying separate `data` and `mask` objects as the first two arguments of the constructor. The mask can be any object which numpy can cast to a boolean array and broadcast to the same shape as the data::

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

`MaskedArray` and `MaskedScalar` support two new attributes relative to `ndarray` which provide readonly access to the data and mask of the `MaskedArray`.

The `.filled` method  returns a readonly view of the `MaskedArray`s data as a plain `ndarray`, with all masked elements replaced by a fill value, which is 0 by default. The `.filled()` method is the primary way of converting a `MaskedArray` back to an `ndarray`. 

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

General Principles of Behavior
------------------------------

This module follows some general principles which should give some intuition for its behavior, which will be described in more detail in the reference section below.

First, in most computations any masked elements will be "skipped", which means the following. In scientific computing there are two general paradigms for the propagation of "masked" elements: The first is for operations involving any masked element, for instance a sum, to return a masked element. This is often called "sticky" masks. The other way is for such operations to exclude any masked elements from the computation, often called "skipping" those elements. This module uses the latter "skip" style, so for instance `np.sum` will return the sum only of the non-masked elements of an array. This applies in general to other NumPy API functions, for instance `np.dot`, `np.mean` and more.
    
 * some ufuncs (eg divide) will mask outputs which are out-of-domain (eg div by zero)
 * out arguments of ufuncs must be maskedarrays. This forces users to use safe code which doesn't lose the mask or risk using uninitialized data under mask.
 * methods which return `int` arrays meant for indexing (eg, argsort,
   nonzero) will return plain ndarrays. Otherwise API methods return maskedarrays.
 * masks sort as greater than all other values
 * np.any treats mask as False. np.all treats mask as True. This affects many other methods which depend on these. Discuss how to use `filled` on boolean masked arrays to choose other behaviors.
 * masked scalar with mask=True evaluates to false (important for if-statements)
   (not implemented yet... should this raise?)

Implementation Details and Cautionary Notes
-------------------------------------------

 * we store a separate boolean array for the mask (compare to R (sentinels) and Julia (similar)).
 * .filled and .mask return views, and Constructor views the inputs. Be careful!
 * MaskedScalars behave differently from numpy scalars in quite a few cases -- use filled a lot!
 * for complex datatypes, the masks of .real and .imag behave in a certain way


`MaskedArray` Reference
=======================

Construction
------------

Indexing
--------
