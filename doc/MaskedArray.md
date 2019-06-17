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

`MaskedArray` and `MaskedScalar` are implemented by storing a "data" NumPy `ndarray` of any NumPy datatype along with an auxiliary boolean "mask" `ndarray` of the same shape describing which elements of the data array are masked. Both the data and mask `ndarrays` are normally hidden from the user, but it is useful to know this to understand `MaskedArray` behavior.

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

Another way is by specifying separate `data` and `mask` objects as the first two arguments of the constructor. The mask can be any object which NumPy can cast to a boolean array and broadcast to the same shape as the data:

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

The `.mask` attribute gives a readonly view of the boolean `ndarray` representing which elements are masked.

```python
>>> a.mask
array([[False,  True, False],
       [ True,  True, False],
       [ True, False, False]])
```

Otherwise `MaskedArray` supports most of the basic attributes of `ndarrays` like `.shape`, `.dtype`, `.strides` and others, with a few exceptions such as `.base`.

MaskedScalars and the masked input marker `X`
---------------------------------------------

When operating on a `MaskedArray`, NumPy will return a `MaskedScalar` object in  situations which ordinarily return a NumPy scalar object, for instance when indexing an array to get a single element. While `MaskedScalar`s support almost all the methods and attributes of NumPy scalars there are some important differences. Notably, `MaskedScalar`s are not part of the NumPy scalar type hierarchy, and so for instance code of the form `isinstance(var, np.inexact)` will fail, and `np.isscalar` will return `False`. In such situations you can use `.filled` first to obtain a NumPy scalar.  Like NumPy scalars, `MaskedScalar`s are immutable (except for structured scalars) and can be used as dict keys.

The special `X` variable is not a `MaskedScalar`, and does not share attributes with NumPy types. Significantly, it does not have a `dtype`. It is a singleton of a special `MaskedX` class which acts as a masked-value signifier. When it is used in `MaskedArray` operations and assignments it is effectively promoted to a masked `MaskedScalar` of the appropriate `dtype`. As a convenience the `X` variable can be explicitly promoted to a masked `MaskedScalar` of a specified `dtype` by calling it with the `dtype` as argument. Note that the `repr` of `MaskedScalar`s whose mask is `True` is represented using this `X(dtype)` construction:

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
==============================

NumPy operations on `MaskedArray`s follow a few general principles of behavior.

Unary and Binary operations
---------------------------

For simple unary and binary operations, such as addition of two arrays, output elements will be masked at positions where either input element is masked:

```python
>>> a = MaskedArray([1, X, X])
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

Reductions
----------

For "reduction"-like operations involving many elements, such as summing an array, the masked elements do not contribute to the reduction. The result will be masked only if there were no unmasked input elements:

```python
>>> np.sum(MaskedArray([1, X, 2])
MaskedScalar(3)
>>> np.sum(MaskedArray([X, X, X], dtype=int))
X(int64)
```

A relevant detail is that `MaskedArray` reductions are implemented by replacing all masked elements by the appropriate identity element and then performing the NumPy reduction. For instance, `np.sum(arr)` is implemented using the "ufunc" reduction  `np.add.reduce(arr)` after replacing all masked elements by `np.add.identity`, which is 0. In consequence `MaskedArray` does not support reductions for "ufuncs" that do not have an identity element.

Truthiness of Masked Values
----------------------------

 When a `MaskedScalar` is tested for truth, for instance in if-statements, it will return `False` if masked, or the same as the corresponding NumPy scalar if not masked. To get masked values to test as `True`, use `scalar.filled(True)` first.

`np.nonzero` and `np.where` will similarly treat masked values as `False` or zero-like, and so do not return the indices of masked values.

`np.any` treats masked values as `False` (since it is implemented as `np.logical_or.reduce`, whose identity value is `False`), and `np.all` treats masked values as `True`. Use `.filled` to get different behaviors.

Fancy Indexing
--------------

`MaskedArray` supports fancy-indexing with boolean or integer array indexes, with some differences relative to `ndarray`.

For integer array fancy-indexing, the index array must be an `ndarray` and not a `MaskedArray`. Masked integer fancy indexes are not currently supported. Otherwise `MaskedArray` indexing with an integer array behaves just like for `ndarray`.

For boolean array fancy-indexing the index array may be a `MaskedArray`, in which case masked values are treated as `False`. Thus `arr[bool_index]` behaves the same as `arr[np.nonzero(bool_index)]`.

`MaskedArray` in indexing-like operations
-----------------------------------------

Unlike most other NumPy API functions, NumPy functions which perform indexing-like operations or produce index arrays, such as `np.take`, `np.nonzero`, `np.argsort` will generally use `ndarray`s instead of `MaskedArrays` for index variables.  For operations with integer-index inputs, eg `np.take`, the input index array should should be an `ndarray`. For operations with integer-index outputs, eg `np.nonzero`, the returned array will be an `ndarray`.

`out` arguments
---------------

For NumPy functions which takes an `out` argument, such as `np.sum` or `np.add`, if a `MaskedArray` is one of the inputs then the out argument must also be a `MaskedArray` if given, with exceptions for index-like operations. This is to prevent situations in which the mask is lost and uninitialized data is exposed to the user. 

Sorting masked values
---------------------

In sorting operations such as `np.sort`, `np.lexsort`, `np.argsort`, masked elements will be treated as greater than the greatest value of the `MaskedArray`'s dtype. In other words they will sort to the "end" of the sorted array.

For `np.max`, `np.min`, `np.argmin`, and `np.argmax`, non-masked values are returned before any masked value. In other words, these will only return a masked value, or index of a masked value, if all input values were masked. Note that the `arg*` methods will return `ndarrays` or `int` scalars, following the discussion of indexing-like operations above.

Views and Copies
----------------

It is important to understand when a NumPy operation returns a view or a copy, and this is doubly true for `MaskedArray`s as either the mask or data may be views. `MaskedArray` frequently uses views for performance reasons, but this means that with improper use the user can sometimes get access to the nonsense values at masked positions.

During `MaskedArray` construction, when supplying the data and mask as `ndarrays`,  in most cases these are viewed by the `MaskedArray`and will be modified by operations on the `MaskedArray`. This can lead to confusing or undesirable behavior if you subsequently use the data array, particularly as `MaskedArray` may fill the data `ndarray` with nonsense values at masked positions. To avoid this you can use the `copy=True` argument to the `MaskedArray` constructor.

The `.mask` attribute of a `MaskedArray` is a readonly view of the internal mask data, and will be updated if the `MaskedArray` is assigned to. The `.filled()` method on the other hand returns a copy of the data by default, but if the `view` argument is set to `True` it will return a readonly view. Use caution with this view, because subsequent operations with the `MaskedArray` may put nonsense values at masked positions of its internal data array.

Unlike `ndarray`s, `MaskedArray`s do not support a `.base` attribute which can be used to tell if an array is a view. However, it is possible to check the `.base` attribute of the `ndarray`s returned by `.mask`, or `.filled` with `view=True`.

Complex Values 
--------------

`MaskedArray`s of complex `dtype` have `.real` and `.imag` attributes, like `ndarrays`, which can also be obtained using `np.real` and `np.imag`. These return `MaskedArray`s whose data is a view of the original `MaskedArray`s real or imag part, similarly to `ndarray`s, but importantly the mask is a copy of the original `MaskedArray`s mask, and not a view. This means that operations which modify the mask of the `.real` or `.imag` part of a `MaskedArray` will not modify the original mask. Significantly, if you try to assign masked values to the real or imag part at locations which are unmasked in the original array, this may cause nonsense masked values to be visible from the original array. [TODO: Is it possible to avoid this without lots of complications?]

If you wish to make extensive use of masked complex values, or wish to mask the real and complex parts separately, it may be preferable to use plain `ndarray` instead of `MaskedArray` and use `nan` in place of a mask. You can then use the `np.nan*` functions which ignore `nan` elements similarly to how `MaskedArray`s ignore masked elements.

[TODO: Perhaps it would be better to make `.real` and `.imag` readonly views? That would break code that tries to modify them, but such code would often be broken anyway in the current implementation]

Structured dtypes
-----------------

`MaskedArray` does not support masking individual fields of a structured datatype, instead each structured element as a whole can be masked. If you wish to mask individual fields, as an alternative consider using the `MaskedArrayCollection` class which behaves similarly to structured arrays but allows the user to separately mask each named array in the collection.

Similarly to complex types, if you index a `MaskedArray` of structured dtype with a field name or names, the resulting `MaskedArray`s data is a view of the original `MaskedArray`s data, but the mask is a copy.

Appendix: Behavior Changes relative to np.ma.MaskedArray
========================================================

 * No support for masking individual fields of structured dtypes. Use an `ArrayCollection` of `MaskedArray`s instead, or (better) use `MaskedArrayCollection`.
 * No more `nomask`: the mask is always stored as a full ndarray.
 * No more returned `masked` singleton: Instead scalar values are returned as MaskedScalar instances. See comments in MaskedArray.py for discussion. A special masked singleton called `X` with no dtype exists for array construction purposes only.
 * No more `fill_value` stored with instances: You must always supply the desired fill as an argument to `filled()`
 * No more attempt to preserve the hidden data behind masked values. MaskedArray is free to modify these elements arbitrarily.
 * Features preserved from numpy's maskedarray: 1. The mask is not "sticky", it behaves as the "ignore" or "skipna" style described in the MaskedArray NEP. "na" style will not be supported. 2. Ufuncs replace out-of-domain inputs with mask.
 * out arguments to ufuncs/methods must be MaskedArrays too
 * more careful preservation of dtype (old MA would often cast to float64)
 * np.sort now doesn't mix maxval and masked vals up.
