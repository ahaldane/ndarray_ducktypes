ArrayCollection
===============

`ArrayCollection` is a NumPy `ndarray`-like type which stores an ordered set of
named NumPy arrays with a common shape but which may be different dtypes. It
supports NumPy-style indexing of the arrays as a group.

```python
>>> a = ArrayCollection({'age': [8, 10, 7, 8], 'weight': [10, 20, 13, 15]})
>>> a
ArrayCollection([( 8, 10), (10, 20), ( 7, 13), ( 8, 15)],
                dtype=[('age', '<i8'), ('weight', '<i8')])
>>> a[:2]
ArrayCollection([( 8, 10), (10, 20)],
                dtype=[('age', '<i8'), ('weight', '<i8')])
>>> a['age']
array([ 8, 10,  7,  8])
```

Comparison with similar Python types
------------------------------------

An ArrayCollection behaves very similarly to a NumPy structured `ndarray`, but
is more extensible, has additional features, and will often have better
performance because it uses a different memory layout. While NumPy's structured arrays are designed for lowlevel manipulation of binary blobs and for interfacing with C programs, in contrast `ArrayCollection` is meant for simple manipulation of multidimensional, multi-datatype datasets. In other words, `ArrayCollection` can be used as a "lite" version of Pandas `DataFrame` or Xarray `Dataset`. You are encouraged to consider using these other projects as they provide additional high-level features useful for tabular data analysis, such as named axes, which `ArrayCollection` lacks as it is only meant as a simple wrapper around a set of `ndarray`s.

Quickstart
==========

`ArrayCollection` Construction
------------------------------

`ArrayCollection` uses the same datatype specification as `NumPy`'s structured arrays.

Indexing and Assignment
-----------------------

Tips
----

One can use `np.broadcast_to` to broadcast the input arrays to the
same shape in a memory-saving way, though this makes the arrays readonly.
See the `np.broadcast_to` docstring for details.
