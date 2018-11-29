#!/usr/bin/env python
from ArrayCollection import ArrayCollection
from MaskedArray import MaskedArray
from MaskedArrayCollection import MaskedArrayCollection
import numpy as np

# Tests for Masked ArrayCollections.
#
# First try: Simply make an arraycollection of MaskedArrays. Downside: this
# strategy does not give a "filled" method. Probably to get a masked
# ArrayCollection we should really subclass ArrayCollection to have a
# fill_value and a filled() method

a = MaskedArray(np.arange(10), np.arange(10)%3)
b = MaskedArray(np.arange(10.) + 13, np.arange(10)%2)

c = ArrayCollection([('age', a), ('weight', b)])
print(repr(c))
c['age'] += 100
print(repr(c))

# second try: Subclass of ArrayCollection

c = MaskedArrayCollection([('age', a), ('weight', b)])
print(repr(c))
c['age'] += 100
print(repr(c))
print(repr(c.filled()))
