#!/usr/bin/env python
from ArrayCollection import ArrayCollection
from MaskedArray import MaskedArray
import numpy as np

a = MaskedArray(np.arange(10), np.arange(10)%3)
b = MaskedArray(np.arange(10.) + 13, np.arange(10)%2)

c = ArrayCollection([('age', a), ('weight', b)])
print(repr(c))
c['age'] += 100
print(repr(c))

