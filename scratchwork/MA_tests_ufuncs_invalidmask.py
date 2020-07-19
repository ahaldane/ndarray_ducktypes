
    def test_divide_on_different_shapes(self):
        x = MaskedArray(np.arange(6, dtype=float))
        x.shape = (2, 3)
        y = MaskedArray(np.arange(3, dtype=float))

        z = x / y
        assert_masked_equal(z, MaskedArray([[X, 1., 1.], [X, 4., 2.5]]))

        z = x / y[None,:]
        assert_masked_equal(z, MaskedArray([[X, 1., 1.], [X, 4., 2.5]]))

        y = MaskedArray(np.arange(2, dtype=float))
        z = x / y[:, None]
        assert_masked_equal(z, MaskedArray([[X, X, X], [3., 4., 5.]]))

    def test_mixed_arithmetic(self):
        # Tests mixed arithmetics.
        na = np.array([1])
        ma = MaskedArray([1])
        assert_(isinstance(na + ma, MaskedArray))
        assert_(isinstance(ma + na, MaskedArray))

    def test_limits_arithmetic(self):
        tiny = np.finfo(float).tiny
        a = MaskedArray([tiny, 1. / tiny, 0.])
        assert_equal((a / 2).mask, [0, 0, 0])
        assert_equal((2 / a).mask, [0, 0, 1]) # changed from numpy

    def test_masked_singleton_arithmetic(self):
        # Tests some scalar arithmetics on MaskedArrays.
        # Masked singleton should remain masked no matter what
        xm = MaskedArray(0, mask=1)
        assert_((1 / MaskedArray(0)).mask)
        assert_((1 + xm).mask)
        assert_((-xm).mask)
        assert_(np.maximum(xm, xm).mask)
        assert_(np.minimum(xm, xm).mask)


    def test_scalar_arithmetic(self):
        # Next lines from ogigianl tests, no longer valid. XXX so this means .filled used to return a view?
        #x = MaskedArray(0, mask=0)
        #assert_equal(x.filled().ctypes.data, x.ctypes.data)

        # Make sure we don't lose the shape in some circumstances
        xm = MaskedArray((0, 0)) / 0.
        assert_equal(xm.shape, (2,))
        assert_equal(xm.mask, [1, 1])

    def test_mod(self):
        # Tests mod
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        mx = MaskedArray(x)
        my = MaskedArray(y)
        assert_equal(np.mod(x, y), np.mod(mx, my).filled(np.nan))

        test = np.mod(ym, xm)
        assert_equal(test.mask, xm.mask | ym.mask)
        test = np.mod(xm, ym)
        assert_equal(test.mask, xm.mask | ym.mask | (ym == 0).filled(True))

    def test_numpyarithmetics(self):
        # Check that the mask is not back-propagated when using numpy functions
        a = MaskedArray([-1, 0, 1, 2, X])
        assert_masked_equal(np.log(a), MaskedArray([X, X, 0, np.log(2), X]))
        assert_equal(a.mask, [0, 0, 0, 0, 1])

    def test_ndarray_mask(self):
        # Check that the mask of the result is a ndarray (not a MaskedArray...)
        a = MaskedArray([-1, 0, 1, 2, X])
        test = np.sqrt(a)
        control = MaskedArray([X, 0, 1, np.sqrt(2), X])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_(not isinstance(test.mask, MaskedArray))

    def test_inplace_division_scalar_float(self):
        # Test of inplace division
        (x, y, xm) = self.floatdata
        x = x.copy()
        xm = xm.copy()

        x /= 2.0
        assert_equal(x, y / 2.0)
        xm /= np.arange(10)
        assert_masked_equal(xm, MaskedArray([X, 1, X, 1, 1, 1, 1, 1, 1, 1.]))

    def test_inplace_division_array_float(self):
        # Test of inplace division
        (x, y, xm) = self.floatdata
        x = x.copy()
        xm = xm.copy()

        m = xm.mask
        a = MaskedArray(np.arange(10, dtype=float))
        a[-1] = X
        x /= a
        xm /= a
        assert_equal(x, y / a)
        assert_equal(xm, y / a)
        assert_equal(xm.mask, m | (a == 0).filled(True))

    def test_inplace_division_misc(self):
        x = MaskedArray([X, 1., 1., -2., pi / 2., 4., X, -10., 10., 1., 2., 3.])
        y = MaskedArray([5., 0., X, 2., -1., X, X, -10., 10., 1., 0., X])
        control = MaskedArray([X, X, X, -1, -pi/2, X, X, 1, 1, 1, X, X])

        z = xm / ym
        assert_masked_equal(z, control)

        xm = xm.copy()
        xm /= ym
        assert_masked_equal(xm, control)

    def test_inplace_floor_division_array_type(self):
        # Test of inplace division
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = MaskedArray(np.arange(10, dtype=t))
                a[-1] = X
                x //= a
                xm //= a
                assert_equal(x, y // a)
                assert_equal(xm, y // a)
                assert_equal(xm.mask, m | (a == t(0)).filled(True))

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_floor_division_array_type(self):
        # Test of inplace division
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = MaskedArray(np.arange(10, dtype=t))
                a[-1] = X
                x //= a
                xm //= a
                assert_equal(x, y // a)
                assert_equal(xm, y // a)
                assert_equal(xm.mask, m)

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_division_array_type(self):
        # Test of inplace division
        for t in self.othertypes:
            with suppress_warnings() as sup:
                sup.record(UserWarning)
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = MaskedArray(np.arange(10, dtype=t))
                a[-1] = X

                # May get a DeprecationWarning or a TypeError.
                #
                # This is a consequence of the fact that this is true divide
                # and will require casting to float for calculation and
                # casting back to the original type. This will only be raised
                # with integers. Whether it is an error or warning is only
                # dependent on how stringent the casting rules are.
                #
                # Will handle the same way.
                try:
                    x /= a
                    assert_equal(x, y / a)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)
                try:
                    xm /= a
                    assert_equal(xm, y / a)
                    assert_equal(xm.mask, m | (a == t(0)).filled(True))
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)

                if issubclass(t, np.integer):
                    assert_equal(len(sup.log), 2, "Failed on type=%s." % t)
                else:
                    assert_equal(len(sup.log), 0, "Failed on type=%s." % t)
