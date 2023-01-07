import pytest

from numpy.testing import assert_array_equal
from numpy.testing.overrides import get_overridable_numpy_ufuncs
from test_MaskedArray import assert_masked_equal
from ndarray_ducktypes.MaskedArray import MaskedArray, _masked_ufuncs
import numpy as np


def test_all_ufuncs_overridden():
    # If any new ufuncs get created, we need a matching masked version
    # that can handle the masks properly.
    ufuncs = set(x.__name__ for x in get_overridable_numpy_ufuncs())
    ma_funcs = set(x.__name__ for x in _masked_ufuncs)
    # TODO: implement these ufuncs
    # - divmod (two outputs)
    # - frexp (two outputs)
    # - matmul (different shaped output)
    # - modf (two outputs)
    currently_unsupported = {'divmod', 'frexp', 'matmul', 'modf'}
    diff = ufuncs - ma_funcs - currently_unsupported
    assert not diff, f"These ufuncs should be implemented {sorted(diff)}"


@pytest.fixture
def xm():
    return MaskedArray([-2, -1, 0, 1, 2], mask=[False, True, True, True, False])

@pytest.fixture()
def xm_pos(xm):
    return np.abs(xm) + 1


def test_expm1(xm):
    assert_masked_equal(np.expm1(xm), np.exp(xm) - 1)


def test_exp2(xm_pos):
    assert_masked_equal(np.exp2(xm_pos), 2 ** xm_pos)


def test_cbrt(xm_pos):
    assert_masked_equal(np.cbrt(xm_pos), xm_pos ** (1/3))


def test_square(xm):
    assert_masked_equal(np.square(xm), xm ** 2)


def test_copysign(xm):
    assert_masked_equal(np.copysign(xm, 1), np.abs(xm))
    assert_masked_equal(np.copysign(xm, -1), -1*np.abs(xm))
    assert_masked_equal(np.copysign(1, xm), xm / np.abs(xm))


def test_sign(xm):
    assert_masked_equal(np.sign(xm), xm / np.abs(xm))


def test_signbit(xm):
    assert_masked_equal(np.signbit(xm), xm < 0)
    assert np.signbit(xm).dtype == np.dtype(bool)


def test_degree_radians(xm):
    assert_masked_equal(np.deg2rad(xm), xm * np.pi / 180)
    assert_masked_equal(np.rad2deg(xm), xm * 180 / np.pi)
    assert_masked_equal(np.deg2rad(np.rad2deg(xm)), xm)

    assert_masked_equal(np.radians(xm), np.deg2rad(xm))
    assert_masked_equal(np.degrees(xm), np.rad2deg(xm))


def test_float_power(xm_pos):
    assert_masked_equal(np.float_power(xm_pos, xm_pos), xm_pos ** xm_pos)


def test_fmin_fmax(xm):
    expected = MaskedArray([-2, -1, 0, 0, 0], mask=xm.mask)
    assert_masked_equal(np.fmin(xm, 0), expected)
    assert_masked_equal(np.fmin(xm, xm), xm)

    expected = MaskedArray([0, 0, 0, 1, 2], mask=xm.mask)
    assert_masked_equal(np.fmax(xm, 0), expected)
    assert_masked_equal(np.fmax(xm, xm), xm)


def test_gcd(xm):
    assert_masked_equal(np.gcd(5*xm, 20), 5*np.abs(xm))


def test_heaviside(xm):
    # zeros except for the final entry
    # TODO: Add some tests for various combinations of the mask and zero values
    expected = 0 * xm.copy()
    expected[-1] = 1
    assert_masked_equal(np.heaviside(xm, 1), expected)


def test_isnat():
    x = MaskedArray(["NaT", "NaT", "2000-01-01", "2000-01-02"],
                     mask=[False, True, True, False], dtype=np.datetime64)
    expected = MaskedArray([True, True, False, False],
                            mask=[False, True, True, False])
    assert_masked_equal(np.isnat(x), expected)


def test_lcm(xm):
    assert_masked_equal(np.lcm(xm, 7), 7*np.abs(xm))


def test_ldexp(xm_pos):
    assert_masked_equal(np.ldexp(xm_pos, xm_pos), xm_pos * 2 ** xm_pos)
    # TODO: Add this in once frexp is implemented
    # assert_masked_equal(np.ldexp(*np.frexp(xm)))


def test_log1p(xm_pos):
    assert_masked_equal(np.log1p(xm_pos), np.log(1 + xm_pos))


def test_left_right_shift():
    x = MaskedArray([0, 1, 2, 3], mask=[True, True, False, False], dtype=np.uint8)
    y = MaskedArray([1, 2, 3, 4], mask=[False, True, True, False], dtype=np.uint8)
    expected = MaskedArray([0, 0, 0, 48], mask=[True, True, True, False], dtype=np.uint8)
    assert_masked_equal(np.left_shift(x, y), expected)
    assert_masked_equal(np.left_shift(x, y), x << y)

    expected = MaskedArray([0, 0, 0, 1], mask=[True, True, True, False], dtype=np.uint8)
    assert_masked_equal(np.right_shift(8 * x, y), expected)
    assert_masked_equal(np.right_shift(8 * x, y), (8 * x) >> y)


def test_logaddexp(xm, xm_pos):
    assert_masked_equal(np.logaddexp(xm, 2*xm), np.log(np.exp(xm) + np.exp(2*xm)))

    # base 2
    assert_masked_equal(np.logaddexp2(xm_pos, 2*xm_pos), np.log2(2 ** xm_pos + 2 ** (2 * xm_pos)))


def test_nextafter(xm):
    eps = np.finfo(np.float64).eps
    assert_masked_equal(np.nextafter(xm, 2), eps + xm)


def test_positive(xm):
    assert_masked_equal(np.positive(xm), xm)
    assert_masked_equal(np.positive(xm), +xm)


def test_reciprocal(xm):
    # NOTE: reciprocal is meant to work with floats
    assert_masked_equal(np.reciprocal(xm.astype(float)), 1 / xm)


def test_rint(xm):
    expected = MaskedArray([-1, 0, 0, 0, 3], mask=[False, True, True, True, False])
    assert_masked_equal(np.rint(xm + 0.7), expected)


def test_spacing():
    x = MaskedArray([1, 2, np.inf, np.inf, np.nan, np.nan], mask=[False, True] * 3)
    eps = np.finfo(np.float64).eps
    expected = MaskedArray([eps, eps, np.nan, np.nan, np.nan, np.nan], mask=[False, True] * 3)
    assert_masked_equal(np.spacing(x), expected)


def test_trunc(xm):
    assert_masked_equal(np.trunc(xm * 1.2), xm)
    assert_masked_equal(np.trunc(xm / 1.2), xm / np.abs(xm))
