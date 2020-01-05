import pandas as pd
from patsy import dmatrix
from patsyplus.varnames import _column_product, partial
from collections import OrderedDict
import numpy as np
from numpy.testing import assert_allclose


def test_column_product():
    x = OrderedDict([('a', [1, 2, 3]), ('b', ['a', 'b'])])
    y = OrderedDict([('a', [1, 1, 2, 2, 3, 3]),
                     ('b', ['a', 'b', 'a', 'b', 'a', 'b'])])
    x = _column_product(x)
    assert x['a'] == y['a']
    assert x['b'] == y['b']


def test_partial_from_patsy():
    x = pd.DataFrame({'a': [1, 2, 3], 'b': ['M', 'M', 'F']})
    x['c'] = pd.Categorical(x['b'], categories=['M', 'F'])

    dm = dmatrix('a + b + c', x)
    di = dm.design_info

    actual = partial(di, {'a': pd.Series([2, 3, 4])})
    column = di.slice('a').start
    expected = np.zeros((3, 4))
    expected[:, column] = [2, 3, 4]
    assert_allclose(actual, expected)

    actual = partial(di, {'b': pd.Series(['M', 'F'])})
    column = di.slice('b').start
    expected = np.zeros((2, 4))
    expected[:, column] = [1, 0]
    assert_allclose(actual, expected)

    actual = partial(di, {'c': pd.Categorical(pd.Series(['M', 'F']),
                                              categories=['M', 'F'])})
    column = di.slice('c').start
    expected = np.zeros((2, 4))
    expected[:, column] = [0, 1]
    assert_allclose(actual, expected)
