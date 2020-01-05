import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix, DesignMatrix
from patsyplus import partial
import pytest


@pytest.fixture
def basic_ols():
    np.random.seed(9876789)
    nsample = 1000
    data = pd.DataFrame({'a': np.linspace(1, 10, nsample),
                         'b': np.random.choice([1, 2, 3], nsample),
                         'c': pd.Categorical(
                          np.random.choice(['M', 'F', 'X'], nsample),
                          ['M', 'F', 'X'])})
    X = dmatrix('a + np.log(a) + C(b) + c', data)
    beta = np.array([1, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3])
    e = np.random.normal(size=nsample)
    y = np.dot(X, beta) + e
    model = sm.OLS.from_formula((y, X), data)
    results = model.fit()
    return results


def test_predict(basic_ols):
    design_info = basic_ols.model.data.design_info
    X = np.zeros((2, 7))
    X[:, design_info.slice('np.log(a)')] = np.log([1, 2])[:, np.newaxis]
    X[:, design_info.slice('a')] = np.array([1, 2])[:, np.newaxis]
    X = DesignMatrix(X, design_info)
    expected = basic_ols.get_prediction(X, transform=False)
    actual = basic_ols.get_prediction(partial(design_info, {'a': [1, 2]}),
                                      transform=False)
    assert_allclose(expected.predicted_mean, actual.predicted_mean)
    assert_allclose(expected.conf_int(), actual.conf_int())

    X[:, 0] = 1
    expected = basic_ols.get_prediction(X, transform=False)
    actual = basic_ols.get_prediction(partial(design_info, {'a': [1, 2]},
                                              intercept=True),
                                      transform=False)
