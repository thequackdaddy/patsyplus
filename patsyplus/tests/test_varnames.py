from patsy import (EvalFactor, EvalEnvironment, design_matrix_builders,
                   dmatrix, Term)
from patsyplus import var_names, terms_from_var_names, index_from_var_names
import pandas as pd
import numpy as np  # noqa
from numpy.testing import assert_allclose


def test_EvalFactor_varnames():
    e = EvalFactor('a + b')
    assert var_names(e) == {'a', 'b'}
    from patsy.state import stateful_transform

    class bar(object):
        pass

    foo = stateful_transform(lambda: "FOO-OBJ")
    zed = stateful_transform(lambda: "ZED-OBJ")
    bah = stateful_transform(lambda: "BAH-OBJ")
    eval_env = EvalEnvironment.capture(0)
    e = EvalFactor('foo(a) + bar.qux(b) + zed(bah(c))+ d')
    state = {}
    eval_env = EvalEnvironment.capture(0)
    passes = e.memorize_passes_needed(state, eval_env)
    print(passes)
    print(state)
    assert passes == 2
    for name in ["foo", "bah", "zed"]:
        assert state["eval_env"].namespace[name] is locals()[name]
    assert var_names(e, eval_env=eval_env) == {'a', 'b', 'c', 'd'}


def make_termlist(*entries):
    terms = []
    for entry in entries:
        terms.append(Term([EvalFactor(name) for name in entry]))
    return terms


def test_build_design_matrices_dtype():
    data = {"x": [1, 2, 3]}

    def iter_maker():
        yield data
    builder = design_matrix_builders([make_termlist("x")], iter_maker, 0)[0]

    assert var_names(builder) == [{'x'}]
    assert var_names(builder.terms[0]) == {'x'}


def test_terms_from_var_names():
    x = pd.DataFrame({'a': [1, 2, 3], 'b': ['M', 'M', 'F']})
    x['c'] = pd.Categorical(x['b'], categories=['M', 'F'])

    dm = dmatrix('a + b + c + np.log(a)', x)
    di = dm.design_info

    actual = terms_from_var_names(di, ['a'])
    expected = ['a', 'np.log(a)']
    assert actual == expected

    actual = terms_from_var_names(di, ['a'], intercept=True)
    expected = ['Intercept', 'a', 'np.log(a)']
    assert actual == expected


def test_terms_from_var_names():
    x = pd.DataFrame({'a': [1, 2, 3], 'b': ['M', 'M', 'F']})
    x['c'] = pd.Categorical(x['b'], categories=['M', 'F'])

    dm = dmatrix('a + b + c + np.log(a)', x)
    di = dm.design_info

    actual = terms_from_var_names(di, ['a'])
    expected = ['a', 'np.log(a)']
    assert actual == expected

    actual = terms_from_var_names(di, ['a'], intercept=True)
    expected = ['Intercept', 'a', 'np.log(a)']
    assert actual == expected


def test_index_from_var_names():
    x = pd.DataFrame({'a': [1, 2, 3], 'b': ['M', 'M', 'F']})
    x['c'] = pd.Categorical(x['b'], categories=['M', 'F'])

    dm = dmatrix('a + b + c + np.log(a)', x)
    di = dm.design_info

    actual = index_from_var_names(di, ['a'])
    slices = [di.slice(s) for s in ['a', 'np.log(a)']]
    expected = np.repeat([False], 5)
    expected[slices[0]] = True
    expected[slices[1]] = True
    assert_allclose(actual, expected)

    actual = index_from_var_names(di, ['a'], intercept=True)
    slices = [di.slice(s) for s in ['Intercept', 'a', 'np.log(a)']]
    expected = np.repeat([False], 5)
    expected[slices[0]] = True
    expected[slices[1]] = True
    expected[slices[2]] = True
    assert_allclose(actual, expected)
