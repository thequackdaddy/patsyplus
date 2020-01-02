from patsy import (EvalFactor, EvalEnvironment, design_matrix_builders, Term)
from patsyplus.varnames import var_names


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
