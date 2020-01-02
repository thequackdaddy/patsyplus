# This module is used to determine variable names from a patsy model

from patsy import EvalEnvironment, EvalFactor, DesignMatrix, DesignInfo
from patsy.eval import ast_names
import six

_builtins_dict = {}
six.exec_("from patsy.builtins import *", {}, _builtins_dict)
# This is purely to make the existence of patsy.builtins visible to systems
# like py2app and py2exe. It's basically free, since the above line guarantees
# that patsy.builtins will be present in sys.modules in any case.
import patsy.builtins  # noqa


def var_names(obj, eval_env=0):
    if not eval_env:
        eval_env = EvalEnvironment.capture(eval_env, reference=1)
    if isinstance(obj, EvalFactor):
        return var_names_eval_factor(obj, eval_env)
    elif isinstance(obj, DesignMatrix):
        return var_names_design_info(obj.design_info, eval_env)
    elif isinstance(obj, DesignInfo):
        return var_names_design_info(obj, eval_env)


def var_names_design_info(design_info, eval_env=0):
    if not eval_env:
        eval_env = EvalEnvironment.capture(eval_env, reference=1)
    terms = design_info.terms
    varnames = []
    for term in terms:
        varset = set()
        for factor in term.factors:
            varset = varset.union(var_names(factor, eval_env))
        varnames.append(varset)
    return varnames


def var_names_eval_factor(eval_factor, eval_env=0):
    """Returns a set of variable names that are used in the
    :class:`EvalFactor`, but not available in the current evalulation
    environment. These are likely to be provided by data.
    :arg eval_env: Either a :class:`EvalEnvironment` which will be used to
      look up any variables referenced in the :class:`EvalFactor` that
      cannot be found in :class:`EvalEnvironment`, or else a depth
      represented as an integer which will be passed to
      :meth:`EvalEnvironment.capture`. ``eval_env=0`` means to use the
      context of the function calling :meth:`var_names` for lookups.
      If calling this function from a library, you probably want
      ``eval_env=1``, which means that variables should be resolved in
      *your* caller's namespace.
    :returns: A set of strings of the potential variable names.
    """
    if not eval_env:
        eval_env = EvalEnvironment.capture(eval_env, reference=1)
    eval_env = eval_env.with_outer_namespace(_builtins_dict)
    env_namespace = eval_env.namespace
    names = set(name for name in ast_names(eval_factor.code)
                if name not in env_namespace)
    return names
