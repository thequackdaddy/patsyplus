# This module is used to determine variable names from a patsy DesignInfo

from patsy import (EvalEnvironment, EvalFactor, DesignMatrix, DesignInfo,
                   dmatrix, Term)
from patsy.eval import ast_names
from types import ModuleType
from collections import OrderedDict
import numpy as np
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
    elif isinstance(obj, Term):
        if hasattr(obj, 'factors') and len(obj.factors) > 0:
            return var_names(obj.factors[0], eval_env)
        else:
            return set()
    return set()


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


def partial(design_info, columns, product=False, intercept=False, eval_env=0):
    """Returns a partial prediction array where only the variables in the
    dict ``columns`` are tranformed per the :class:`DesignInfo`
    transformations. The terms that are not influenced by ``columns``
    return as zero.

    This is useful to perform a partial prediction on unseen data and to
    view marginal differences in factors.

    :arg columns: A dict with the keys as the column names for the marginal
    predictions desired and values as the marginal values to be predicted.

    :arg product: When `True`, the resturned numpy array represents the
    Cartesian product of the values ``columns``.

    :returns: A numpy array of the partial design matrix.
    """
    if not eval_env:
        eval_env = EvalEnvironment.capture(eval_env, reference=1)

    # We need to get rid of the non-callable items from the eval_env
    namespaces = [{key: value} for ns in eval_env._namespaces
                  for key, value in six.iteritems(ns)
                  if callable(value) or isinstance(value, ModuleType)]
    eval_env._namespaces = namespaces

    if product:
        columns = _column_product(columns)
    column_names = [k for k in six.iterkeys(columns)]
    for k in six.iterkeys(columns):
        rows = len(columns[k])
        break
    terms = terms_from_var_names(design_info, column_names, intercept,
                                 eval_env)
    parts = []
    for term, subterm in six.iteritems(design_info.term_codings):
        if term.name() in terms:
            if term.name() == 'Intercept':
                parts.append(np.ones((rows, 1)))
            else:
                di = design_info.subset([term.name()])
                parts.append(dmatrix(di, columns))
        else:
            num_columns = sum(s.num_columns for s in subterm)
            dm = np.zeros((rows, num_columns))
            parts.append(dm)
    return DesignMatrix(np.hstack(parts), design_info)


def _column_product(columns):
    from itertools import product
    cols = []
    values = []
    for col, value in six.iteritems(columns):
        cols.append(col)
        values.append(value)
    values = [value for value in product(*values)]
    values = [value for value in zip(*values)]
    return OrderedDict([(col, list(value))
                        for col, value in zip(cols, values)])


def terms_from_var_names(design_info, columns, intercept=False, eval_env=0):
    """Returns a subset of the design_info based on the names of columns being
    existing in the terms.

    :arg columns: A list of the variables to return a subset of terms for.

    :arg intercept: Whether to return the intercept term.

    :returns: A list of the terms be used in a subset.
    """
    if not eval_env:
        eval_env = EvalEnvironment.capture(eval_env, reference=1)

    # We need to get rid of the non-callable items from the eval_env
    namespaces = [{key: value} for ns in eval_env._namespaces
                  for key, value in six.iteritems(ns)
                  if callable(value) or isinstance(value, ModuleType)]
    eval_env._namespaces = namespaces

    columns = set(columns)
    parts = []
    if intercept:
        parts.append('Intercept')
    for term, subterm in six.iteritems(design_info.term_codings):
        term_vars = var_names(term, eval_env)
        if len(term_vars.intersection(columns)) > 0:
            parts.append(term.name())
    return parts


def index_from_var_names(design_info, columns, intercept=False, eval_env=0):
    """Returns an index of the design_info representing the DesignMatrix's
    columns that are repesented by the colum

    :arg columns: A list of the variables to return the index for.

    :arg intercept: Whether to include the intercept term.

    :returns: An index which is a numpy arrow which selects elements included
    in the column."""

    if not eval_env:
        eval_env = EvalEnvironment.capture(eval_env, reference=1)

    terms = terms_from_var_names(design_info, columns, intercept, eval_env)
    index = [True if key in terms else False
             for key in design_info.term_name_slices]
    return np.array(index)
