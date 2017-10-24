import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from tectosaur.util.timer import Timer
from tectosaur.constraints import build_constraint_matrix

import logging
logger = logging.getLogger(__name__)

def prec_diagonal_matfree(n, mv):
    P = 1.0 / (mv(np.ones(n)))
    factor = np.mean(np.abs(P))
    P /= factor
    def prec_f(x):
        return P * x
    return prec_f

def prec_identity():
    def prec_f(x):
        return x
    return prec_f

def prec_spilu(cm, cmT, iop):
    near_reduced = None
    for M in iop.ops[0].nearfield.mat_no_correction[:4]:
        M_scipy = M.to_bsr().to_scipy()
        M_red = cmT.dot(M_scipy.dot(cm))
        if near_reduced is None:
            near_reduced = M_red
        else:
            near_reduced += M_red
    P = scipy.sparse.linalg.spilu(near_reduced)
    def prec_f(x):
        return P.solve(x)
    return prec_f

def iterative_solve(iop, constraints, rhs = None, **kwargs):
    timer = Timer(logger = logger)

    cm, c_rhs = build_constraint_matrix(constraints, iop.shape[1])
    cm = cm.tocsr()
    cmT = cm.T
    timer.report('assemble constraint matrix')

    if rhs is None:
        rhs_constrained = cmT.dot(-iop.dot(c_rhs))
    else:
        rhs_constrained = cmT.dot(rhs - iop.dot(c_rhs))
    n = rhs_constrained.shape[0]
    timer.report('constrain rhs')

    def mv(v):
        t = Timer(logger = logger)
        mv.iter += 1
        cm_res = cm.dot(v)
        iop_res = iop.dot(cm_res)
        out = cmT.dot(iop.dot(cm.dot(v)))
        t.report('iteration # ' + str(mv.iter))
        return out
    mv.iter = 0
    A = sparse.linalg.LinearOperator((n, n), matvec = mv)
    timer.report('setup linear operator')

    prec = kwargs.get('prec', 'none')
    if prec == 'diag':
        P = prec_diagonal_matfree(n, mv)
    elif prec == 'ilu':
        P = prec_spilu(cm, cmT, iop)
    else:
        P = prec_identity()

    M = sparse.linalg.LinearOperator((n, n), matvec = P)
    timer.report("setup preconditioner")

    def report_res(R):
        logger.debug('residual: ' + str(R))
        pass

    soln = sparse.linalg.gmres(
        A, rhs_constrained, M = M,
        tol = kwargs.get('tol', 1e-8),
        callback = report_res, restart = 500
    )
    timer.report("GMRES")

    return cm.dot(soln[0]) + c_rhs
