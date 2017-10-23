import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from tectosaur.util.timer import Timer
from tectosaur.constraints import build_constraint_matrix

import logging
logger = logging.getLogger(__name__)

def prec_solve_unconstrained(iop, cm, cmT):
    def prec_f(x):
        n = iop.shape[0]

        def mv(v):
            mv.iter += 1
            logger.debug('inner iteration # ' + str(mv.iter))
            return iop.ops[0].nearfield_no_correction_dot(v) + iop.ops[1].nearfield_dot(v)
        mv.iter = 0

        A = sparse.linalg.LinearOperator((n, n), matvec = mv)


        soln = sparse.linalg.lgmres(A, cm.dot(x), tol = 1e-2)
        return cmT.dot(soln[0])
    return prec_f

def prec_solve_constrained(iop, cm, cmT):
    def prec_f(x):
        n = cmT.shape[0]

        def mv(v):
            mv.iter += 1
            logger.debug('inner iteration # ' + str(mv.iter))
            cm_res = cm.dot(v)
            # iop_res = iop.dot(cm_res)
            iop_res = iop.ops[0].nearfield_no_correction_dot(cm_res) + iop.ops[1].nearfield_dot(cm_res)
            out = cmT.dot(iop_res)
            return out / mv.factor
        mv.iter = 0
        mv.factor = 1.0
        mv.factor = np.mean(np.abs(mv(np.ones(n))))
        A = sparse.linalg.LinearOperator((n, n), matvec = mv)

        soln = sparse.linalg.lgmres(A, x, tol = 1e-2)
        return soln[0]
    return prec_f

def prec_diagonal_matfree(n, mv):
    P = 1.0 / (mv(np.ones(n)))
    factor = np.mean(np.abs(P))
    P /= factor
    def prec_f(x):
        return P * x
    return prec_f

def prec_opposite_order(iop2, cm, cmT):
    def prec_f(x):
        # return cmT.dot(iop2.dot(cm.dot(x)))
        return iop2.dot(x)
    return prec_f

def prec_identity():
    def prec_f(x):
        return x
    return prec_f

def prec_spilu(Aish):
    P = scipy.sparse.linalg.spilu(Aish)
    def prec_f(x):
        return P.solve(x)
    return prec_f

def iterative_solve(iop, constraints, rhs = None, tol = 1e-8, iop2 = None):
    timer = Timer(logger = logger)

    cm, c_rhs = build_constraint_matrix(constraints, iop.shape[1])

    cm = cm.tocsr()
    cmT = cm.T
    timer.report('assemble constraint matrix')

    near_reduced = None
    for M in iop.ops[0].nearfield.mat_no_correction[:4]:
        M_scipy = M.to_bsr().to_scipy()
        M_red = cmT.dot(M_scipy.dot(cm))
        if near_reduced is None:
            near_reduced = M_red
        else:
            near_reduced += M_red
    timer.report('reduce nearfield')

    if rhs is None:
        rhs_constrained = cmT.dot(-iop.dot(c_rhs))
    else:
        rhs_constrained = cmT.dot(rhs - iop.dot(c_rhs))
    n = rhs_constrained.shape[0]
    timer.report('constrain rhs')

    def mv(v):
        t = Timer(logger = logger)
        mv.iter += 1
        logger.debug('' + str(mv.iter))
        cm_res = cm.dot(v)
        iop_res = iop.dot(cm_res)
        out = cmT.dot(iop.dot(cm.dot(v)))
        t.report('iteration # ' + str(mv.iter))
        return out
    mv.iter = 0
    A = sparse.linalg.LinearOperator((n, n), matvec = mv)
    timer.report('setup linear operator')

    # P = prec_identity()
    # P = prec_diagonal_matfree(n, mv)
    # P = prec_solve_constrained(iop, cm, cmT)
    # P = prec_solve_unconstrained(iop, cm, cmT)
    # P = prec_opposite_order(iop2, cm, cmT)
    P = prec_spilu(near_reduced)
    M = sparse.linalg.LinearOperator((n, n), matvec = P)
    timer.report("setup preconditioner")

    def report_res(R):
        logger.debug('residual: ' + str(R))
        pass

    soln = sparse.linalg.gmres(
        A, rhs_constrained, M = M, tol = tol, callback = report_res, restart = 500
    )
    timer.report("GMRES")

    return cm.dot(soln[0]) + c_rhs
