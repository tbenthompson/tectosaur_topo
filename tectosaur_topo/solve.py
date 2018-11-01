import numpy as np
import scipy.sparse.linalg

import tectosaur_topo
from tectosaur.util.timer import Timer

import logging
logger = logging.getLogger(__name__)

defaults = dict(
    solver_tol = 1e-8,
    log_level = logging.DEBUG
)

def forward_solve(system, fault_slip, **kwargs):
    cfg = tectosaur_topo.cfg.setup_cfg(defaults, kwargs)
    m, lhs, rhs_op, cm, prec = system
    rhs = rhs_op.dot(fault_slip)
    soln = iterative_solve(lhs, cm, rhs, prec, cfg)
    full_soln = np.concatenate((soln, fault_slip))
    return m.pts, m.tris, m.get_start('fault'), full_soln

def adjoint_solve(system, surf_disp, **kwargs):
    cfg = tectosaur_topo.cfg.setup_cfg(defaults, kwargs)
    m, lhs, post_op, cm, prec = system
    rhs = m.get_dofs(surf_disp, 'surf')
    soln = iterative_solve(lhs, cm, rhs, prec, cfg)
    to_slip = post_op.dot(soln)
    return m.pts, m.tris, m.get_start('fault'), to_slip

def iterative_solve(iop, cm, rhs, prec, cfg):
    rhs_constrained = cm.T.dot(rhs)
    n = rhs_constrained.shape[0]

    def mv(v):
        t = Timer(output_fnc = logger.debug)
        mv.iter += 1
        out = cm.T.dot(iop.dot(cm.dot(v)))
        t.report('iteration # ' + str(mv.iter))
        return out
    mv.iter = 0
    A = scipy.sparse.linalg.LinearOperator((n, n), matvec = mv)
    M = scipy.sparse.linalg.LinearOperator((n, n), matvec = prec)

    soln = scipy.sparse.linalg.gmres(
        A, rhs_constrained, M = M, tol = cfg['solver_tol'],
        callback = report_res, restart = 500
    )

    out = cm.dot(soln[0])
    return out

def report_res(R):
    logger.debug('residual: ' + str(R))
    pass
