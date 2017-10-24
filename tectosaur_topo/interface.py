import attr
import numpy as np
import pprint

import tectosaur
from tectosaur.mesh.combined_mesh import CombinedMesh
from tectosaur.constraint_builders import continuity_constraints, \
    all_bc_constraints, free_edge_constraints
from tectosaur.interior import interior_integral
from tectosaur.ops.sparse_integral_op import SparseIntegralOp, FMMFarfieldBuilder
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sum_op import SumOp

import tectosaur_topo.solve

import logging
logger = logging.getLogger(__name__)

def solve_topo(surf, fault, fault_slip, sm, pr, **kwargs):
    cfg = dict(
        quad_mass_order = 3,
        quad_vertadj_order = 6,
        quad_far_order = 2,
        quad_near_order = 5,
        quad_near_threshold = 2.0,
        float_type = np.float32,
        fmm_order = 150,
        fmm_mac = 3.0,
        pts_per_cell = 450,
        solver_tol = 1e-8,
        preconditioner = 'none',
        verbose = True
    )
    cfg.update(kwargs)

    if not cfg['verbose']:
        tectosaur_topo.logger.setLevel(logging.WARNING)
        tectosaur.logger.setLevel(logging.WARNING)

    logger.debug('tectosaur_topo.solve_topo configuration: \n' + pprint.pformat(cfg))

    m = CombinedMesh([('surf', surf), ('fault', fault)])

    cs = continuity_constraints(
        m.get_piece_tris('surf'), m.get_piece_tris('fault'), m.pts
    )
    cs.extend(all_bc_constraints(
        m.get_start('fault'), m.get_past_end('fault'), fault_slip
    ))
    cs.extend(free_edge_constraints(m.get_piece_tris('surf')))

    mass_op = MassOp(cfg['quad_mass_order'], m.pts, m.tris)

    T_op = SparseIntegralOp(
        cfg['quad_vertadj_order'], cfg['quad_far_order'],
        cfg['quad_near_order'], cfg['quad_near_threshold'],
        'elasticT3', [sm, pr], m.pts, m.tris, cfg['float_type'],
        farfield_op_type = FMMFarfieldBuilder(
            cfg['fmm_order'], cfg['fmm_mac'], cfg['pts_per_cell']
        )
    )
    iop = SumOp([T_op, mass_op])

    soln = tectosaur_topo.solve.iterative_solve(
        iop, cs, tol =  cfg['solver_tol'], prec = cfg['preconditioner']
    )
    return m.pts, m.tris, m.get_start('fault'), soln

def evaluate_interior(obs_pts, m, soln, sm, pr, **kwargs):
    cfg = dict(
        quad_far_order = 3,
        quad_near_order = 8,
        float_type = np.float32,
    )
    cfg.update(kwargs)
    logger.debug('tectosaur_topo.evaluate_interior configuration: \n' + pprint.pformat(cfg))

    return -interior_integral(
        obs_pts, obs_pts, m, soln, 'elasticT3',
        cfg['quad_far_order'], cfg['quad_near_order'], [sm, pr], cfg['float_type'],
        # fmm_params = [100, 3.0, 3000, 25]
    )
