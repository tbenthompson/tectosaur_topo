import attr
import numpy as np

import tectosaur
from tectosaur.mesh.combined_mesh import CombinedMesh
from tectosaur.constraint_builders import continuity_constraints, \
    all_bc_constraints, free_edge_constraints
from tectosaur.interior import interior_integral
from tectosaur.ops.sparse_integral_op import SparseIntegralOp, FMMFarfieldBuilder
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sum_op import SumOp

from tectosaur_topo.solve import iterative_solve

import logging
logger = logging.getLogger(__name__)

def solve_topo(surf, fault, fault_slip, sm, pr, **kwargs):
    m = CombinedMesh([('surf', surf), ('fault', fault)])

    cs = continuity_constraints(
        m.get_piece_tris('surf'), m.get_piece_tris('fault'), m.pts
    )
    cs.extend(all_bc_constraints(
        m.get_start('fault'), m.get_past_end('fault'), fault_slip
    ))
    cs.extend(free_edge_constraints(m.get_piece_tris('surf')))

    mass_op = MassOp(kwargs.get('quad_mass_order', 3), m.pts, m.tris)

    T_op = SparseIntegralOp(
        kwargs.get('quad_vertadj_order', 6),
        kwargs.get('quad_far_order', 2),
        kwargs.get('quad_near_order', 5),
        kwargs.get('quad_near_threshold', 2.0),
        'elasticT3',
        [sm, pr],
        m.pts,
        m.tris,
        kwargs.get('float_type', np.float32),
        farfield_op_type = FMMFarfieldBuilder(
            kwargs.get('fmm_order', 150),
            kwargs.get('fmm_mac', 3.0),
            kwargs.get('pts_per_cell', 450)
        )
    )
    iop = SumOp([T_op, mass_op])

    soln = iterative_solve(
        iop,
        cs,
        tol = kwargs.get('solver_tol', 1e-8),
        prec = kwargs.get('preconditioner', 'none')
    )
    return m.pts, m.tris, m.get_start('fault'), soln

def evaluate_interior(obs_pts, m, soln, sm, pr, **kwargs):
    return -interior_integral(
        obs_pts, obs_pts, m, soln, 'elasticT3',
        kwargs.get('quad_far_order', 3),
        kwargs.get('quad_near_order', 8),
        [sm, pr], kwargs.get('float_type', np.float32),
        # fmm_params = [100, 3.0, 3000, 25]
    )
