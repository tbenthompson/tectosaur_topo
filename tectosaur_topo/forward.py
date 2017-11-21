import numpy as np
import pprint

from tectosaur.mesh.combined_mesh import CombinedMesh
from tectosaur.constraint_builders import continuity_constraints, \
    all_bc_constraints, free_edge_constraints
from tectosaur.ops.sparse_integral_op import SparseIntegralOp, FMMFarfieldBuilder
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sum_op import SumOp

import tectosaur_topo.cfg
import tectosaur_topo.solve

import logging
logger = logging.getLogger(__name__)

def forward(surf, fault, fault_slip, sm, pr, **kwargs):
    cfg = setup_solve_cfg(kwargs)
    tectosaur_topo.cfg.set_logging_levels(cfg['verbose'])
    logger.info('tectosaur_topo.solve_topo configuration: \n' + pprint.pformat(cfg))

    m = CombinedMesh([('surf', surf), ('fault', fault)])
    # TODO: Need to fix bugs with check_for_problems before using this.
    # tectosaur_topo.cfg.alert_mesh_problems(m)
    cs = build_constraints(m)

    lhs, rhs = setup_system(m, fault_slip, [sm, pr], cfg)

    soln = tectosaur_topo.solve.iterative_solve(
        lhs, cs, rhs = rhs, tol = cfg['solver_tol'], prec = cfg['preconditioner']
    )
    full_soln = np.concatenate((soln, fault_slip))
    return m.pts, m.tris, m.get_start('fault'), full_soln

def setup_solve_cfg(kwargs):
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
    tectosaur_topo.cfg.check_valid_options(kwargs, cfg)
    cfg.update(kwargs)
    return cfg

def build_constraints(m):
    cs = continuity_constraints(
        m.get_piece_tris('surf'), m.get_piece_tris('fault')
    )
    cs.extend(free_edge_constraints(m.get_piece_tris('surf')))
    return cs

def setup_system(m, fault_slip, k_params, cfg):
    mass_op = make_mass_op(m, cfg)
    Tuu_op = make_Top(m, k_params, cfg, 'surf', 'surf')
    Tus_op = make_Top(m, k_params, cfg, 'surf', 'fault')
    rhs = -Tus_op.dot(fault_slip)
    lhs = SumOp([Tuu_op, mass_op])
    return lhs, rhs

def make_Top(m, k_params, cfg, name1, name2):
    return SparseIntegralOp(
        cfg['quad_vertadj_order'], cfg['quad_far_order'],
        cfg['quad_near_order'], cfg['quad_near_threshold'],
        'elasticT3', k_params, m.pts, m.tris, cfg['float_type'],
        farfield_op_type = FMMFarfieldBuilder(
            cfg['fmm_order'], cfg['fmm_mac'], cfg['pts_per_cell']
        ),
        obs_subset = m.get_piece_tri_idxs(name1),
        src_subset = m.get_piece_tri_idxs(name2)
    )

def make_mass_op(m, cfg):
    return MassOp(cfg['quad_mass_order'], m.pts, m.tris[:m.get_past_end('surf')])
