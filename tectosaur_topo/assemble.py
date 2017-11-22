import numpy as np

from tectosaur.mesh.combined_mesh import CombinedMesh
from tectosaur.constraint_builders import continuity_constraints, \
    all_bc_constraints, free_edge_constraints
from tectosaur.ops.sparse_integral_op import SparseIntegralOp, FMMFarfieldBuilder
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sum_op import SumOp
from tectosaur.ops.neg_op import NegOp

import tectosaur_topo.cfg

import logging
logger = logging.getLogger(__name__)

defaults = dict(
    quad_mass_order = 3,
    quad_vertadj_order = 6,
    quad_far_order = 2,
    quad_near_order = 5,
    quad_near_threshold = 2.0,
    float_type = np.float32,
    use_fmm = True,
    fmm_order = 150,
    fmm_mac = 3.0,
    pts_per_cell = 450,
    log_level = logging.DEBUG
)

def forward_assemble(surf, fault, sm, pr, **kwargs):
    cfg = tectosaur_topo.cfg.setup_cfg(defaults, kwargs)

    m = CombinedMesh([('surf', surf), ('fault', fault)])

    # TODO: Need to fix bugs with check_for_problems before using this.
    # tectosaur_topo.cfg.alert_mesh_problems(m)

    cs = build_constraints(m)

    lhs, rhs_op = forward_system(m, [sm, pr], cfg)
    return m, lhs, rhs_op, cs

def build_constraints(m):
    cs = continuity_constraints(
        m.get_piece_tris('surf'), m.get_piece_tris('fault')
    )
    cs.extend(free_edge_constraints(m.get_piece_tris('surf')))
    return cs

def forward_system(m, k_params, cfg):
    mass_op = make_mass_op(m, cfg)
    Tuu_op = make_integral_op(m, 'elasticT3', k_params, cfg, 'surf', 'surf')
    lhs = SumOp([Tuu_op, mass_op])
    rhs_op = NegOp(make_integral_op(m, 'elasticT3', k_params, cfg, 'surf', 'fault'))
    return lhs, rhs_op

def make_mass_op(m, cfg):
    return MassOp(cfg['quad_mass_order'], m.pts, m.tris[:m.get_past_end('surf')])

def adjoint_assemble(forward_system, sm, pr, **kwargs):
    cfg = tectosaur_topo.cfg.setup_cfg(defaults, kwargs)
    m, forward_lhs, forward_rhs_op, cs = forward_system
    lhs, post_op = adjoint_system(m, [sm, pr], cfg)
    return m, lhs, post_op, cs

def adjoint_system(m, k_params, cfg):
    post_op = NegOp(make_integral_op(m, 'elasticA3', k_params, cfg, 'fault', 'surf'))
    lhs = make_integral_op(m, 'elasticA3', k_params, cfg, 'surf', 'surf')
    return lhs, post_op

def make_integral_op(m, k_name, k_params, cfg, name1, name2):
    if cfg['use_fmm']:
        farfield = FMMFarfieldBuilder(
            cfg['fmm_order'], cfg['fmm_mac'], cfg['pts_per_cell']
        )
    else:
        farfield = None
    return SparseIntegralOp(
        cfg['quad_vertadj_order'], cfg['quad_far_order'],
        cfg['quad_near_order'], cfg['quad_near_threshold'],
        k_name, k_params, m.pts, m.tris, cfg['float_type'],
        farfield_op_type = ,
        obs_subset = m.get_piece_tri_idxs(name1),
        src_subset = m.get_piece_tri_idxs(name2)
    )

