import attr
import numpy as np

import tectosaur
from tectosaur.mesh.combined_mesh import CombinedMesh
from tectosaur.constraint_builders import continuity_constraints, \
    all_bc_constraints, free_edge_constraints
# from tectosaur.interior import interior_integral
from tectosaur.ops.sparse_integral_op import SparseIntegralOp, FMMFarfieldBuilder
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sum_op import SumOp

from tectosaur_topo.solve import iterative_solve
logger = tectosaur.setup_logger(__name__)

@attr.s
class Result:
    pass

def solve_topo(surf, fault, fault_slip, should_plot = False):
    sm = 1.0
    pr = 0.25
    float_type = np.float32
    k_params = [sm, pr]

    m = CombinedMesh([('surf', surf), ('fault', fault)])

    cs = continuity_constraints(
        m.get_piece_tris('surf'), m.get_piece_tris('fault'), m.pts
    )
    cs.extend(all_bc_constraints(
        m.get_start('fault'), m.get_past_end('fault'), fault_slip
    ))
    cs.extend(free_edge_constraints(m.get_piece_tris('surf')))

    T_op = SparseIntegralOp(
        6, 2, 5, 2.0,
        'elasticT3', k_params, m.pts, m.tris,
        float_type,
        farfield_op_type = FMMFarfieldBuilder(150, 3.0, 450)
    )
    mass_op = MassOp(3, m.pts, m.tris)
    iop = SumOp([T_op, mass_op])
    soln = iterative_solve(iop, cs)

    surf_pts, surf_disp = m.extract_pts_vals('surf', soln)

    if should_plot:
        plot_result(surf_pts, m.get_piece_tris('surf'), surf_disp)

    return surf_pts, surf_disp
#
# def test_okada(n_surf):
#     sm = 1.0
#     pr = 0.25
#     k_params = [sm, pr]
#     fault_L = 1.0
#     top_depth = -0.5
#     load_soln = False
#     float_type = np.float32
#     n_fault = max(2, n_surf // 5)
#
#     timer = Timer()
#     all_mesh, surface_tris, fault_tris = make_meshes(fault_L, top_depth, n_surf, n_fault)
#     timer.report('make meshes')
#     logger.info('n_elements: ' + str(all_mesh[1].shape[0]))
#
#     # to check that the fault-surface alignment is correct
#     # plt.triplot(all_mesh[0][:,0], all_mesh[0][:,1], surface_tris, linewidth = 0.3)
#     # plt.triplot(all_mesh[0][:,0], all_mesh[0][:,1], fault_tris, linewidth = 0.3)
#     # plt.plot(all_mesh[0][:,0], all_mesh[0][:,1], 'o', markersize = 3)
#     # plt.show()
#
#     cs = build_constraints(surface_tris, fault_tris, all_mesh[0])
#     timer.report("Constraints")
#
#     surface_pt_idxs = np.unique(surface_tris)
#     obs_pts = all_mesh[0][surface_pt_idxs,:]
#
#     T_op = SparseIntegralOp(
#         6, 2, 5, 2.0,
#         'elasticT3', k_params, all_mesh[0], all_mesh[1],
#         float_type,
#         farfield_op_type = FMMFarfieldBuilder(150, 3.0, 450)
#     )
#     timer.report("Integrals")
#
#     mass_op = MassOp(3, all_mesh[0], all_mesh[1])
#     iop = SumOp([T_op, mass_op])
#     timer.report('mass op/sum op')
#
#     soln = iterative_solve(iop, cs, tol = 1e-6)
#     # soln = direct_solve(iop, cs)
#     timer.report("Solve")
#
#     disp = soln[:iop.shape[0]].reshape(
#         (int(iop.shape[0] / 9), 3, 3)
#     )[:-fault_tris.shape[0]]
#     vals = [None] * surface_pt_idxs.shape[0]
#     for i in range(surface_tris.shape[0]):
#         for b in range(3):
#             idx = surface_tris[i, b]
#             # if vals[idx] is not None:
#             #     np.testing.assert_almost_equal(vals[idx], disp[i,b,:], 9)
#             vals[idx] = disp[i,b,:]
#     vals = np.array(vals)
#     timer.report("Extract surface displacement")
#
#     u = okada_exact(obs_pts, fault_L, top_depth, sm, pr)
#     # plot_results(obs_pts, surface_tris, u, vals)
#     # plot_interior_displacement(fault_L, top_depth, k_params, all_mesh, soln)
#     return print_error(obs_pts, u, vals)
