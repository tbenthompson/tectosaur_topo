import logging
import numpy as np
import matplotlib.pyplot as plt

import tectosaur_topo as tt

from okada import make_meshes
import scipy.sparse

def add_hill(surf):
    hill_height = 1.0
    hill_R = 7.0
    C = [0,0]
    x, y = surf[0][:,0], surf[0][:,1]
    z = hill_height * np.exp(
        -(((x - C[0]) / hill_R) ** 2 + ((y - C[1]) / hill_R) ** 2)
    )
    surf[0][:,2] = z


def build_soln_to_obs_map(m, obs_pt_idxs, which_dims):
    tris = m.get_piece_tris('surf')
    surf_pts_map = np.unique(tris)
    surf_pts = m.pts[surf_pts_map]
    soln_to_obs = scipy.sparse.dok_matrix((
        surf_pts.shape[0] * len(which_dims),
        m.tris.shape[0] * 9
    ))
    done_pts = dict()
    for i in range(tris.shape[0]):
        for b in range(3):
            if tris[i,b] in done_pts:
                continue
            if tris[i,b] not in obs_pt_idxs:
                continue
            done_pts[tris[i,b]] = 1
            for d in which_dims:
                out_idx = tris[i,b] * len(which_dims) + d
                soln_to_obs[out_idx, i * 9 + b * 3 + d] = 1.0
    assert(soln_to_obs.shape[0] == soln_to_obs.getnnz())
    return soln_to_obs

def build_gfs(surf, fault, sm, pr, **kwargs):
    gfs = []
    for i in range(fault[1].shape[0]):
        for b in range(3):
            for d in range(3):
                print(i, b, d, fault[1].shape[0])
                slip = np.zeros((3, 3))
                slip[b,d] = 1.0
                slip = slip.flatten()
                subfault_tris = np.array([[0,1,2]])
                subfault_pts = np.array([fault[0][fault[1][i,:]]]).reshape((-1, 3))
                pts, tris, fault_start_idx, soln = tt.forward(
                    surf, (subfault_pts, subfault_tris), slip, sm, pr, **kwargs
                )
                gfs.append(soln[:(fault_start_idx * 9)])
    return gfs

def main():
    log_level = logging.INFO
    fault_L = 1.0
    top_depth = -0.5
    w = 10
    n_surf = 25
    n_fault = max(2, n_surf // 5)
    sm = 1.0
    pr = 0.25

    surf, fault = make_meshes(fault_L, top_depth, w, n_surf, n_fault)

    filename = 'examples/gfs.npy'
    # np.save(filename, build_gfs(surf, fault, sm, pr, log_level = log_level))
    gfs = np.load(filename).T

    add_hill(surf)

    slip = np.array([[1, 0, 0] * fault[1].size]).flatten()
    forward_system = tt.forward_assemble(surf, fault, sm, pr, log_level = log_level)
    pts, tris, fault_start_idx, soln = tt.forward_solve(
        forward_system, slip, preconditioner = 'ilu', log_level = log_level
    )
    # soln2 = np.zeros_like(soln)
    # for i in range(fault[1].shape[0]):
    #     for b in range(3):
    #         soln2 += np.concatenate((
    #             gfs.reshape((-1, fault[1].shape[0], 3, 3))[:, i, b, 0],
    #             slip
    #         ))
    # import ipdb
    # ipdb.set_trace()

    m = forward_system[0]
    which_dims = [0, 1]
    obs_pt_idxs = m.get_piece_pt_idxs('surf')
    soln_to_obs = build_soln_to_obs_map(forward_system[0], obs_pt_idxs, which_dims)
    u_hill = soln_to_obs.dot(soln)

    inv_surf = surf
    forward_system = tt.forward_assemble(inv_surf, fault, sm, pr, log_level = log_level)
    adjoint_system = tt.adjoint_assemble(forward_system, sm, pr, log_level = log_level)
    def mv(v):
        _,_,_,soln = tt.forward_solve(forward_system, v, log_level = log_level)
        return soln_to_obs.dot(soln)

    def rmv(v):
        rhs = soln_to_obs.T.dot(v)
        _,_,_,soln = tt.adjoint_solve(adjoint_system, rhs, log_level = log_level)
        return soln


    n_slip = fault[1].shape[0] * 9
    n_data = u_hill.shape[0]

    rand_data = np.random.rand(n_data)
    y1 = rmv(rand_data)
    y2 = gfs.T.dot(m.get_vector_subset(soln_to_obs.T.dot(rand_data), 'surf'))
    plt.plot(y1, 'b-')
    plt.plot(y2, 'r-')
    plt.show()
    import ipdb
    ipdb.set_trace()

    rand_slip = np.random.rand(n_slip)
    x1 = mv(rand_slip)
    x2 = soln_to_obs.dot(np.concatenate((gfs.dot(rand_slip), rand_slip)))
    plt.plot(x1, 'b-')
    plt.plot(x2, 'r-')
    plt.show()

    np.testing.assert_almost_equal(x1, x2)
    np.testing.assert_almost_equal(y1, y2)

    A = scipy.sparse.linalg.LinearOperator((n_data, n_slip), matvec = mv, rmatvec = rmv)
    inverse_soln = scipy.sparse.linalg.lsmr(A, u_hill, show = True)
    print(inverse_soln)

if __name__ == "__main__":
    main()
