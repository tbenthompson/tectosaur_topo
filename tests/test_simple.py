import numpy as np
import matplotlib.pyplot as plt

import okada_wrapper

import tectosaur_topo as tt
import tectosaur.mesh.mesh_gen as mesh_gen

def okada_exact(obs_pts, fault_L, top_depth, sm, pr):
    lam = 2 * sm * pr / (1 - 2 * pr)
    alpha = (lam + sm) / (lam + 2 * sm)
    print(lam, sm, pr, alpha)

    n_pts = obs_pts.shape[0]
    u = np.empty((n_pts, 3))
    for i in range(n_pts):
        pt = obs_pts[i, :]
        [suc, uv, grad_uv] = okada_wrapper.dc3dwrapper(
            alpha, pt, 0.5, 90.0,
            [-fault_L, fault_L], [-1.0, 0.0], [1.0, 0.0, 0.0]
        )
        if suc != 0:
            u[i, :] = 0
        else:
            u[i, :] = uv
    return u

def plot_results(pts, tris, u):
    vmax = np.max(u)
    for d in range(3):
        plt.figure()
        plt.tripcolor(
            pts[:,0], pts[:, 1], tris,
            u[:,d], #shading='gouraud',
            cmap = 'PuOr', vmin = -vmax, vmax = vmax
        )
        plt.title("u " + ['x', 'y', 'z'][d])
        plt.colorbar()
    plt.show()

def print_error(pts, correct, est):
    close = np.sqrt(np.sum(pts ** 2, axis = 1)) < 4.0
    diff = correct[close,:] - est[close,:]
    l2diff = np.sum(diff ** 2)
    l2correct = np.sum(correct[close,:] ** 2)
    linferr = np.max(np.abs(diff))
    print("L2diff: " + str(l2diff))
    print("L2correct: " + str(l2correct))
    print("L2relerr: " + str(l2diff / l2correct))
    print("maxerr: " + str(linferr))
    return linferr

def make_free_surface(w, n):
    corners = [[-w, -w, 0], [-w, w, 0], [w, w, 0], [w, -w, 0]]
    return mesh_gen.make_rect(n, n, corners)

def make_fault(L, top_depth, n_fault):
    return mesh_gen.make_rect(n_fault, n_fault, [
        [-L, 0, top_depth], [-L, 0, top_depth - 1],
        [L, 0, top_depth - 1], [L, 0, top_depth]
    ])

def test_compare_to_okada():
    fault_L = 1.0
    top_depth = -0.5
    n_surf = 50
    sm = 1.0
    pr = 0.25
    n_fault = max(2, n_surf // 5)

    surf = make_free_surface(10, n_surf)
    fault = make_fault(fault_L, top_depth, n_fault)
    slip = np.array([[1, 0, 0] * fault[1].size]).flatten()
    pts, tris, fault_start_idx, soln = tt.solve_topo(
        surf, fault, slip, sm, pr,
        preconditioner = 'none'
    )

    surf_pts_map = np.unique(tris[:fault_start_idx])
    surf_pts = pts[surf_pts_map]
    surf_disp_all = np.empty((np.max(surf_pts_map) + 1, 3))
    surf_disp_all[tris[:fault_start_idx], :] = soln.reshape((-1, 3, 3))[:fault_start_idx]
    surf_disp = surf_disp_all[surf_pts_map]

    u = okada_exact(surf_pts, fault_L, top_depth, sm, pr)
    print_error(surf_pts, u, surf_disp)
    plot_results(surf_pts, surf[1], surf_disp)

    nxy = 100
    xs = np.linspace(-10, 10, nxy)
    xs = np.linspace(-10, 10, nxy)
    X, Y = np.meshgrid(xs, xs)
    z = -4.0
    obs_pts = np.array([X.flatten(), Y.flatten(), z * np.ones(Y.size)]).T.copy()

    interior_disp = tt.interior_evaluate(obs_pts, (pts, tris), soln, sm, pr)
    interior_disp = interior_disp.reshape((nxy, nxy, 3))
    plt.figure()
    plt.pcolor(xs, xs, interior_disp[:,:,0])
    plt.colorbar()
    plt.title('at z = ' + ('%.3f' % z) + '    ux')
    plt.show()

if __name__ == "__main__":
    test_compare_to_okada()
