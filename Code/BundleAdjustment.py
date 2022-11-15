import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from BuildVisibilityMatrix import *
import time
from Plotting import *

def compute_error(K, pts_2d, param_cam, world_pts_3d):
    ones = np.ones((world_pts_3d.shape[0], 1))
    world_pts_3d = np.hstack((world_pts_3d, ones))

    pt_img_proj = np.empty((0, 2), dtype=np.float32)

    for i, p in enumerate(world_pts_3d):
        R = to_euler(param_cam[i, :4])
        C = param_cam[i, 4:]
        M = projection_matrix(K, R, C)
        p = p.reshape((4, 1))
        proj_pt = np.dot(M, p)
        proj_pt = proj_pt/proj_pt[2]
        proj_pt = proj_pt[:2]
        proj_pt = proj_pt.reshape((1, 2))
        pt_img_proj = np.append(pt_img_proj, proj_pt, axis=0)

    reproj_err = pts_2d - pt_img_proj
    # reproj_err = reproj_err**2
    # reproj_err = np.sum(reproj_err, axis=1)

    return reproj_err.ravel()


def get_params(pose_set, X_world_all, map_2d_3d):

    # defining the params, 2d points, 3d points' indices
    x0 = np.empty(0, dtype=np.float32)
    indices = np.empty(0, dtype=int)
    pts_2d = np.empty((0, 2), dtype=np.float32)
    indices_cam = np.empty(0, dtype=int)

    n_cam = max(pose_set.keys())

    # for each camera pose
    for k in pose_set.keys():

        # convert to quaternion
        Q = to_quaternion(pose_set[k][:, 0:3])
        C = pose_set[k][:, 3]
        # append the parameters
        x0 = np.append(x0, Q.reshape(-1), axis=0)
        x0 = np.append(x0, C, axis=0)

        for p in map_2d_3d[k]:
            indices = np.append(indices, [p[1]], axis=0)
            pts_2d = np.append(pts_2d, [p[0]], axis=0)
            indices_cam = np.append(indices_cam, [k-1], axis=0)

    x0 = np.append(x0, X_world_all.flatten(), axis=0)
    n_3d = X_world_all.shape[0]

    return n_cam, n_3d, indices, pts_2d, indices_cam, x0


def loss(x0, n_cam, n_3d, indices, pts_2d, indices_cam, K):
    param_cam = x0[:n_cam*7].reshape((n_cam, 7))
    world_pts_3d = x0[n_cam*7:].reshape((-1, 3))
    reproj_err = compute_error(K, pts_2d, param_cam[indices_cam], world_pts_3d[indices])
    return reproj_err


def bundle_adjustment(pose_set, X_world_all, map_2d_3d, K):

    n_cam, n_3d, indices, pts_2d, indices_cam, x0 = get_params(pose_set, X_world_all, map_2d_3d)
    
    A = sparse_matrix(n_cam, n_3d, indices_cam, indices)
    start = time.time()
    result = least_squares(fun=loss, x0=x0, jac_sparsity=A, verbose=2, x_scale='jac',
    ftol=1e-4, method='trf', args=(n_cam, n_3d, indices, pts_2d, indices_cam, K))
    end = time.time()
    print("optimisation took -- {} seconds".format(start-end))

    param_cam = result.x[:n_cam*7].reshape((n_cam, 7))
    X_world_all_opt = result.x[n_cam*7:].reshape((n_3d, 3))
    pose_set_opt = {}
    i = 1
    for cp in param_cam:
        R = to_euler(cp[:4])
        C = cp[4:].reshape((3, 1))
        pose_set_opt[i] = np.hstack((R, C))
        i += 1

    return pose_set_opt, X_world_all_opt
