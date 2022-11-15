
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from Plotting import *



def compute_reproj_err(M, pt_img, X):

    # make X homogenous
    X = X.reshape((3, 1))
    X = np.append(X, 1)

    # get the projected image points
    pt_img_proj = np.dot(M, X)

    # convert to non homog form
    pt_img_proj[0] = pt_img_proj[0]/pt_img_proj[2]
    pt_img_proj[1] = pt_img_proj[1]/pt_img_proj[2]

    reproj_err = ((pt_img[0] - pt_img_proj[0])**2) + ((pt_img[1] - pt_img_proj[1])**2)

    return reproj_err


def optimize_params(x0, K, pts_img_all, X_all):

    # calculate reprojection error
    reproj_err_all = []
    R = to_euler(x0[:4])
    C = x0[4:]

    
    P = projection_matrix(K, R, C)

    for pt_img, X in zip(pts_img_all, X_all):
        reproj_err = compute_reproj_err(P, pt_img, X)
        reproj_err_all.append(reproj_err)
    reproj_err_all = np.array(reproj_err_all)
    reproj_err_all = reproj_err_all.reshape(reproj_err_all.shape[0],)

    return reproj_err_all


def nonlinear_pnp(K, R,C, points2d,X):

    # extract image points
    poses_non_linear = {}

   
    pts_img_all = points2d
    X_all =X

    # make the projection projection matrix
    C = C.reshape((3, 1))

    # convert rotation matrix to quaternion form
    Q = to_quaternion(R)

    # defining the paramerter to optimize
    x0 = np.append(Q, C)

    result = least_squares(fun=optimize_params, x0=x0, args=(K, pts_img_all, X_all), ftol=1e-10)
    opt = result.x

    # quaternion to rotation matrix
    R_best = to_euler(opt[:4])
    C_best = opt[4:]
    C_best = C_best.reshape((3, 1))
    

    return R_best,C_best