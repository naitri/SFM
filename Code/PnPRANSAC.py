from LinearPnP import *
import numpy as np
import random


def error(x_img, P, x_world):
    ones = np.ones((x_world.shape[0], 1))
    X_all = np.hstack((x_world, ones))
    X_all = X_all.T

    pts = np.dot(P, X_all)
    pts = pts.T
    #u
    pts[:, 0] = pts[:, 0]/pts[:, 2]
    #v
    pts[:, 1] = pts[:, 1]/pts[:, 2]

    pts = pts[:, 0:2]

    # compute errror for all points
    err = x_img - pts
    err = err**2
    err = np.sum(err, axis=1)

    return err

def PnPRANSAC(x_world, x_img, k):
    max_inliers = 0
    thresh = 20
    R_best = None
    C_best = np.zeros(3)
    n_rows = len(x_img)
    # perform RANSAC to estimate the best pose
    for i in range(1000):

        # choose 6 random points and get linear pnp estimate
        random = np.random.choice(n_rows,size = 6)
        x_world_ = x_world[random,:]
        x_img_ = x_img[random,:]
        R, C = linear_pnp(x_world_, x_img_, k)

        

        # form the projection matrix
        C = C.reshape((3, 1))
        I = np.identity(3)
        M = np.hstack((I, -C))
        P = np.dot(k, np.dot(R, M))

     
        reproj_err = error(x_img, P, x_world)
        locs = np.where(reproj_err < thresh)[0]
        count = np.shape(locs)[0]
        if count > max_inliers:
            max_inliers = count
            img_pt = x_img[locs]
            world_pt = x_world[locs]
            R_best = R
            C_best = C

    return R_best,C_best,img_pt,world_pt
