
import numpy as np
from scipy.spatial.transform import Rotation
import scipy.optimize as opt

def loss(cq, K, x_world, x_img):
    x_world = np.hstack((x_world, np.ones((x_world.shape[0], 1))))
    c = cq[:3]
    c = c.reshape(-1, 1)
    r = cq[3:7]

    r = Rotation.from_quat([r[0], r[1], r[2], r[3]])
    r = r.as_matrix()
    p = np.matmul(np.matmul(K, r), np.hstack((np.identity(3), -c)))

    u_proj = (np.matmul(p[0, :], x_world.T)).T / (np.matmul(p[2, :], x_world.T)).T
    v_proj = (np.matmul(p[1, :], x_world.T)).T / (np.matmul(p[2, :], x_world.T)).T
    e1 = x_img[:, 0] - u_proj
    e2 = x_img[:, 1] - v_proj
    e_sum = e1 + e2

    return sum(e_sum)

def NonlinearPnP(x_world, x_img, K, c0, r0):
    q = Rotation.from_matrix(r0)
    q = q.as_quat()

    cq = [c0[0], c0[1], c0[2], q[0], q[1], q[2], q[3]]
    params = opt.least_squares(fun=loss, method='dogbox', x0=cq, args=[K, x_world, x_img])
    c_final = params.x[:3]
    r_final = params.x[3:7]
    r_final = Rotation.from_quat([r_final[0], r_final[1], r_final[2], r_final[3]])
    r_final = r_final.as_matrix()

    return r_final, c_final