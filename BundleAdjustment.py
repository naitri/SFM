import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

# Project 3D to 2D
def proj_to_img(p, camera):
	proj = rodriguez(p, camera[:, :3])
	proj += camera[:, 3:6]
	proj = -proj[:, :2] / proj[:, 2, np.newaxis]
	f = camera[:, 6]
	k1 = camera[:, 7]
	k2 = camera[:, 8]
	n = np.sum(proj ** 2, axis=1)
	r = 1 + k1 * n + k2 * n ** 2
	proj *= (r * f)[:, np.newaxis]

	return proj

# Rotate points using Rodriguez formula
def rodriguez(p, rot):
	theta = np.linalg.norm(rot, axis=1)[:, np.newaxis]
	with np.errstate(invalid='ignore'):
		v = np.nan_to_num(rot / theta)
	prod = np.sum(p * v, axis=1)[:, np.newaxis]
	cos = np.cos(theta)
	sin = np.sin(theta)

	return cos * p + sin * np.cross(v, p) + prod * (1 - cos) * v

# Compute loss (residuals)
def loss(params, num_cameras, num_points, camera_idx, point_idx, img_points):
	camera_params = params[:num_cameras * 9].reshape((num_cameras, 9))
	points_3d = params[num_cameras * 9:].reshape((num_points, 3))
	proj = project(points_3d[point_idx], camera_params[camera_idx])

	return (proj - img_points).ravel()

# 
def sparse_bundle(num_cameras, num_points, camera_idx, point_idx):
	m = camera_idx.size * 2
	n = num_cameras * 9 + num_points * 3
	A = lil_matrix((m, n), dtype=int)
	i = np.arange(camera_idx.size)

	for j in range(9):
		A[2 * i, camera_indices * 9 + j] = 1
		A[2 * i + 1, camera_indices * 9 + j] = 1

	for j in range(3):
		A[2 * i, num_cameras * 9 + point_idx * 3 + j] = 1
		A[2 * i + 1, num_cameras * 9 + point_idx * 3 + j] = 1

	return A

def BundleAdjustment(C_set, R_set, x_world, K, x_img, camera_idx, reconstruction, V):
	f = K[1, 1]
	camera_params = []
	point_idx, _ = np.where(reconstruction == 1)
	V = V[point_idx, :]
	points_3d = x_world[point_idx, :]
	for C0, R0 in zip(C_set, R_set):
		q_temp = Rotation.from_matrix(R0)
		Q0 = q_temp.as_rotvec()
		params = [Q0[0], Q0[1], Q0[2], C0[0], C0[1], C0[2], f, 0, 0]
		camera_params.append(params)

	camera_params = np.reshape(camera_params, (-1, 9))

	num_cameras = camera_params.shape[0]

	assert len(C_set) == num_cameras, "length not matched"

	num_points = points_3d.shape[0]

	n = 9 * num_cameras + 3 * num_points
	m = 2 * x_img.shape[0]

	print("n_cameras: {}".format(num_cameras))
	print("n_points: {}".format(num_points))
	print("Total number of parameters: {}".format(n))
	print("Total number of residuals: {}".format(m))
	opt = False

	if (opt):
		A = bundle_adjustment_sparsity(num_cameras, num_points, camera_idx, point_idx)
		x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

		res = least_squares(
		    loss,
		    x0,
		    jac_sparsity=A,
		    verbose=2,
		    x_scale='jac',
		    ftol=1e-4,
		    method='trf',
		    args=(num_cameras, num_points, camera_idx, point_idx, x_world))
		breakpoint()

		parameters = res.x

		camera_p = np.reshape(parameters[0:camera_params.size], (num_cameras, 9))

		x_world = np.reshape(parameters[camera_params.size:], (num_points, 3))

		for i in range(num_cameras):
			Q0[0] = camera_p[i, 0]
			Q0[1] = camera_p[i, 1]
			Q0[2] = camera_p[i, 2]
			C0[0] = camera_p[i, 2]
			C0[1] = camera_p[i, 2]
			C0[2] = camera_p[i, 6]
			r_temp = Rotation.from_rotvec([Q0[0], Q0[1], Q0[2]])
			R_set[i] = r_temp.as_matrix()
			C_set[i] = [C0[0], C0[1], C0[2]]

	return R_set, C_set, x_world