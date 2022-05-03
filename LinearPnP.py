import numpy as np

# def build_matrix(x_world, x_img):
# 	# Normalize 2D points to isolate camera parameters
# 	x_norm = np.matmul(np.linalg.inv(K), x_img)

# 	a = np.zeros((3, 12))
# 	a[0, 4 :8] = -x_world
# 	a[0, 8:] = x_norm[1] * x_world
# 	a[1, :4] = x_world
# 	a[1, 8:] = -x_norm[0] * x_world
# 	a[2, :4] = -x_norm[1] * x_world
# 	a[2, 4:8] = x_norm[0] * x_world
# 	return a

def to_homogeneous(x):
	# x = np.array(x)
	# if len(x.shape) == 1:
	# 	x = np.hstack((x, [1])).reshape(x.shape, 1)
	# else:
	# 	if x.shape[1] == 1:
	# 		x = np.vstack((x, [1]))
	# 	elif x.shape[0] == 1:
	# 		x = np.hstack((x, np.array(1).reshape(1,1)))
	# 		x = x.reshape(x.shape[0], 1)
	m, n = x.shape

	if n == 3 or n == 2:
		x = np.hstack((x, np.ones((m, 1))))

	return x

def linear_pnp(x_world, x_img, K):
	# TO-DO: stack x_world and x_img depending on their formatting
	# a_stack = np.empty((0, 12), np.float32)
	# for p_world, p_img in zip(x_world, x_img):
	# 	a = build_matrix(x_world, x_img, K)
	# 	a_stack = np.vstack((a_stack, a))
	A = []
	N = x_world.shape[0]
	n = x_img.shape[0]
	x_world = np.hstack((x_world, np.ones((n, 1))))
	x_img = np.hstack((x_img, np.ones((n, 1))))
	x_img = np.transpose(np.matmul(np.linalg.inv(K), x_img.T))

	for i in range(N):
		xt = x_world[i, :].reshape((1, 4))
		z = np.zeros((1, 4))
		p = x_img[i, :]

		a1 = np.hstack((np.hstack((z, -xt)), p[1] * xt))
		a2 = np.hstack((np.hstack((xt, z)), -p[0] * xt))
		a3 = np.hstack((np.hstack((-p[1] * xt, p[0] * xt)), z))
		a = np.vstack((np.vstack((a1, a2)), a3))

		if (i == 0):
			A = a
		else:
			A = np.vstack((A, a))


	_, _, v = np.linalg.svd(A)
	p = v[-1].reshape((3, 4))
	r = p[:, :3]
	t = p[:, -1]
	u, _, v = np.linalg.svd(r)

	d = np.identity(3)
	d[2][2] = np.linalg.det(np.matmul(u, v))
	r = np.matmul(np.matmul(u, d), v)
	c = -np.matmul(np.linalg.inv(r), t)

	if np.linalg.det(r) < 0:
		r = -r
		c = -c

	return r, c