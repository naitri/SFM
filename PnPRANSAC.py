from LinearPnP import *
import numpy as np
import random

# projects 3D points to 2D
def project(x_world, K, C, R):
	x_world = x_world.reshape(-1, 1)
	C = C.reshape(-1, 1)
	p = np.matmul(np.matmul(K, R), np.hstack((np.identity(3), -C)))
	x_world = np.vstack((x_world, 1))
	u_img = (np.matmul(p[0, :], x_world)).T / (np.matmul(p[2, :], x_world)).T
	v_img = (np.matmul(p[1, :], x_world)).T / (np.matmul(p[2, :], x_world)).T
	return np.hstack((u_img, v_img))

def PnPRANSAC(x_world, x_img, K):
	num_points = len(x_world) # N
	num_iter = 600 # M
	threshold = 10 # epsilon_r
	n = 0
	m = x_img.shape[0]
	x_hom = to_homogeneous(x_img)
	c_ransac = np.zeros((3, 1))
	r_ransac = np.identity(3)

	for i in range(num_iter):
		rand = random.sample(range(m), 6)
		r, c = linear_pnp(x_world[rand, :], x_img[rand, :], K)
		s = []

		# for j, (p_world, p_img) in enumerate(zip(x_world, x_img)):
			# p = np.matmul(K, np.matmul(r, np.hstack(np.identity(3), -c.reshape(3, 1))))
			# e = np.square(p_img[0] - np.matmul(p[0], p_world)[0] / np.matmul(p[2], p_world)) + 
			# 	np.square(p_img[1] - np.matmul(p[1], p_world) / np.matmul(p[2], p_world))
		for j in range(m):
			proj = project(x_hom[j, :], K, c, r)
			e = np.sqrt(
				np.square((x_hom[j, 0]) - proj[0]) +
				np.square((x_hom[j, 1] - proj[1])))
			if e < threshold:
				s.append(j)

		s_len = len(s)
		if n < s_len:
			n = s_len
			r_ransac = r
			c_ransac = c

		if s_len == m:
			break

	return r_ransac, c_ransac