from EstimateFundamentalMatrix import *
import numpy as np
import cv2


def ransac(pt1,pt2):
    n_rows = np.array(pt1).shape[0]
    no_iter = 1000
    thresh = 0.05
    ones = np.ones((pt1.shape[0], 1))
    pts_img1 = np.hstack((pt1, ones))
    pts_img2 = np.hstack((pt2, ones))

    ''' 42 - 858
    '''
    # random.seed(42)
    ## RANSAC
    max_inliers = 0

    for i in range(10000):
     
        random = np.random.choice(n_rows,size = 8)

        # estimate fundamental qmatrix
        img1_8pt = pt1[random,:]
        img2_8pt = pt2[random,:]
       
        F = estimate_Fmatrix(img1_8pt,img2_8pt)

        # compute (x2.T)Fx1
        vals = np.abs(np.diag(np.dot(np.dot(pts_img2, F), pts_img1.T)))

        # setting threshold
        # print(vals)
        inliers_index = np.where(vals<thresh)
        outliers_index = np.where(vals>=thresh)

        # inliers_index = np.where(vals<0.05)
        # outliers_index = np.where(vals>=0.05)

        # checking for max_inliersand saving it's index
        if np.shape(inliers_index[0])[0] > max_inliers:
            max_inliers = np.shape(inliers_index[0])[0]
            max_inliers_index = inliers_index
            min_outliers_index = outliers_index
            F_max_inliers = F

    img1_points = pt1[max_inliers_index ]
    img2_points = pt2[max_inliers_index ]
    F = estimate_Fmatrix(img1_points,img2_points)

    return img1_points,img2_points, F

 