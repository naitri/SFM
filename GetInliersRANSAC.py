from EstimateFundamentalMatrix import *
import numpy as np
import cv2


def ransac(pt1,pt2):
    n_rows = np.array(pt1).shape[0]
    no_iter = 2000
    threshold = 0.08
    inliers = 0
    
    final_indices = []
    for i in range(no_iter):
        indices = []
     
        #randomly select 8 points
        random = np.random.choice(n_rows,size = 8)
        img1_8pt = pt1[random,:]
        img2_8pt = pt2[random,:]
        F_est = estimate_Fmatrix(img1_8pt,img2_8pt)
        for j in range(n_rows):
            x1 = pt1[j]
            x2 = pt2[j]

            #error computation
            pt1_ = np.array([x1[0],x1[1],1])
            pt2_ = np.array([x2[0],x2[1],1])
            error = np.dot(pt1_.T,np.dot(F_est,pt2_))
            
            if np.abs(error) < threshold:
                indices.append(j)
                
        if len(indices) > inliers:
            inliers = len(indices)
            final_indices = indices
            F = F_est

 
    img1_points = pt1[final_indices]
    img2_points = pt2[final_indices]


    return img1_points,img2_points, F

 