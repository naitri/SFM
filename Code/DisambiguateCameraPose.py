import numpy as np
import cv2

def compute_cheriality(pt,r3,t):
    count_depth = 0
    for xy in pt:
        if np.dot(r3,(xy[:3]-t)) > 0 and t[2] > 0:
            count_depth +=1
    return count_depth

def extract_pose(R_set,T_set,pts_3d_set):
    threshold = 0
    index = None
    #Four sets are available for each possibility
    for i in range(len(R_set)):
        R = R_set[i]
        T = T_set[i]
        r3 = R[2]
        pt3d = pts_3d_set[i]
        #calculating which R satisfies the condition
        num_depth_positive = compute_cheriality(pt3d,r3,T)
        print(num_depth_positive)
        if num_depth_positive > threshold:
            index = i 
            threshold = num_depth_positive

            R_best = R_set[index]
            T_best = T_set[index]
            X_best = pts_3d_set[index]


    return R_best,T_best,X_best,index