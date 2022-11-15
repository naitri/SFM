import numpy as np
import cv2

def point_triangulation(k,pt1,pt2,R1,C1,R2,C2):
    points_3d = []

    I = np.identity(3)
    C1 = C1.reshape(3,1)
    C2 = C2.reshape(3,1)

    #calculating projection matrix P = K[R|T]
    P1 = np.dot(k,np.dot(R1,np.hstack((I,-C1))))
    P2 = np.dot(k,np.dot(R2,np.hstack((I,-C2))))
  
    #homogeneous coordinates for images
    xy = np.hstack((pt1,np.ones((len(pt1),1))))
    xy_cap = np.hstack((pt2,np.ones((len(pt1),1))))

    
    p1,p2,p3 = P1
    p1_cap, p2_cap,p3_cap = P2

    #constructing contraints matrix
    for i in range(len(xy)):
        A = []
        x = xy[i][0]
        y = xy[i][1]
        x_cap = xy_cap[i][0]
        y_cap = xy_cap[i][1] 
        
        A.append((y*p3) - p2)
        A.append((x*p3) - p1)
        
        A.append((y_cap*p3_cap)- p2_cap)
        A.append((x_cap*p3_cap) - p1_cap)

        A = np.array(A).reshape(4,4)

        _, _, v = np.linalg.svd(A)
        x_ = v[-1,:]
        x_ = x_/x_[-1]
        # x_ =x_[:3]
        points_3d.append(x_)


    return np.array(points_3d)

def linear_triangulation(R_Set,T_Set,pt1,pt2,k):
    R1_ = np.identity(3)
    T1_ = np.zeros((3,1))
    points_3d_set = []
    for i in range(len(R_Set)):
        points3d = point_triangulation(k,pt1,pt2,R1_,T1_,R_Set[i],T_Set[i])
        points_3d_set.append(points3d)

    return points_3d_set

