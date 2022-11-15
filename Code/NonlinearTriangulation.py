import numpy as np
import cv2
import scipy.optimize



def non_linear_triangulation(R1,T1,R2,T2,pt1,pt2,X,k):
    
    R1= R1.reshape((3,3))
    T1 = T1.reshape((3,1))
    R2 = R2.reshape((3,3))
    T2 = T2.reshape((3,1))

    I = np.identity(3)
    #calculating projection matrix P = K[R|T]
  
    P1 = np.dot(k,np.dot(R1,np.hstack((I,-T1))))
    P2 = np.dot(k,np.dot(R2,np.hstack((I,-T2))))
    #calculate new 3D points as per reprojection error
    points3D_new_set = []
    # X = np.hstack((X, np.ones((len(X),1))))
    for i in range(len(X)):
        opt = scipy.optimize.least_squares(fun=loss,x0 = X[i],args = [pt1[i], pt2[i],P1,P2])
        points3D_new = opt.x
        # points3D_new=points3D_new / points3D_new[-1]
        points3D_new_set.append(points3D_new)
    return np.array(points3D_new_set)

def mean_error(R1,T1,R2,T2,pt1,pt2,X,k):

    R1= R1.reshape((3,3))
    T1 = T1.reshape((3,1))
    R2 = R2.reshape((3,3))
    T2 = T2.reshape((3,1))

    I = np.identity(3)
    #calculating projection matrix P = K[R|T]
    P1 = np.dot(k,np.dot(R1,np.hstack((I,-T1))))
    P2 = np.dot(k,np.dot(R2,np.hstack((I,-T2))))

    e = []
    for i in range(len(X)):
        error = loss(X[i],pt1[i],pt2[i],P1,P2)
        e.append(error)
    return np.mean(e)

def loss(X,pt1,pt2,P1,P2):
    p11,p12,p13 = P1
    p21,p22,p23 = P2
    p11,p12,p13 = p11.reshape(1,-1),p12.reshape(1,-1),p13.reshape(1,-1)
    p21,p22,p23 = p21.reshape(1,-1),p22.reshape(1,-1),p23.reshape(1,-1)
    
    
   
   #for camera 1 (identity/origin)
    u1 = pt1[0]
    v1 = pt1[1]
   
    #for camera 2
    u2 = pt2[0]
    v2 = pt2[1]
    u1_ = np.divide(np.dot(p11,X),np.dot(p13,X))
    v1_ = np.divide(np.dot(p12,X),np.dot(p13,X))
   


    u2_ = np.divide(np.dot(p21,X),np.dot(p23,X))
    v2_ = np.divide(np.dot(p22,X),np.dot(p23,X))

    error1 = np.square(u1-u1_) + np.square(v1-v1_)
    error2 = np.square(u2-u2_) + np.square(v2-v2_)
    error = error2 + error1
    
    return error

