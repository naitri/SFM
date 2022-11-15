import numpy as np
import cv2
import glob


def normalize_points(pts):
    '''
    https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/arao83/index.html
    '''
    mean_ =np.mean(pts,axis=0)

    #finding centre
    u = pts[:,0] - mean_[0]
    v = pts[:,1] - mean_[1]

    sd_u = 1/np.std(pts[:,0])
    sd_v = 1/np.std(pts[:,1])
    Tscale = np.array([[sd_u,0,0],[0,sd_v,0],[0,0,1]])
    Ta = np.array([[1,0,-mean_[0]],[0,1,-mean_[1]],[0,0,1]])
    T = np.dot(Tscale,Ta)

    pt = np.column_stack((pts,np.ones(len(pts))))
    norm_pts = (np.dot(T,pt.T)).T

    return norm_pts,T

def estimate_Fmatrix(img1_pts,img2_pts):

    #normalize points
    # img1_pts,T1 = normalize_points(img1_pts)
    # img2_pts,T2 = normalize_points(img2_pts)
    x1 = img1_pts[:,0]
    y1 = img1_pts[:,1]
    x1dash = img2_pts[:,0]
    y1dash = img2_pts[:,1]
    A = np.zeros((len(x1),9))
    for i in range(len(x1)):
        A[i] = np.array([x1dash[i]*x1[i],x1dash[i]*y1[i],x1dash[i], y1dash[i]*x1[i],y1dash[i]*y1[i],y1dash[i],x1[i],y1[i],1])

    #taking SVD of A for estimation of F
    U, S, V = np.linalg.svd(A,full_matrices=True)
    F_est = V[-1, :]
    F_est = F_est.reshape(3,3)

    # Enforcing rank 2 for F
    ua,sa,va = np.linalg.svd(F_est,full_matrices=True)
    sa = np.diag(sa)

    sa[2,2] = 0
   

    F = np.dot(ua,np.dot(sa,va))
    # F, mask = cv2.findFundamentalMat(img1_pts,img2_pts,cv2.FM_LMEDS)
    # F = np.dot(T2.T, np.dot(F, T1))
    # F = F / F[2,2]

    return F