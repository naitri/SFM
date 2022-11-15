import numpy as np
import cv2


def get_RTset(E):

    U, S, V = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    R1 = np.dot(U,np.dot(W,V))
    R2 = np.dot(U,np.dot(W,V))
    R3 = np.dot(U,np.dot(W.T,V))
    R4 = np.dot(U,np.dot(W.T,V))

    T1 = U[:,2]
    T2 = -U[:,2]
    T3 = U[:,2]
    T4 = -U[:,2]

    R = [R1,R2,R3,R4]
    T = [T1,T2,T3,T4]

    for i in range(len(R)):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            T[i] = -T[i]

    return R, T