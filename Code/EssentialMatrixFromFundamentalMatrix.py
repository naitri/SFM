import numpy as np
import cv2

def estimate_Essentialmatrix(k,F):
    E_est = np.dot(k.T,np.dot(F,k))
    #reconstructing E by correcting singular values
    U, S, V = np.linalg.svd(E_est,full_matrices=True)
    S = np.diag(S)
    S[0,0],S[1,1],S[2,2] = 1,1,0
    E = np.dot(U,np.dot(S,V))

    return E