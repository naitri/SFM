import numpy as np
import cv2
import glob
import argparse

from GetData import *
from GetInliersRANSAC import *

from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
def main():

    k =  np.array([[568.996140852, 0 ,643.21055941],
     [0, 568.988362396, 477.982801038],
     [0 ,0,1]])
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Path', default="../Data/", help='data files path')
    Args = Parser.parse_args()
    folder = Args.Path
    images = [cv2.imread(img) for img in sorted(glob.glob(str(folder)+'/*.jpg'))]
    n_imgs = len(images)
    features_x,features_y,features = getFeatures(len(images),folder)

    fundamental_matrix = np.zeros(shape=(6, 6), dtype=object)
    features_filtered = np.zeros_like(features)


    for i in range(n_imgs-1):
        for j in range(i+1, n_imgs):
            print("RANSAC for image" +str(i+1)+ "and" +str(j+1))


            find_idx = np.logical_and(features[:,i],features[:,j])
            index = np.where(find_idx == True)
            index = np.array(index).reshape(-1)
            points1 = np.hstack((features_x[index,i].reshape((-1, 1)), features_y[index,i].reshape((-1, 1))))
            points2 = np.hstack((features_x[index,j].reshape((-1, 1)), features_y[index,j].reshape((-1, 1))))
            if len(index) > 0:
                index_fil,F_best = ransac(points1,points2,index)
                fundamental_matrix[i,j] = F_best
                features_filtered[index_fil,i]=1
                features_filtered[index_fil,j] =1
            else:
                continue
            

    #Rest of the pipeline only for 1st two images

    F_matrix = fundamental_matrix[0,1]

    #Estimate Essential Matrix from Fundamental Matrix
    E_matrix = estimate_Essentialmatrix(k,F_matrix)

    #Extract Poses of Camera (will be 4)
    R_set, T_set = get_RTset(E_matrix)

    #Linear Triangulation 
    find_idx = np.logical_and(features_filtered[:,0],features_filtered[:,1])
    index = np.where(find_idx == True)
    index = np.array(index).reshape(-1)
    pt1 = np.hstack((features_x[index,0].reshape((-1, 1)), features_y[index,0].reshape((-1, 1))))
    pt2 = np.hstack((features_x[index,1].reshape((-1, 1)), features_y[index,1].reshape((-1, 1))))
    point3D_set = linear_triangulation(R_set,T_set,pt1,pt2,k)

    #Get pose of camera using cheirality condition
    R_best, T_best = extract_pose(R_set,T_set,point3D_set)


    

if __name__ == '__main__':
    main()