import numpy as np
import cv2
import glob
import argparse
from distutils.util import strtobool
from scipy.spatial.transform import Rotation 
from GetData import *
from GetInliersRANSAC import *
import matplotlib.pyplot as plt
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
def main():

    k =  np.array([[568.996140852, 0 ,643.21055941],
     [0, 568.988362396, 477.982801038],
     [0 ,0,1]])
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Path', default="../Data/", help='Path to data folder')
    Parser.add_argument('--Filtered', default="False",  type=lambda x: bool(strtobool(x)),help='If filtered data is available')
    Args = Parser.parse_args()
    folder = Args.Path
    filtered_avail = Args.Filtered
    images = [cv2.imread(img) for img in sorted(glob.glob(str(folder)+'/*.jpg'))]
    n_imgs = len(images)
    features_x,features_y,features = getFeatures(len(images),folder)
    print(filtered_avail)

    if filtered_avail:
        fundamental_matrix = np.load('fmatrix.npy',allow_pickle=True)
        features_filtered = np.load('features.npy',allow_pickle=True)
    else:
        print("Computing RANSAC")

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
        np.save('fmatrix.npy',fundamental_matrix)
        np.save('features.npy',features_filtered)


    #Rest of the pipeline only for 1st two images

    F_matrix = fundamental_matrix[0,1]
    
    #Estimate Essential Matrix from Fundamental Matrix
    E_matrix = estimate_Essentialmatrix(k,F_matrix)

    #Extract Poses of Camera (will be 4)
    R_set, T_set = get_RTset(E_matrix)


    
    find_idx = np.logical_and(features_filtered[:,0],features_filtered[:,1])
    index = np.where(find_idx == True)
    index = np.array(index).reshape(-1)
    pt1 = np.hstack((features_x[index,0].reshape((-1, 1)), features_y[index,0].reshape((-1, 1))))
    pt2 = np.hstack((features_x[index,1].reshape((-1, 1)), features_y[index,1].reshape((-1, 1))))

    #Linear Triangulation 
    point3D_set = linear_triangulation(R_set,T_set,pt1,pt2,k)
  
    # #Get pose of camera using cheirality condition
    R_best, T_best,X = extract_pose(R_set,T_set,point3D_set)

    #Non-Linear Triangulation
    X_ = non_linear_triangulation(R_best,T_best,pt1,pt2,X,k)
   
     # # Plotting non linear triangulation output
    plt.scatter(X_[:, 0], X_[:, 2], c='r', s=4)
    plt.scatter(X[:, 0], X[:, 2], c='g', s=4)
    ax = plt.gca()
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    R1 = Rotation.from_matrix(R_best).as_rotvec()
    R1 = np.rad2deg(R1)
    plt.plot(T_best[0],T_best[2], marker=(3, 0, int(R1[1])), markersize=15, linestyle='None')
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([0, 2])

    plt.show()
    

if __name__ == '__main__':
    main()