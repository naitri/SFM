import numpy as np
import cv2
import glob
from os.path import exists
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
from Plotting import *
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
    n_imgs = 6

    #Uncomment the line below to extract points and store it as matches

    # extract_features(folder)

    if filtered_avail:
        fundamental_matrix = np.load('fmatrix.npy',allow_pickle=True)
    else:
        fundamental_matrix = np.zeros(shape=(6, 6), dtype=object)
        for i in range(n_imgs-1):
            for j in range(i+1, n_imgs):

                print("RANSAC for image" +str(i+1)+ "and" +str(j+1))
                pair_num = str(i+1)+str(j+1)
                file_name = "matches" +pair_num+".txt"
                if exists(folder+"/"+file_name):
                    points1,points2 = get_pts(folder, file_name)
                    point1_fil,point2_fil,F_best = ransac(points1,points2)
                    save_file_name = "ransac"+pair_num+".txt"
                    for idx in range(len(point1_fil)):
                        save_file = open(save_file_name, 'a')
                        save_file.write(str(point1_fil[idx][0])+ " " + str(point1_fil[idx][1]) + " " + str(point2_fil[idx][0]) + " " + str(point2_fil[idx][1]) + "\n")
                        save_file.close()
                        print(idx)

                    fundamental_matrix[i,j] = F_best.reshape((3,3))



                    display_ransac(images[i], images[j], point1_fil,point2_fil,points1,points2,pair_num)
                else:
                    continue




    # #Rest of the pipeline only for 1st two images
    file_name = "matches" +str(12)+".txt"
    points1,points2 = get_pts(folder, file_name)
    point1_fil,point2_fil,F_best = ransac(points1,points2)

    F_matrix = F_best
    print(F_matrix)
    
    #Estimate Essential Matrix from Fundamental Matrix
    E_matrix = estimate_Essentialmatrix(k,F_matrix)
    print(E_matrix)
    #Extract Poses of Camera (will be 4)
    R_set, T_set = get_RTset(E_matrix)


  

    # # #Linear Triangulation 
    point3D_set = linear_triangulation(R_set,T_set,point1_fil,point2_fil,k)
  
    # # # #Get pose of camera using cheirality condition
    R_best, T_best,X_ ,index= extract_pose(R_set,T_set,point3D_set)
    
    #plot all poses
    plot_poses(R_set,T_set,point3D_set)

    # # #Non-Linear Triangulation
    X_nl = non_linear_triangulation(R_best,T_best,point1_fil,point2_fil,X_,k)
    
    #plot linear vs non linear
    linear_nonlinear(X_,X_nl,index)

    # # #calculate error
    error_prior = mean_error(R_best,T_best,point1_fil,point2_fil,X_,k)
    print(error_prior)
    error_post = mean_error(R_best,T_best,point1_fil,point2_fil,X_nl,k)
    print(error_post)

    #PnP to estimate poses of other images
    T_set_ = []
    R_set_ = []

    T0 = np.zeros(3)
    R0 = np.identity(3)
    T_set_.append(C0)
    R_set_.append(R0)

    T_set_.append(T_best)
    R_set_.append(R_best)

    for i in range(2,n_imgs):

        file_name = "ransac" +str(i)+str(i+1)+".txt"
        points1,points2 = get_ransac_pts(folder, file_name)

    

if __name__ == '__main__':
    main()