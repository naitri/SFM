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
from PnPRANSAC import *
from NonlinearPnP import *
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
    # F_best = estimate_Fmatrix(point1_fil,point2_fil)
    F_matrix = F_best
    print(F_matrix)
    
    #Estimate Essential Matrix from Fundamental Matrix
    E_matrix = estimate_Essentialmatrix(k,F_matrix)
    print(E_matrix)
    #Extract Poses of Camera (will be 4)
    R_set, T_set = get_RTset(E_matrix)


  

    # # #Linear Triangulation 
    point3D_set = linear_triangulation(R_set,T_set,point1_fil,point2_fil,k)
    
    plot_poses(R_set,T_set,point3D_set)

    # # # #Get pose of camera using cheirality condition
    R_best, T_best,X_ ,index= extract_pose(R_set,T_set,point3D_set)

    
    # plot_selectedpose(T_best,R_best,X_,index)
   
    

    # # #Non-Linear Triangulation
    X_nl = non_linear_triangulation(R_best,T_best,point1_fil,point2_fil,X_,k)
    
    plot_linear_nonlinear(X_,X_nl,index)
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
    T_set_.append(T0)
    R_set_.append(R0)

    T_set_.append(T_best)
    R_set_.append(R_best)

    #Corresponding 3D points are required for all images
    #Hence, taking reference of initial image, 

    #Rest of the pipeline computes from image 3 onwards where reference image will me image2

    #Algorithm:
    '''
    1) Compute 2d and corresponding 3d points of reference image
    2) Get matches of reference image and new image
    3) COmpute 3D points for new image
    '''

    for i in range(2,n_imgs):


        file_name = "ransac" +str(i)+str(i+1)+".txt"
        print(file_name)
        points1,points2 = get_ransac_pts(folder, file_name)
        img3_2d,img3_3d = compute_correspondences(point2_fil,X_nl,points1,points2)
        
        print("Computing poses using PnP")
        R_new,T_new = PnPRANSAC(img3_3d, img3_2d, k)
        P = projection_matrix(k,R_new,T_new)
        pnp_error = np.mean(error(img3_2d, P, img3_3d))
        print(pnp_error)
        R_new, T_new = NonlinearPnP(img3_3d, img3_2d, k, T_new, R_new)
        P = projection_matrix(k,R_new,T_new)
        nonpnp_error = np.mean(error(img3_2d, P, img3_3d))
        print(nonpnp_error)
        


        #Triangulating points
        for j in range(1,i):



if __name__ == '__main__':
    main()