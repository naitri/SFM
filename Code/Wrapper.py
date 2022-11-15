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
from BundleAdjustment import *
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

    # f = open('error.csv', mode='w')
    # error_log = csv.writer(f)

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
   
    
    R1 = np.identity(3)
    T1 = np.zeros((3,1))
    # # #Non-Linear Triangulation
    X_nl = non_linear_triangulation(R1,T1,R_best,T_best,point1_fil,point2_fil,X_,k)
    
    plot_linear_nonlinear(X_,X_nl,index)
    # # #calculate error
    error_prior = mean_error(R1,T1,R_best,T_best,point1_fil,point2_fil,X_,k)
    print("Linear Triangulation",error_prior)
    print("--------------------------------------------------------------------")
    error_post = mean_error(R1,T1,R_best,T_best,point1_fil,point2_fil,X_nl,k)
    print("Non-linear triangulation",error_post)
    print("--------------------------------------------------------------------")
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    print("performing linear PnP to estimate pose of cameras 3-6")
    # using correspondences between the following image pairs for PnP

    # create a dict consisting of 2d-3d correspondences of all images
    corresp_2d_3d = {}

    # X_set stores all the 3d points
    X_set = []

    # first we need to get inliers of image i(3-6) wrt previously estimated camera pose so that we
    # match the 2D image point with the already calculated 3D point
    img1_2d_3d = point1_fil
    X_list_refined = np.reshape(X_nl[:,:3], (img1_2d_3d.shape[0], 3))
    img1_2d_3d = np.hstack((img1_2d_3d, X_list_refined))
    corresp_2d_3d[1] = img1_2d_3d

    # same thing for image 2
    img2_2d_3d = point2_fil
    img2_2d_3d = np.hstack((img2_2d_3d, X_list_refined))
    corresp_2d_3d[2] = img2_2d_3d

    # add the 3d points to X_set
    X_set.append(X_list_refined)
    X_set = np.array(X_set).reshape((X_list_refined.shape))

    # map is used for BA. It stores image points and indices of corresp 3d points
    ba_map = {}
    ba_map[1] = zip(corresp_2d_3d[1][:, 0:2], range(X_set.shape[0]))
    ba_map[2] = zip(corresp_2d_3d[2][:, 0:2], range(X_set.shape[0]))

    pose_set = {}
    pose_set_pnp ={}
    T1 = np.zeros(3)
    R1 = np.identity(3)
    pose_set[1] = np.hstack((R1,T1.reshape((3,1))))
    pose_set[2] =np.hstack((R_best,T_best.reshape((3,1))))

    pose_set_pnp[1] = np.hstack((R1,T1.reshape((3,1))))
    pose_set_pnp[2] =np.hstack((R_best,T_best.reshape((3,1))))
    # estimate pose for the remaining cams
    for i in range(2,n_imgs):

        ref_img_num = i
        new_img_num = i+1
        img_pair = str(ref_img_num)+str(new_img_num)
        file_name = "ransac"+img_pair+".txt"
        print(file_name)
        # construct projection matrix of ref image
        R_ref = pose_set[ref_img_num][:, 0:3].reshape((3,3))
        print(R_ref)
        C_ref = pose_set[ref_img_num][:, 3].reshape((3, 1))
     

        # get the 2d-3d correspondences for the 1st ref image
        ref_img_2d_3d = corresp_2d_3d[ref_img_num]
        
        ref_2d = ref_img_2d_3d[:,0:2]
        ref_3d = ref_img_2d_3d[:,2:]

        # next we must compare it with the points found using given matches
        points1,points2 = get_ransac_pts(folder, file_name)

        # obtain the 3D corresp for the new image
        ref_2d,new_2d,new_3d,matches = compute_correspondences(ref_2d,ref_3d,points1,points2)
        print(len(matches))


        '''.............................PnP RANSAC...........................'''
        print("performing PnP RANSAC to refine the poses")
        R_new_lt,T_new_lt,pnp_2d,pnp_3d = PnPRANSAC(new_3d, new_2d, k)
        P = projection_matrix(k,R_new_lt,T_new_lt)
        pnp_error = np.mean(error(new_2d, P, new_3d))
        print("linear PnP error",pnp_error)
        print("--------------------------------------------------------------------")
   
        '''.............................Non-linear PnP...........................'''
        print("performing Non-linear PnP to obtain optimal pose")
        R_new, T_new = nonlinear_pnp(k, R_new_lt,T_new_lt, pnp_2d,pnp_3d)

     
        P = projection_matrix(k,R_new,T_new)
        nonpnp_error = np.mean(error(new_2d, P, new_3d))
        print("Non-linear PnP error",nonpnp_error)
        print("--------------------------------------------------------------------")

  
        '''.............................Linear triangulation...........................'''
        print("performing Linear Triangulation to obtain 3d equiv for remaining 2d points")
        pt1 = matches[:,0:2]
        pt2 = matches[:,2:4]
        # pt1 = ref_2d
        # pt2 = new_2d
     
    
        # find the 2d-3d mapping for the remaining image points in the new image by doing triangulation
        X_lin_tri = point_triangulation(k,pt1,pt2,R_ref,C_ref,R_new,T_new)
        # X_lin_tri  = lt(Mref, T_new.reshape((3, 1)), R_new, k, matches)
        lt_error = mean_error(R_ref,C_ref,R_new,T_new,pt1,pt2,X_lin_tri,k)
        print("linear Triangulation error",lt_error)
        print("--------------------------------------------------------------------")

        colormap = ['r', 'b', 'g', 'y']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        C = T_new
        R = R_new
        X = X_lin_tri
        R = Rotation.from_matrix(R).as_rotvec()
        R1 = np.rad2deg(R)
        t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        t._transform = t.get_transform().rotate_deg(int(R1[1]))
        ax = plt.gca()
        ax.scatter((C[0]), (C[2]), marker=t, s=250, color=colormap[index])
        ax.scatter(X[:, 0], X[:, 2], s=4, color=colormap[index], label='cam'+str(index))
        plt.xlim(-20, 20)
        plt.ylim(-20, 30)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.legend()
        plt.savefig("linear_t"+img_pair+".png")
        # plt.show()
        
        

        '''.............................Non-Linear triangulation...........................'''
        print("performing Non-Linear Triangulation to obtain 3d equiv for remaining 2d points")
        X_new_nl = non_linear_triangulation(R_ref,C_ref,R_new,T_new,pt1,pt2,X_lin_tri,k)
        nlt_error = mean_error(R_ref,C_ref,R_new,T_new,pt1,pt2,X_new_nl,k)
        print("Non-linear Triangulation erro",nlt_error)
        print("--------------------------------------------------------------------")


        colormap = ['r', 'b', 'g', 'y']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        C = T_new
        R = R_new
        X = X_new_nl
        R = Rotation.from_matrix(R).as_rotvec()
        R1 = np.rad2deg(R)
        t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        t._transform = t.get_transform().rotate_deg(int(R1[1]))
        ax = plt.gca()
        ax.scatter((C[0]), (C[2]), marker=t, s=250, color=colormap[index])
        ax.scatter(X[:, 0], X[:, 2], s=4, color=colormap[index], label='cam'+str(index))
        plt.xlim(-20, 20)
        plt.ylim(-20, 30)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.legend()
        plt.savefig("non_linear_t"+img_pair+".png")
        
        
        # store the current pose after non linear pnp
        pose_set[new_img_num] = np.hstack((R_new, T_new.reshape((3, 1))))
        pose_set_pnp[new_img_num]=np.hstack((R_new, T_new.reshape((3, 1))))
        pts_img_all = pt2
        X_all = X_new_nl[:,:3]
        corresp_2d_3d[new_img_num] = np.hstack((pts_img_all, X_all))

        P = projection_matrix(k,R_new,T_new)
        draw_projected_points(images[new_img_num-1],X_all,pts_img_all,P,str(new_img_num))
        
        # do bundle adjustment
        index_start = X_set.shape[0]
        index_end = X_set.shape[0] + X_all.shape[0]
        ba_map[new_img_num] = zip(pts_img_all, range(index_start, index_end))
        X_set = np.append(X_set, X_all, axis=0)

        print("doing Bundle Adjustment --> ")
        pose_set_opt, X_set_opt = bundle_adjustment(pose_set, X_set, ba_map, k)

        # compute reproj error after BA
        R_ba = pose_set_opt[new_img_num][:, 0:3]
        C_ba = pose_set_opt[new_img_num][:, 3]
        X_all_ba = X_set_opt[index_start:index_end].reshape((X_all.shape[0], 3))

        X_set = X_set_opt

      
        corresp_2d_3d[new_img_num] = np.hstack((pts_img_all, X_all_ba))
        pose_set = pose_set_opt

        pose = pose_set[new_img_num]
        T_ = pose[:,3]
        R_ = pose[:, 0:3]
        P = projection_matrix(k,R_,T_)
        ba_error = np.mean(error(pts_img_all, P, X_all_ba))
        print("BA error:",ba_error)

        colormap = ['r', 'b', 'g', 'y','c','m']
        for i in range(len(pose_set)):
            pose = pose_set[i+1]
            pt = corresp_2d_3d[i+1]
            C = pose[:,3]
            R = pose[:, 0:3]
            X = pt[:,2:]
            R = Rotation.from_matrix(R).as_rotvec()
            R1 = np.rad2deg(R)
            t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
            t._transform = t.get_transform().rotate_deg(int(R1[1]))
            ax = plt.gca()
            ax.scatter((C[0]), (C[2]), marker=t, s=250, color=colormap[i])
            ax.scatter(X[:, 0], X[:, 2], s=4, color=colormap[i], label='cam'+str(i))
        plt.xlim(-20, 20)
        plt.ylim(-20, 30)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.legend()
        plt.savefig(img_pair+'.png')

        input('q')

        print("......................................")
    colormap = ['y', 'b', 'c', 'm', 'r', 'k']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colormap = ['r', 'b', 'g', 'y','c','m']
    for i in range(len(pose_set)):
        pose = pose_set[i+1]
        pt = corresp_2d_3d[i+1]
        C = pose[:,3]
        R = pose[:, 0:3]
        X = pt[:,2:]
        R = Rotation.from_matrix(R).as_rotvec()
        R1 = np.rad2deg(R)
        t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        t._transform = t.get_transform().rotate_deg(int(R1[1]))
        ax = plt.gca()
        ax.scatter((C[0]), (C[2]), marker=t, s=250, color=colormap[i])
        ax.scatter(X[:, 0], X[:, 2], s=4, color=colormap[i], label='cam'+str(i))
    plt.xlim(-20, 20)
    plt.ylim(-20, 30)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.legend()
    plt.savefig('BundleAdjustment'+img_pair+'.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colormap = ['r', 'b', 'g', 'y','c','m']
    for i in range(len(pose_set_pnp)):
        pose = pose_set_pnp[i+1]
        C = pose[:,3]
        R = pose[:, 0:3]
        X = pt[:,2:]
        R = Rotation.from_matrix(R).as_rotvec()
        R1 = np.rad2deg(R)
        t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        t._transform = t.get_transform().rotate_deg(int(R1[1]))
        ax = plt.gca()
        ax.scatter((C[0]), (C[2]), marker=t, s=250, color=colormap[i])
    plt.xlim(-20, 20)
    plt.ylim(-20, 30)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.legend()
    plt.savefig('PnP'+img_pair+'.png')


        
if __name__ == '__main__':
    main()