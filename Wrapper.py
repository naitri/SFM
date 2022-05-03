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
from PnPRANSAC import *
from NonlinearPnP import *
from BuildVisibilityMatrix import *
def main():

    k =  np.array([[568.996140852, 0 ,643.21055941],
     [0, 568.988362396, 477.982801038],
     [0 ,0,1]])
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Path', default="../Data/", help='Path to data folder')
    Parser.add_argument('--Filtered', default="True",  type=lambda x: bool(strtobool(x)),help='If filtered data is available')
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
    point3D_set = []

    for i in range(4):
        point3D_set.append(linear_triangulation(np.identity(3), np.zeros((3,1)), R_set[i],T_set[i],pt1,pt2,k))


    # #Get pose of camera using cheirality condition
    R_best, T_best,X = extract_pose(R_set,T_set,point3D_set)

    #Non-Linear Triangulation

    X_ = non_linear_triangulation(np.identity(3), np.zeros((3,1)), R_best,T_best,pt1,pt2,X,k)
   
    # Plotting non linear triangulation output
    # plt.scatter(X_[:, 0], X_[:, 2], c='r', s=4)
    # plt.scatter(X[:, 0], X[:, 2], c='g', s=4)
    # ax = plt.gca()
    # ax.set_xlabel('x')
    # ax.set_ylabel('z')
    # R1 = Rotation.from_matrix(R_best).as_rotvec()
    # R1 = np.rad2deg(R1)
    # plt.plot(T_best[0],T_best[2], marker=(3, 0, int(R1[1])), markersize=15, linestyle='None')
    # ax.set_xlim([-0.5, 0.5])
    # ax.set_ylim([0, 2])

    # plt.show()
    X_ = X_[:, :-1]
    points_3D = np.zeros((features_filtered.shape[0], 3))
    points_3D[index, :] = X_
    reconstruction = np.zeros((features_filtered.shape[0], 1))
    reconstruction[index] = 1
    vis_matrix = np.zeros((features_filtered.shape[0], n_imgs))
    vis_matrix[index, 0] = 1
    vis_matrix[index, 1] = 1

    C_matrices = [T_best]
    R_matrices = [R_best]

    # the previous part of the pipeline focused on img0 and img1
    img_indices = [0, 1]

    for i in range(n_imgs):
        print(i)
        if np.isin(img_indices, i)[0]:
            continue

        match = np.logical_and(reconstruction, features_filtered[:, i].reshape(-1, 1))
        idx, _ = np.where(match == True)

        # checking if there are less than 8 point correspondences
        if len(idx) < 8:
            continue

        x_img = np.transpose([features_x[idx, i], features_y[idx, i]])
        x_world = points_3D[idx, :]

        r, c = PnPRANSAC(x_world, x_img, k)
        r, c = NonlinearPnP(x_world, x_img, k, c, r)
        R_matrices.append(r)
        C_matrices.append(c)
        img_indices.append(i)
        vis_matrix[idx, i] = 1

        for j in range(len(img_indices) - 1):
            match = np.logical_and(np.logical_and(1 - reconstruction, features_filtered[:, img_indices[j]].reshape(-1, 1)), 
                    features_filtered[:, i].reshape(-1, 1))
            idx, _ = np.where(match == True)

            if len(idx) < 8:
                continue

            pts1 = np.hstack((features_x[idx, img_indices[j]].reshape(-1, 1), features_y[idx, img_indices[j]].reshape(-1, 1)))
            pts2 = np.hstack((features_x[idx, i].reshape(-1, 1), features_y[idx, i].reshape(-1, 1)))

            X_new = linear_triangulation(R_matrices[j], C_matrices[j], r, c, pts1, pts2, k)

            X_new = non_linear_triangulation(R_matrices[j], C_matrices[j], r, c, pts1, pts2, X_new, k)

            points_3D[idx, :] = X_new[:, :-1]
            # points_3D[idx, :] = X_new
            reconstruction[idx] = 1
            vis_matrix[idx, img_indices[j]] = 1
            vis_matrix[img_indices, j] = 1

        for b in range(len(points_3D)):
            if points_3D[b, 2] < 0:
                vis_matrix[b, :] = 0
                reconstruction[b] = 0

        bundle = BuildVisibilityMatrix(vis_matrix, img_indices)
        point_idx = np.where(reconstruction == 1)
        camera_idx = i * np.ones((len(point_idx), 1))
        points2D = np.hstack((features_x[point_idx, i].reshape(-1, 1), features_x[point_idx, i].reshape(-1, 1)))

        # TO-DO: Bundle Adjustment


if __name__ == '__main__':
    main()