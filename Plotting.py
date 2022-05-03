import os
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl



def resizing(imgs):
    images = imgs.copy()
    sizes = []
    resized = []
    for image in images:
        x, y, ch = image.shape
        sizes.append([x, y, ch])

    sizes = np.array(sizes)
    x_target, y_target, _ = np.max(sizes, axis = 0)
    
    

    for i, image in enumerate(images):
        resize = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
        resize[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
        resized.append(resize)

    return resized

def compare(X_linear, X_non_linear, index):

        # extract the x and the z components

        X_linear = np.array(X_linear)
        X_linear = X_linear.reshape((X_linear.shape[0], -1))

        x_l = X_linear[:, 0]
        z_l = X_linear[:, 2]

        x_nl = X_non_linear[:, 0]
        z_nl = X_non_linear[:, 2]

        # plot linear and non linear points and compare
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # define color scheme using index to identify which pose from the previous plot was correct
        colormap = np.array(['y', 'b', 'c', 'r'])

        ax.scatter(x_l, z_l, s=100, marker='+', color = colormap[index], label='linear')
        ax.scatter(x_nl, z_nl, s=7, color = 'k', label='non-linear')
        plt.xlim(-15, 20)
        plt.ylim(-30, 40)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.legend()
        plt.show()


def display_ransac(img1, img2, points1_fil,points2_fil,points1,points2,pair_num):
    image_1 = img1.copy()
    image_2 = img2.copy()
    image_1, image_2 = resizing([image_1, image_2])
    combine = np.concatenate((image_1, image_2), axis = 1)
    corners_1 = points1_fil.copy()
    corners_2  = points2_fil.copy()
    corners_2[:,0] += image_1.shape[1]
    corners_1a = points1.copy()
    corners_2a= points2.copy()
    corners_2a[:,0] += image_1.shape[1]
    for a1,b1,a2,b2 in zip(corners_1, corners_2,corners_1a, corners_2a):
        cv2.line(combine, (int(a1[0]),int(a1[1])), (int(b1[0]),int(b1[1])), (0, 0, 255), 1)
        cv2.circle(combine,(int(a1[0]),int(a1[1])),1,(255,0,0),-1)
        cv2.circle(combine,(int(b1[0]),int(b1[1])),1,(255,0,0),-1)
        cv2.line(combine, (int(a2[0]),int(a2[1])), (int(b2[0]),int(b2[1])), (0, 255, 0), 1)
        cv2.circle(combine,(int(a2[0]),int(a2[1])),1,(255,255,0),-1)
        cv2.circle(combine,(int(b2[0]),int(b2[1])),1,(255,255,0),-1)

    cv2.imwrite('ransac'+pair_num+'.png',combine)
    