import numpy as np
import cv2
import glob
import argparse
import os
def getFeatures(n_imgs, path):
    rgb = []
    featuresx = []
    featuresy = []
    features = []
    for i in range(n_imgs-1):
        file_name = path +"/"+ "matching" + str(i+1) + ".txt"
        file = open(file_name,"r")

        for j, lines in enumerate(file):
            if j == 0:
                continue
            else:
                x_values = np.zeros((1, n_imgs))
                y_values = np.zeros((1, n_imgs))
                feature_table = np.zeros((1, n_imgs),dtype = int )
                line = lines.split()
                data = [float(x) for x in line]
                data = np.array(data)

                #no of features matched

                num_match = data[0]

                #rgb values 
                rgb.append([data[1],data[2],data[3]])

                #storing source x and y values
                x_values[0,i] = data[4]
                y_values[0,i] = data[5]
                feature_table[0,i] = 1
                #correspondence to other imgs
                n = 0
                while num_match >1:
                    dst_img_id = int(data[6+n])
                    x_values[0,dst_img_id -1] =data[7+n]
                    y_values[0,dst_img_id -1] = data[8+n]
                    feature_table[0,dst_img_id-1] = 1
                    n+=3
                    num_match-= 1

                featuresx.append(x_values)
                featuresy.append(y_values)
                features.append(feature_table)
    return np.array(featuresx).reshape(-1,n_imgs),np.array(featuresy).reshape(-1,n_imgs),np.array(features).reshape(-1,n_imgs)


def extract_points(path):

    os.chdir(path)

    for i in range(1, 6):

        file_name = ""
        file_name += "matching" + str(i) + ".txt"
        save_file_name = "matches" + str(i)

        file = open(file_name, 'r')
        content = file.readlines()

        nums = []

        for line in content[1:]:

            nums = line.split()
            num_matches = nums[0]

            matches = nums[6:]
            for j,match in enumerate(matches):

                if(j%3==0):

                    save_file = open(save_file_name + str(match) + ".txt", 'a')

                    points = str(nums[4]) + " " + str(nums[5]) + " " + str(matches[j+1]) + " " + str(matches[j+2]) + " " + str(nums[1]) + " " + str(nums[2]) + " " + str(nums[3]) + "\n"
                    save_file.write(points)
                    save_file.close()

def get_pts(path, file_name):

        os.chdir(path)

        file = open(file_name, 'r')
        content = file.readlines()

        point1 = []
        point2 = []

        for line in content:
            x1, y1, x2, y2,r,g,b = line.split()

            point1.append([np.float32(x1), np.float32(y1)])
            point2.append([np.float32(x2), np.float32(y2)])

        return np.array(point1),np.array(point2)
def get_ransac_pts(path, file_name):

        os.chdir(path)

        file = open(file_name, 'r')
        content = file.readlines()

        point1 = []
        point2 = []

        for line in content:
            x1, y1, x2, y2 = line.split()

            point1.append([np.float32(x1), np.float32(y1)])
            point2.append([np.float32(x2), np.float32(y2)])

        return np.array(point1),np.array(point2)
def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Path', default="/home/naitri/Documents/733/project3", help='data files path')
    Args = Parser.parse_args()
    folder = Args.Path
    images = [cv2.imread(img) for img in sorted(glob.glob(str(folder)+'/*.jpg'))]
    extract_points(folder)



if __name__ == '__main__':
    main()