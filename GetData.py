import numpy as np
import cv2
import glob
import argparse
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


def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Path', default="../Data/", help='data files path')
    Args = Parser.parse_args()
    folder = Args.Path
    images = [cv2.imread(img) for img in sorted(glob.glob(str(folder)+'/*.jpg'))]
    features_x,features_y = getFeatures(len(images),folder)
    print(len(features_x))


if __name__ == '__main__':
    main()