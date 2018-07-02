import cv2
import os

def gray(path):
	for pic_name in os.listdir(path):
		full_path=os.path.join(path,pic_name)
		img=cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
		if os.path.exists(full_path):
			os.remove(full_path)
		cv2.imwrite(full_path,img)