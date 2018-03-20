import numpy as np
import cv2
import os

def load_file(path):
	"""

	"""
	IMG_SIZE=128
	img_list=[]
	label_list=[]
	class_num=0
	for child_dir in os.listdir(path):
		child_path=os.path.join(path,child_dir)
		for file_name in os.listdir(child_path):
			image=cv2.imread(os.path.join(child_path,file_name))
			image=cv2.resize(image,(IMG_SIZE,IMG_SIZE))
			image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
			img_list.append(image)
			label_list.append(class_num)
		class_num+=1
	img_list=np.array(img_list)
	return img_list,label_list,class_num

def load_name_list(path):
	"""
	get all of the class name in the fold
	"""
	
	name_list=[]
	for child_dir in os.listdir(path):
		name_list.append(child_dir)
	return name_list