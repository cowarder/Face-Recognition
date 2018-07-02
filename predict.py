from api import load_name_list
import cv2
from keras.models import Sequential,load_model
import numpy as np
import os

def predict(path):
	path=os.path.join("./grayfaces",path)
	model=load_model("/home/wang/Desktop/face/model/model-1.h5")
	name_list=load_name_list("./grayfaces")
	img=cv2.imread(path)
	img=cv2.resize(img,(128,128))
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img=img.reshape(1,128,128,1)
	result=model.predict_proba(img)
	max_index=np.argmax(result)
	#return max_index,result[0][max_index]	
	print name_list[max_index],result[0][max_index]
