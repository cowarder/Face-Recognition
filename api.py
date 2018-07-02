import numpy as np
import cv2
import os

def load_file(path):
	"""
	load picture filw from given 
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

def gray(path):
	"""
	change pictures in the directory of path to gray
	"""
	for pic_name in os.listdir(path):
		full_path=os.path.join(path,pic_name)
		img=cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
		if os.path.exists(full_path):
			os.remove(full_path)
		cv2.imwrite(full_path,img)

def get_pic_by_camera(name,size,IMG_SIZE=128):
	"""
		get data from camera
		
		name:label of a man
		size:num of pictures for each man
	"""
	if not os.path.exists('data/'+name):
		os.mkdir('data/'+name)
	cameraCapture = cv2.VideoCapture(0)
	success, frame = cameraCapture.read()
	face_cascade=cv2.CascadeClassifier('./model/haarcascade_frontalface_alt.xml')

	num=0
	while 1:
		success, frame=cameraCapture.read()
		gray_pic=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces=face_cascade.detectMultiScale(gray_pic,1.3,5)
		for (x,y,w,h) in faces:
			#face=gray_pic[x:x+w+w,y:y+h+int(h/2)]
			face=gray_pic[x:x+w,y:y+h]
			#,interpolation=cv2.INTER_LINEAR
			#face=cv2.resize(face,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_LINEAR)
			frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
			cv2.imshow("camera",frame)
			if cv2.waitKey(1)&0xff == ord('p'):
				num+=1
				cv2.imwrite('data/'+name+'/'+name+str(num)+str('.pgm'),face)
		if num>=size:
			break
		if cv2.waitKey(1)&0xff == ord('q'):
			break
		

	cameraCapture.release()
	cv2.destroyAllWindows()
