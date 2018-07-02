#coding=utf-8
import cv2

from face_model import Model
from api import load_name_list

class video_reader(object):
	
	def __init__(self):
		self.model=Model()
		self.model.load('./model/model-1.h5')
		self.IMG_SIZE=128

	def build_video(self):
		name_list=load_name_list("./grayfaces")
		cameraCapture = cv2.VideoCapture(0)
		success, frame = cameraCapture.read()
		face_cascade=cv2.CascadeClassifier('./model/haarcascade_frontalface_alt.xml')


		while success and cv2.waitKey(1)==-1:
			success, frame=cameraCapture.read()
			gray_pic=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			faces=face_cascade.detectMultiScale(gray_pic,1.3,5)
			for (x,y,w,h) in faces:
				face=gray_pic[x:x+w,y:y+h]
				face=cv2.resize(face,(self.IMG_SIZE,self.IMG_SIZE),interpolation=cv2.INTER_LINEAR)
				label,prob=self.model.predict(face)
				#name=name_list[label]
				print name_list[label],prob
				if prob>0.7:
					name=name_list[label]
				else:
					name='unknown'
				cv2.putText(frame,name,(x, y - 20),cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)#display name
				frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
			cv2.imshow("camera",frame)
		cameraCapture.release()
		cv2.destroyAllWindows()


if __name__=="__main__":
	video=video_reader()
	video.build_video()
