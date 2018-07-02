import cv2
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential,load_model
from keras.layers import Dense,Flatten,Convolution2D,Activation,MaxPooling2D,Dropout
from api import load_file




class DataSet(object):
	def __init__(self,path):
		self.class_num=None
		self.train_X=None
		self.train_y=None
		self.test_X=None
		self.test_y=None
		self.img_size=128
		self.split_data(path)

	def split_data(self,path):
		img_list,label_list,class_num=load_file(path)
		train_X,test_X,train_y,test_y=train_test_split(img_list,label_list,train_size=0.8,shuffle=True,random_state=random.randint(0,100))
		
		train_X=train_X.reshape(train_X.shape[0],self.img_size,self.img_size,1)/255.0
		test_X=test_X.reshape(test_X.shape[0],self.img_size,self.img_size,1)/255.0
		train_X=train_X.astype("float32")
		test_X=test_X.astype("float32")
		train_y=np_utils.to_categorical(train_y,num_classes=class_num)
		test_y=np_utils.to_categorical(test_y,num_classes=class_num)


		self.train_X=train_X
		self.train_y=train_y
		self.test_X=test_X
		self.test_y=test_y
		self.class_num=class_num

class Model(object):
	FILE_PATH="./model/model-1.h5"
	IMG_SIZE=128

	def __inint__(self):
		self.model=None

	def read_data(self,data_set):
		self.data_set=data_set

	def build_model(self):
		"""
		func:build model
		"""
		self.model=Sequential()
		self.model.add(
				Convolution2D(
						filters=32,
						kernel_size=(4,4),
						padding='same',
						input_shape=self.data_set.train_X.shape[1:],
						dim_ordering='tf'
					)
			)

		self.model.add(
				MaxPooling2D(
						pool_size=(2,2),
						strides=(2,2),
						padding='same'
					)
			)

		self.model.add(Activation('relu'))
		#self.model.add(Dropout(0.1))

		self.model.add(
				Convolution2D(
						filters=64,
						kernel_size=(5,5),
						padding='same'
					)
			)

		self.model.add(
				MaxPooling2D(
						pool_size=(2,2),
						strides=(2,2),
						padding='same'
					)
			)

		self.model.add(Activation('relu'))

		self.model.add(Flatten())
		self.model.add(Dense(512))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(self.data_set.class_num))
		self.model.add(Activation('softmax'))
		self.model.summary()

	def train(self):
		self.model.compile(
				optimizer='adam',
				loss='categorical_crossentropy',
				metrics=['accuracy']
			)
		self.model.fit(self.data_set.train_X,self.data_set.train_y,epochs=12,batch_size=20)

	def evaluate(self):
		loss,accuracy=self.model.evaluate(self.data_set.test_X,self.data_set.test_y)
		print 'loss:'+str(loss)
		print 'accuracy:'+str(accuracy)

	def save(self,file_path=FILE_PATH):
		self.model.save(file_path)

	def load(self,file_path=FILE_PATH):
		self.model=load_model(file_path)

	def predict(self,img):
		img=img.reshape(1,self.IMG_SIZE,self.IMG_SIZE,1)
		img=img.astype('float32')
		img=img/255.0
		result=self.model.predict_proba(img)
		max_index=np.argmax(result)
		return max_index,result[0][max_index]		

if __name__=="__main__":
	data_set=DataSet("./grayfaces")
	model=Model()
	model.read_data(data_set)
	model.build_model()
	model.train()
	model.evaluate()
	model.save()
