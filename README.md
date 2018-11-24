# Face-Recognition
人工智能课程的课程作业，基于keras和opencv的人脸识别demo

准备样本：
训练使用的图片放在grayfaces文件夹下面，每一类图片需要放在同一个子文件夹下，如grayfaces/zhangsan，grayfaces/lisi...
每一个子文件的名字就是子文件内图片的label，文件夹以及图片命名建议使用英文或阿拉伯数字，尽量保证训练使用的图片大小一致

训练模型：
>>>python face_model.py    #运行之后会在model文件夹下保存训练的模型，模型参数在face_model.py内可调

进行检测：
```python
>>>python video_capture.py    #加载模型，利用opencv进行人脸提取，截取人脸之后进行预测，输出概率
```
由于模型训练使用的是单通道的灰度图像，所以需要对彩色图片进行灰度处理，可以采用api中的函数：
```python
>>>from api import gray
>>>gray(./grayfaces/zhangsan)    #将zhangsan文件夹下面的彩色图像灰化
```

在训练模型阶段完成之后，可以利用predict.py对模型的准确性进行检测，输出模型预测的类型以及概率
```python
>>>from predict import predict
>>>predict('./grayfaces/zhangsan/1.jpg')
zhangsan 0.99
```

后来在真正实验运行过程中发现效果并不好，排查之后发现可能是像素的问题
opencv在进行人脸提取时存在误差，电脑摄像头采集的图片与手机采集的图片像素相去甚远
解决方案是采用电脑摄像头采集训练图片进行训练
```python
>>>from api import get_pic_by_camera
>>>get_pic_by_camera("wang",10,128)    
```
label为wang，采集图片张数为10，图片大小默认为128*128按“p”键采集图片，按“q”键退出采集，采集的图片放在data文件夹下
