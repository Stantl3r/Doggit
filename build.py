import cv2
import numpy as np
import os
from random import shuffle

training_images = 'C:/Users/Stanley/Desktop/Programs/RedditBot/images'
testing_images = 'C:/Users/Stanley/Desktop/Programs/RedditBot/images_test'
size = 100

model_name = 'dogs.model'

def get_dog(image):
	name = image.split('!')[0]
	if name == 'german': 
		return [1,0]
	elif name == 'golden': 
		return [0,1]

def train_data():
	training_data = []
	for image in os.listdir(training_images):
		breed = get_dog(image)
		path = os.path.join(training_images, image)
		try:
			image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (size, size))
			training_data.append([np.array(image), np.array(breed)])
		except:
			continue
	shuffle(training_data)
	np.save('train_data.npy', training_data)
	return training_data

def test_data():
	testing_data = []
	for image in os.listdir(testing_images):
		path = os.path.join(testing_images,image)
		image_num = image.split('.')[0]
		image = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (size, size))
		testing_data.append([np.array(image), image_num])   
	shuffle(testing_data)
	np.save('test_data.npy', testing_data)
	return testing_data

if os.path.exists('train_data.npy'):
	train_data = np.load('train_data.npy')
else:
	train_data = train_data()

if os.path.exists('test_data.npy'):
	test_data = np.load('test_data.npy')
else:
	test_data = test_data()


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, size, size, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')



if os.path.exists('{}.meta'.format(model_name)):
	model.load(model_name)
	print('model loaded!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,size,size,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,size,size,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=model_name)
model.save(model_name)


import matplotlib.pyplot as plt

fig=plt.figure()

for num,data in enumerate(test_data[:12]):

	image_num = data[1]
	image_data = data[0]

	y = fig.add_subplot(3,4,num+1)
	orig = image_data
	data = image_data.reshape(size,size,1)
	model_out = model.predict([data])[0]

	if np.argmax(model_out) == 0: 
		str_label='German'
	elif np.argmax(model_out) == 1:
		str_label='Golden'

	y.imshow(orig,cmap='gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show()