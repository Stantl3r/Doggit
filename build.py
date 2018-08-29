import cv2
import numpy as np
import os
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
training_images = '/Users/Stanley/Documents/Programs/RedditBot/images'
testing_images = '/Users/Stanley/Documents/Programs/RedditBot/images_test'
submission_image = '/Users/Stanley/Documents/Programs/RedditBot/submission'
size = 100

model_name = 'dogs.model'

def get_dog(image):
	name = image.split('_')[0]
	if name == 'labrador retriever': 
		return [1,0,0,0,0,0,0,0,0,0,0,0,0,0]
	elif name == 'german shepherd': 
		return [0,1,0,0,0,0,0,0,0,0,0,0,0,0]
	elif name == 'bulldog':
		return [0,0,1,0,0,0,0,0,0,0,0,0,0,0]
	elif name == 'beagle':
		return [0,0,0,1,0,0,0,0,0,0,0,0,0,0]
	elif name == 'poodle':
		return [0,0,0,0,1,0,0,0,0,0,0,0,0,0]
	elif name == 'rottweiler':
		return [0,0,0,0,0,1,0,0,0,0,0,0,0,0]
	elif name == 'yorkshire terrier':
		return [0,0,0,0,0,0,1,0,0,0,0,0,0,0]
	elif name == 'pointer':
		return [0,0,0,0,0,0,0,1,0,0,0,0,0,0]
	elif name == 'siberian husky':
		return [0,0,0,0,0,0,0,0,1,0,0,0,0,0]
	elif name == 'corgi':
		return [0,0,0,0,0,0,0,0,0,1,0,0,0,0]
	elif name == 'dachshund':
		return [0,0,0,0,0,0,0,0,0,0,1,0,0,0]
	elif name == 'australian shepherd':
		return [0,0,0,0,0,0,0,0,0,0,0,1,0,0]
	elif name == 'miniature schnauzer':
		return [0,0,0,0,0,0,0,0,0,0,0,0,1,0]
	elif name == 'boxer':
		return [0,0,0,0,0,0,0,0,0,0,0,0,0,1]

def create_train_data():
	training_data = []
	for image in os.listdir(training_images):
		breed = get_dog(image)
		path = os.path.join(training_images, image)
		image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (size, size))
		training_data.append([np.array(image), np.array(breed)])
	shuffle(training_data)
	np.save('train_data.npy', training_data)
	return training_data

def create_test_data():
	testing_data = []
	for image in os.listdir(testing_images):
		path = os.path.join(testing_images,image)
		image_num = image.split('.')[0]
		image = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (size, size))
		testing_data.append([np.array(image), image_num])   
	shuffle(testing_data)
	np.save('test_data.npy', testing_data)
	return testing_data

def create_submission_data(url):
	submission_data = []
	for image in os.listdir(submission_image):
		try:
			print(image)
			path = os.path.join(submission_image,image)
			image = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (size, size))
			submission_data.append([np.array(image), url])   
		except:
			continue
	np.save('submission_data.npy', submission_data)
	return submission_data

def load_model():
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

	convnet = fully_connected(convnet, 14, activation='softmax')
	convnet = regression(convnet, optimizer='RMSprop', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(convnet, tensorboard_dir='log')
	if os.path.exists('{}.meta'.format(model_name)):
		model.load(model_name)
		print('model loaded!')

	return model


def load_train_data():
	if os.path.exists('train_data.npy'):
		train_data = np.load('train_data.npy')
	else:
		train_data = create_train_data()
	return train_data

def load_test_data():
	if os.path.exists('test_data.npy'):
		test_data = np.load('test_data.npy')
	else:
		test_data = create_test_data()
	return test_data

