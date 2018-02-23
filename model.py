#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import cv2
from conv_layer import Conv2DPaddingSameStride1
from maxpooling_layer import MaxPoolingPaddingValid
from flatten_layer import Flatten
from full_connect_layer import FCN
from activation_layer import Relu, Softmax
from dl_net import Net


def construct_model(array_conv1_kernels=None, array_conv1_biases=None, array_conv2_kernels=None,
					array_conv2_biases=None, array_fcn1_kernels=None, array_fcn1_biases=None,
					array_fcn2_kernels=None, array_fcn2_biases=None):
	mnist_net = Net(name='mnist_net', class_num = 10)

	conv1 = Conv2DPaddingSameStride1(input_shape=(28, 28, 1), kernels_shape=(5, 5, 1, 32), layer_name='conv1',
									 kernels_weights=array_conv1_kernels, biases_weights=array_conv1_biases)
	relu1 = Relu(name='relu1')
	maxpooling1 = MaxPoolingPaddingValid(input_shape=(28, 28), pool_size=(2, 2), layer_name='maxpooling1')
	conv2 = Conv2DPaddingSameStride1(input_shape=(14, 14, 32), kernels_shape=(5, 5, 32, 64), layer_name='conv2',
									 kernels_weights=array_conv2_kernels, biases_weights=array_conv2_biases)
	relu2 = Relu(name='relu2')
	maxpooling2 = MaxPoolingPaddingValid(input_shape=(14, 14), pool_size=(2, 2), layer_name='maxpooling2')
	flatten1 = Flatten(name='flatten1')
	fcn1 = FCN(input_length=7 * 7 * 64, output_length=1024, layer_name='fcn1',
			   kernels_weights=array_fcn1_kernels, biases_weights=array_fcn1_biases)
	relu3 = Relu(name='relu3')
	fcn2 = FCN(input_length=1024, output_length=10, layer_name='fcn2',
			   kernels_weights=array_fcn2_kernels, biases_weights=array_fcn2_biases)

	softmax1 = Softmax(name='softmax1')
	mnist_net.add_layer(conv1)
	mnist_net.add_layer(relu1)
	mnist_net.add_layer(maxpooling1)
	mnist_net.add_layer(conv2)
	mnist_net.add_layer(relu2)
	mnist_net.add_layer(maxpooling2)
	mnist_net.add_layer(flatten1)
	mnist_net.add_layer(fcn1)
	mnist_net.add_layer(relu3)
	mnist_net.add_layer(fcn2)
	mnist_net.add_layer(softmax1)

	return mnist_net


def train_model(train_data_path, train_label_path, valid_data_path, valid_label_path, save_weights_dir,
				conv1_filters_path=None, conv1_bias_path=None, conv2_filters_path=None,
				conv2_bias_path=None, fcn1_kernels_path=None, fcn1_bias_path=None,
				fcn2_kernels_path=None, fcn2_bias_path=None):

	array_conv1_kernels = None
	if conv1_filters_path is not None:
		array_conv1_kernels = np.load(conv1_filters_path)

	array_conv1_biases = None
	if conv1_bias_path is not None:
		array_conv1_biases = np.load(conv1_bias_path)

	array_conv2_kernels = None
	if conv2_filters_path is not None:
		array_conv2_kernels = np.load(conv2_filters_path)

	array_conv2_biases = None
	if conv2_bias_path is not None:
		array_conv2_biases = np.load(conv2_bias_path)

	array_fcn1_kernels = None
	if fcn1_kernels_path is not None:
		array_fcn1_kernels = np.load(fcn1_kernels_path)

	array_fcn1_biases = None
	if fcn1_bias_path is not None:
		array_fcn1_biases = np.load(fcn1_bias_path)

	array_fcn2_kernels = None
	if fcn2_kernels_path is not None:
		array_fcn2_kernels = np.load(fcn2_kernels_path)

	array_fcn2_biases = None
	if fcn2_bias_path is not None:
		array_fcn2_biases = np.load(fcn2_bias_path)

	mnist_net = construct_model(array_conv1_kernels, array_conv1_biases, array_conv2_kernels,
								array_conv2_biases, array_fcn1_kernels, array_fcn1_biases,
								array_fcn2_kernels, array_fcn2_biases)

	train_data = np.load(train_data_path)
	valid_data = np.load(valid_data_path)

	train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
	valid_data = valid_data.reshape(valid_data.shape[0], 28, 28, 1)

	train_label = np.load(train_label_path)
	valid_label = np.load(valid_label_path)

	# print train_label.shape
	# print train_label[0]

	mnist_net.train(train_data=train_data, train_label=train_label, train_batch=64, valid_data=valid_data,
					valid_label=valid_label, valid_batch=32, epochs=20, save_weights_dir=save_weights_dir)


def classify_data(conv1_filters_path, conv1_bias_path, conv2_filters_path, conv2_bias_path, fcn1_kernels_path,
				  fcn1_bias_path, fcn2_kernels_path, fcn2_bias_path, test_imgs_dir):

	array_conv1_kernels = np.load(conv1_filters_path)
	array_conv1_biases = np.load(conv1_bias_path)
	array_conv2_kernels = np.load(conv2_filters_path)
	array_conv2_biases = np.load(conv2_bias_path)
	array_fcn1_kernels = np.load(fcn1_kernels_path)
	array_fcn1_biases = np.load(fcn1_bias_path)
	array_fcn2_kernels = np.load(fcn2_kernels_path)
	array_fcn2_biases = np.load(fcn2_bias_path)

	mnist_net = construct_model(array_conv1_kernels, array_conv1_biases, array_conv2_kernels,
								array_conv2_biases, array_fcn1_kernels, array_fcn1_biases,
								array_fcn2_kernels, array_fcn2_biases)

	list_imgs = os.listdir(test_imgs_dir)
	imgs_num = len(list_imgs)

	for i in range(imgs_num):
		img = os.path.join(test_imgs_dir, list_imgs[i])
		array_img = cv2.imread(img)
		if array_img.shape[0] != 28 or array_img.shape[1] != 28:
			array_img = cv2.resize(array_img, 28, 28)
		if array_img.shape[2] == 3:
			tmp_trans = np.array([0.30, 0.59, 0.11])
			array_img = np.dot(array_img, tmp_trans)
			array_img = np.expand_dims(array_img, axis=2)
		array_img = np.expand_dims(array_img, axis=0)
		predict_ret = mnist_net.predict(array_img)
		print 'img:{} predict number is:{}'.format(list_imgs[i], np.argmax(predict_ret))

if __name__ == '__main__':
	"""
	train_model(train_data_path="/home/wepleo/.keras/datasets/trainInps.npy",
				train_label_path="/home/wepleo/.keras/datasets/trainTargs.npy",
				valid_data_path="/home/wepleo/.keras/datasets/testInps.npy",
				valid_label_path="/home/wepleo/.keras/datasets/testTargs.npy",
				save_weights_dir='/home/wepleo/learn_keras/dl_python_wepleo/save_weights',
				conv1_filters_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/conv1_filters.npy",
				conv1_bias_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/conv1_bias.npy")
				#conv2_filters_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/conv2_filters.npy",
				#conv2_bias_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/conv2_bias.npy",
				#fcn1_kernels_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/fcn1_kernels.npy",
				#fcn1_bias_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/fcn1_bias.npy",
				#fcn2_kernels_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/fcn2_kernels.npy")
				#fcn2_bias_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/fcn2_bias.npy")
	"""
	classify_data(conv1_filters_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/conv1_filters.npy",
				conv1_bias_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/conv1_bias.npy",
				conv2_filters_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/conv2_filters.npy",
				conv2_bias_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/conv2_bias.npy",
				fcn1_kernels_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/fcn1_kernels.npy",
				fcn1_bias_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/fcn1_bias.npy",
				fcn2_kernels_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/fcn2_kernels.npy",
				fcn2_bias_path="/home/wepleo/learn_keras/dl_python_wepleo/save_weights/fcn2_bias.npy",
				test_imgs_dir = "/home/wepleo/learn_keras/dl_python_wepleo/test_img")
