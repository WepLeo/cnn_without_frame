#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import common_util as util
import time

class FCN:
	def __init__(self, input_length, output_length, layer_name='fcn',
	             kernels_initializer='glorot_uniform', biases_initializer='zeros',
	             learning_rate=0.01, momentum=0.9,
	             kernels_weights=None, biases_weights=None):
		"""
		:param input_length: int
		:param output_length: int
		:param layer_name:
		:param kernels_initializer:
		:param biases_initializer:
		:param learning_rate:
		:param momentum:
		:param kernels_weights: trained weights numpy.ndarray
		:param biases_weights: trained bias numpy.ndarray
		"""
		self.input_length = input_length
		self.output_length = output_length
		self.kernels = util.init_parameter((self.input_length, self.output_length), kernels_initializer, kernels_weights)
		biases_size = (self.output_length,)
		self.biases = util.init_parameter(biases_size, biases_initializer, biases_weights)
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.layer_name = layer_name
		self.prev_grad_kernels = np.zeros_like(self.kernels)
		self.prev_grad_biases = np.zeros_like(self.biases)

	def forward(self, input_tensor):
		"""
		:param input_tensor: (batch, self.input_length)
		:return:output_tensor: (batch, self.output_length)
		"""
		start_time = time.time()
		assert input_tensor.ndim == 2
		assert self.kernels.shape[0] == input_tensor.shape[1]
		output_tensor = np.dot(input_tensor, self.kernels) + self.biases
		self.input_tensor = input_tensor
		# print '\n===========%s====forward=======' % self.layer_name
		# print output_tensor
		# print '{} forward cost time:{}'.format(self.layer_name, time.time() - start_time)
		return output_tensor

	def backward(self, residual_tensor):
		"""
		:param residual_tensor: (batch, self.output_length)
		:return:residual_x: (batch, self.input_length)
		"""
		start_time = time.time()
		assert residual_tensor.ndim == 2
		assert residual_tensor.shape[1] == self.output_length
		assert residual_tensor.shape[0] == self.input_tensor.shape[0]

		batch = residual_tensor.shape[0]
		grad_kernels = np.dot(self.input_tensor.T, residual_tensor) / batch
		grad_biases = np.sum(residual_tensor, axis=0) / batch
		residual_x = np.dot(residual_tensor, self.kernels.T)
		self.prev_grad_kernels = self.prev_grad_kernels * self.momentum - self.learning_rate * grad_kernels
		self.kernels += self.prev_grad_kernels

		# print '\n===========%s=====backward======' % self.layer_name
		# print self.prev_grad_kernels

		self.prev_grad_biases = self.prev_grad_biases * self.momentum - self.learning_rate * grad_biases
		self.biases += self.prev_grad_biases
		# print '{} backward cost time:{}'.format(self.layer_name, time.time() - start_time)
		return residual_x

	def save_weights(self, save_weights_dir):
		kernels_save_path = save_weights_dir + '/' + self.layer_name + '_kernels.npy'
		bias_save_path = save_weights_dir + '/' + self.layer_name + '_biases.npy'
		np.save(kernels_save_path, self.kernels)
		np.save(bias_save_path, self.biases)
