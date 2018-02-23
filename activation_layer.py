#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


# https://github.com/hsmyy/zhihuzhuanlan/blob/master/cnn.py
class Relu:
	def __init__(self, name):
		self.name = name

	def forward(self, input_tensor):
		self.input_tensor = input_tensor
		output_tensor = input_tensor.copy()
		output_tensor[input_tensor <= 0] = 0
		return output_tensor

	def backward(self, residual_tensor):
		residual_x = residual_tensor.copy()
		residual_x[self.input_tensor < 0] = 0
		return residual_x


class Softmax:
	def __init__(self, name):
		self.layer_name = name

	def forward(self, input_tensor):
		"""
		:param input_tensor: (batch, nb_classes)
		:return: output_tensor: (batch, nb_classes)
		"""
		# print 'input_tensor.shape:{}'.format(input_tensor.shape)
		assert input_tensor.ndim == 2
		# print '\n===========%s===forward ========' % self.layer_name
		# print input_tensor

		exp_out = np.exp(input_tensor)
		self.one_hot_output = exp_out / np.expand_dims(np.sum(exp_out, axis=1), axis=1)

		return self.one_hot_output

	def backward(self, residual_tensor):
		"""
		softmax should be the last layer, and the residual_tensor is the label, then get:
		`the residual_x = self.one_hot_output - residual_tensor`
		the format is derived from Cross Entropy Loss
		:param residual_tensor: (batch, nb_classes)
		:return: residual_x:(batch, nb_classes)
		"""
		# print '\n===========%s=====backward======' % self.layer_name
		# print self.one_hot_output - residual_tensor

		assert self.one_hot_output.shape == residual_tensor.shape
		return self.one_hot_output - residual_tensor
