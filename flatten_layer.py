#! /usr/bin/python
# -*- coding: utf-8 -*-


class Flatten:
	def __init__(self, name):
		self.layer_name = name
		self.batch = 0
		self.rows = 0
		self.cols = 0
		self.channels = 0

	def __set__(self, instance, value):
		if instance == self.batch:
			self.batch = value
		elif instance == self.rows:
			self.rows = value
		elif instance == self.cols:
			self.cols = value
		elif instance == self.channels:
			self.channels = value

	def forward(self, input_tensor):
		assert input_tensor.ndim == 4
		self.__set__(self.batch, input_tensor.shape[0])
		self.__set__(self.rows, input_tensor.shape[1])
		self.__set__(self.cols, input_tensor.shape[2])
		self.__set__(self.channels, input_tensor.shape[3])

		return input_tensor.reshape(self.batch, self.rows * self.cols * self.channels)

	def backward(self, residual_tensor):
		assert residual_tensor.ndim == 2
		# print '\n===========%s===========' % self.layer_name
		# print residual_tensor
		return residual_tensor.reshape(self.batch, self.rows, self.cols, self.channels)
