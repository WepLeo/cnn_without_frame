#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import common_util as util
import time


class Conv2DPaddingSameStride1:
	def __init__(self, input_shape, kernels_shape, layer_name,
	             kernels_initializer='glorot_uniform', biases_initializer='zeros',
				 learning_rate=0.001, momentum=0.9,
				 kernels_weights=None, biases_weights=None):
		"""
		:param input_shape: (rows, cols, input_channels)
		:param kernels_shape: (rows, cols, input_channels, number)
		:param layer_name:
		:param kernels_initializer:
		:param biases_initializer:
		:param learning_rate:
		:param momentum:
		:param kernels_weights: trained kernels, numpy.ndarray
		:param biases_weights: trained biases, numpy.ndarray
		"""

		self.input_shape = util.check_tuple(input_shape, length=3, name='input_shape')
		assert kernels_shape[0] == kernels_shape[1]
		self.kernels_shape = util.check_tuple(kernels_shape, length=4, name='kernels_shape')
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.layer_name = layer_name

		self.kernels = util.init_parameter(self.kernels_shape, kernels_initializer, kernels_weights)

		# print '\n===========%s===self.kernels========' % self.layer_name
		# print self.kernels

		biases_size = (self.kernels_shape[3],)
		self.biases = util.init_parameter(biases_size, biases_initializer, biases_weights)
		self.output_shape = self.conv_layer_output_shape()

		self.prev_gradient_kernels = np.zeros_like(self.kernels)
		self.prev_gradient_biases = np.zeros_like(self.biases)

	def conv_layer_output_shape(self):
		"""
		:return: output_shape (rows, cols, self.kernel_shape[2])
		"""
		# round down in conv, round up in pool
		output_rows = self.input_shape[0]
		output_cols = self.input_shape[1]
		output_channels = self.kernels_shape[3]
		return output_rows, output_cols, output_channels

	def forward(self, input_tensor):
		"""
		:param input_tensor:numpy.ndarray (batch, self.input_shape)
		:return:output_tensor:numpy.ndarray (batch, self.output_shape)
		"""
		start_time = time.time()
		assert input_tensor.ndim == 4
		assert self.input_shape == (input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3])

		batch, rows, cols, channels = input_tensor.shape

		# save for backward
		self.input_tensor = input_tensor
		# padding 'same' o=(i+2p-k)/1 + 1,i=o
		padding_size = (self.kernels_shape[0] - 1) / 2
		# round down in conv ,round up in maxpooling
		padding_input_tensor = np.zeros((batch, rows+2*padding_size, cols+2*padding_size, channels))
		padding_input_tensor[:, padding_size:rows+padding_size, padding_size:cols+padding_size, :] = input_tensor

		output_rows, output_cols, output_channels = self.output_shape
		output_tensor = np.empty((batch, output_rows, output_cols, output_channels))

		for idx_row in range(output_rows):
			for idx_col in range(output_cols):
				sub_tensor = padding_input_tensor[:, idx_row:idx_row + self.kernels_shape[0],
				             idx_col:idx_col + self.kernels_shape[1], :]
				sub_tensor = np.expand_dims(sub_tensor, axis=4)
				sub_tensor = np.repeat(sub_tensor, output_channels, axis=4)
				# print 'sub_tensor.shape:{}'.format(sub_tensor.shape)
				batch_filters = np.expand_dims(self.kernels, axis=0)
				batch_filters = np.repeat(batch_filters, batch, axis=0)
				# print 'batch_filters.shape:{}'.format(batch_filters.shape)
				output_tensor[:, idx_row, idx_col, :] = np.sum(np.sum(np.sum(sub_tensor * batch_filters, axis=1),
				                                                          axis=1), axis=1) + self.biases

		# print '{} forward cost time:{}'.format(self.layer_name, time.time()-start_time)
		# print '\n===========%s====forward=======' % self.layer_name
		# print output_tensor
		return output_tensor

	def backward(self, residual_tensor):
		"""
		:param residual_tensor:numpy.ndarray (batch, self.output_shape)
		:return:residual_x:numpy.ndarray (batch, self.input_shape)
		"""
		start_time = time.time()
		assert residual_tensor.ndim == 4
		assert self.output_shape == (residual_tensor.shape[1], residual_tensor.shape[2], residual_tensor.shape[3])
		assert residual_tensor.shape[0] == self.input_tensor.shape[0]

		residual_batch, residual_rows, residual_cols, residual_channels = residual_tensor.shape
		gradient_biases = np.sum(np.sum(np.sum(residual_tensor, axis=0), axis=0), axis=0) / residual_batch
		gradient_kernels = np.zeros_like(self.kernels)
		residual_x = np.zeros_like(self.input_tensor)
		# padding residual_tensor for calculating gradient_filters and gradient_x,
		# the padding_size is (self.kernel_shape[0] - 1) / 2
		padding_size = (self.kernels_shape[0] - 1) / 2
		padding_residual_tensor = np.zeros((residual_batch, residual_rows+2*padding_size,
		                                    residual_cols+2*padding_size, residual_channels))
		padding_residual_tensor[:, padding_size:residual_rows+padding_size,
								padding_size:residual_cols+padding_size, :] = residual_tensor

		input_rows, input_cols, input_channels = self.input_shape

		# gradient_filters
		filter_rows, filter_cols, _, filter_out_channels = self.kernels_shape
		for idx_b in range(residual_batch):
			for idx_row in range(filter_rows):
				for idx_col in range(filter_cols):
					sub_residual_tensor_f = padding_residual_tensor[idx_b, idx_row:idx_row + input_rows,
					                        idx_col:idx_col + input_cols, :]
					rot180_sub_residual_tensor_f = np.rot90(sub_residual_tensor_f, k=2, axes=(0, 1))
					rot180_sub_residual_tensor_f = np.expand_dims(rot180_sub_residual_tensor_f, axis=2)
					rot180_sub_residual_tensor_f = np.repeat(rot180_sub_residual_tensor_f, input_channels, axis=2)
					# print 'sub_residual_tensor_f.shape:{}'.format(sub_residual_tensor_f.shape)
					rot180_single_input_tensor = np.rot90(self.input_tensor[idx_b], k=2, axes=(0, 1))
					rot180_single_input_tensor = np.expand_dims(rot180_single_input_tensor, axis=3)
					rot180_single_input_tensor = np.repeat(rot180_single_input_tensor, residual_channels, axis=3)
					# print 'rot180_single_input_tensor.shape:{}'.format(rot180_single_input_tensor.shape)

					gradient_kernels[idx_row, idx_col, :, :] += np.sum(
						np.sum(rot180_single_input_tensor * rot180_sub_residual_tensor_f, axis=0), axis=0)

		gradient_kernels /= residual_batch
		# residual_x
		for residual_channel in range(residual_channels):
			for in_row in range(input_rows):
				for in_col in range(input_cols):
					sub_residual_tensor_x = padding_residual_tensor[:, in_row:in_row + filter_rows,
					                        idx_col:idx_col + filter_cols, residual_channel]
					sub_residual_tensor_x = np.expand_dims(sub_residual_tensor_x, axis=3)
					sub_residual_tensor_x = np.repeat(sub_residual_tensor_x, input_channels, axis=3)
					# print 'sub_residual_tensor_x.shape:{}'.format(sub_residual_tensor_x.shape)

					rot180_single_filter = np.rot90(self.kernels[:, :, :, residual_channel], k=2, axes=(0, 1))
					rot180_single_filter = np.expand_dims(rot180_single_filter, axis=0)
					rot180_single_filter = np.repeat(rot180_single_filter, residual_batch, axis=0)
					# print 'rot180_single_filter.shape:{}'.format(rot180_single_filter.shape)
					residual_x[:, in_row, in_col, :] += np.sum(
						np.sum(rot180_single_filter * sub_residual_tensor_x, axis=1), axis=1)

		# residual_x /= filter_out_channels

		print '\n===========%s====self.kernels=======' % self.layer_name
		print self.kernels

		print '\n===========%s====gradient_kernels=======' % self.layer_name
		print gradient_kernels
		# update
		self.prev_gradient_kernels = self.prev_gradient_kernels * self.momentum - self.learning_rate * gradient_kernels

		self.kernels += self.prev_gradient_kernels
		# print '\n===========%s===========' % self.layer_name
		# print np.sum(
		# np.sum(rot180_single_input_tensor * rot180_sub_residual_tensor_f, axis=0), axis=0)
		self.prev_gradient_biases = self.prev_gradient_biases * self.momentum - self.learning_rate * gradient_biases
		self.biases += self.prev_gradient_biases

		print '{} backward cost time:{}'.format(self.layer_name, time.time() - start_time)

		return residual_x

	def save_weights(self, save_weights_dir):
		filters_save_path = save_weights_dir + '/' + self.layer_name + '_kernels.npy'
		bias_save_path = save_weights_dir + '/' + self.layer_name + '_biases.npy'
		np.save(filters_save_path, self.kernels)
		np.save(bias_save_path, self.biases)
