#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import common_util as util
import math
import time

class MaxPoolingPaddingValid:
	def __init__(self, input_shape, pool_size, layer_name):
		"""
		:param input_shape: (rows, cols)
		:param pool_size: (rows, cols)
		:param layer_name:
		"""
		self.input_shape = util.check_tuple(input_shape, length=2, name='input_shape')
		self.pool_size = util.check_tuple(pool_size, length=2, name='pool_size')
		assert pool_size[0] == pool_size[1]
		self.layer_name = layer_name
		self.output_shape = self.max_pooling_layer_output_shape()

	def max_pooling_layer_output_shape(self):
		"""
		:return: output_shape (rows, cols)
		"""
		if self.pool_size[0] > 0 and self.pool_size[1] > 0:
			# round down in conv, round up in pool
			output_rows = int(math.ceil(float(self.input_shape[0]) / self.pool_size[0]))
			output_cols = int(math.ceil(float(self.input_shape[1]) / self.pool_size[1]))
			return output_rows, output_cols
		else:
			raise ValueError('`strides` should be greater than zeros')

	def forward(self, input_tensor):
		"""
		:param input_tensor: (batch, rows, cols, channels)
		:return: output_tensor (batch, self.output_shape, channels)
		"""
		start_time = time.time()
		assert input_tensor.ndim == 4
		assert self.input_shape == (input_tensor.shape[1], input_tensor.shape[2])

		batch, input_rows, input_cols, channels = input_tensor.shape

		output_rows, output_cols = self.output_shape
		pool_rows, pool_cols = self.pool_size[0], self.pool_size[1]

		self.max_flag = np.zeros_like(input_tensor)
		output_tensor = np.empty((batch, output_rows, output_cols, channels))
		for idx_b in range(batch):
			for idx_row in range(output_rows):
				pool_rows_t = pool_rows if (idx_row + 1) * pool_rows <= input_rows else input_rows - idx_row * pool_rows
				for idx_col in range(output_cols):
					pool_cols_t = pool_cols if (idx_col + 1) * pool_cols <= input_cols else input_cols - idx_col * pool_cols
					for idx_channel in range(channels):
						max_idx = np.argmax(input_tensor[idx_b, idx_row * pool_rows: idx_row * pool_rows + pool_rows_t,
						                    idx_col * pool_cols: idx_col * pool_cols + pool_cols_t, idx_channel])
						offset_row, offset_col = np.unravel_index(max_idx, (pool_rows_t, pool_cols_t))
						#offset_row = max_idx / pool_cols_t
						#offset_col = max_idx % pool_cols_t
						self.max_flag[idx_b, idx_row * pool_rows + offset_row, idx_col * pool_cols + offset_col, idx_channel] = 1
						output_tensor[idx_b, idx_row, idx_col, idx_channel] = input_tensor[idx_b,
						                                                         idx_row * pool_rows + offset_row,
						                                                         idx_col * pool_cols + offset_col,
						                                                         idx_channel]
		# print '{} forward cost time:{}'.format(self.layer_name, time.time() - start_time)
		return output_tensor

	def backward(self, residual_tensor):
		"""
		:param residual_tensor:(batch, self.output_shape, channels)
		:return:residual_x:(batch, self.input_shape, channels)
		"""
		start_time = time.time()
		assert residual_tensor.ndim == 4
		assert self.output_shape == (residual_tensor.shape[1], residual_tensor.shape[2])
		assert self.max_flag.shape[0] == residual_tensor.shape[0]
		assert self.max_flag.shape[3] == residual_tensor.shape[3]

		batch, input_rows, input_cols, channels = self.max_flag.shape
		out_rows, out_cols = residual_tensor.shape[1], residual_tensor.shape[2]
		pool_rows, pool_cols = self.pool_size[0], self.pool_size[1]

		residual_x = np.zeros_like(self.max_flag)
		for idx_row in range(out_rows):
			pool_rows_t = pool_rows if (idx_row + 1) * pool_rows <= input_rows else input_rows - idx_row * pool_rows
			for idx_col in range(out_cols):
				pool_cols_t = pool_cols if (idx_col + 1) * pool_cols <= input_cols else input_cols - idx_col * pool_cols
				pool_size_t = residual_tensor[:, idx_row, idx_col, :].reshape(batch, 1, 1, channels)

				residual_x[:, idx_row * pool_rows: idx_row * pool_rows + pool_rows_t, idx_col * pool_cols: idx_col * pool_cols + pool_cols_t, :] = pool_size_t

		'''
		batch, input_rows, input_cols, channels = self.max_flag.shape
		out_rows, out_cols = residual_tensor.shape[1], residual_tensor.shape[2]
		pool_rows, pool_cols = self.pool_size[0], self.pool_size[1]

		residual_x = np.zeros_like(self.max_flag)
		for idx_b in range(batch):
			for idx_row in range(out_rows):
				pool_rows_t = pool_rows if (idx_row + 1) * pool_rows <= input_rows else input_rows - idx_row * pool_rows
				for idx_col in range(out_cols):
					pool_cols_t = pool_cols if (idx_col + 1) * pool_cols <= input_cols else input_cols - idx_col * pool_cols
					for idx_channel in range(channels):
						# max_idx = np.argmax(self.max_flag[idx_b, idx_row * pool_rows: idx_row * pool_rows + pool_rows_t,
						#                    idx_col * pool_cols: idx_col * pool_cols + pool_cols_t, idx_channel])
						# offset_row = max_idx / pool_cols_t
						# offset_col = max_idx % pool_cols_t

						residual_x[idx_b, idx_row * pool_rows: idx_row * pool_rows + pool_rows_t,
						           idx_col * pool_cols: idx_col * pool_cols + pool_cols_t, :] = residual_tensor[idx_b, idx_row, idx_col, idx_channel]
		'''
		residual_x[self.max_flag == 0] = 0

		print '{} backward cost time:{}'.format(self.layer_name, time.time() - start_time)

		return residual_x
