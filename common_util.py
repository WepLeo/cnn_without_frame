#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def check_tuple(input_val, length, name=None):
	if isinstance(input_val, tuple):
		if len(input_val) == length:
			return input_val
		else:
			raise ValueError('The `' + name + '` argument must be a tuple with length:' + str(length) + ', Received: ' + str(input_val))
	else:
		try:
			tuple_val = tuple(input_val)
		except TypeError:
			raise ValueError('The `' + name + '` argument must be a tuple, Received: ' + str(input_val))
	return tuple_val


def check_padding(padding_value):
	lower_padding = padding_value.lower()
	padding_candidate = {'valid', 'same'}

	if lower_padding not in padding_candidate:
		raise ValueError('The `padding` argument must be one of: ' + str(padding_candidate) + '.'
						 'Received: ' + str(padding_value))
	return lower_padding


def get_padding_2d(padding_value='same', filter_size=(2, 2)):
	'''
	:param padding_value:   'valid' or 'same'
	:param filter_size:     tuple (rows, cols)
	:return: padding_size:  tuple (rows, cols)
	'''

	padding_type = check_padding(padding_value)

	if padding_type == 'valid':
		return 0, 0
	elif padding_type == 'same':
		padding_rows = (filter_size[0] - 1) / 2
		padding_cols = (filter_size[1] - 1) / 2
		return padding_rows, padding_cols


def check_init(init_name):
	if init_name is not None:
		lower_init_name = init_name.lower()
		init_name_candidate = {'zeros', 'ones', 'random_normal', 'glorot_uniform'}

		if lower_init_name not in init_name_candidate:
			raise ValueError('The `init_name` argument must be one of: ' + str(init_name_candidate) + '.'
							 'Received: ' + str(init_name))
		return lower_init_name
	else:
		return None


def init_parameter(init_shape, init_name=None, array_weights=None):
	if array_weights is not None:
		print init_shape
		print array_weights.shape
		assert init_shape == array_weights.shape
		return array_weights
	else:
		init_type = check_init(init_name)
		if init_type == 'zeros':
			return np.zeros(init_shape)
		elif init_type == 'ones':
			return np.ones(init_shape)
		elif init_type == 'random_normal':
			return np.random.normal(size=init_shape)
		elif init_type == 'glorot_uniform':
			if len(init_shape) in (2, 4):
				if len(init_shape) == 2:
					fan_in = init_shape[0]
					fan_out = init_shape[1]
				elif len(init_shape) == 4:
					fan_in = init_shape[2] * init_shape[0] * init_shape[1]
					fan_out = init_shape[3] * init_shape[0] * init_shape[1]

				limit = np.sqrt(6.0 / (fan_out + fan_in))

				return np.random.uniform(-limit, limit, init_shape)
			else:
				raise ValueError('The length of `init_shape` argument must be one of: 2, 4. Received: ' + len(init_shape))





# code from keras2.0.4 keras.np_util
def to_categorical(y, num_classes=None):
	"""Converts a class vector (integers) to binary class matrix.

	E.g. for use with categorical_crossentropy.

	# Arguments
		y: class vector to be converted into a matrix
			(integers from 0 to num_classes).
		num_classes: total number of classes.

	# Returns
		A binary matrix representation of the input.
	"""
	y = np.array(y, dtype='int').ravel()
	if not num_classes:
		num_classes = np.max(y) + 1
	n = y.shape[0]
	categorical = np.zeros((n, num_classes))
	categorical[np.arange(n), y] = 1
	return categorical
