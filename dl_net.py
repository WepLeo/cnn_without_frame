#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from conv_layer import Conv2DPaddingSameStride1
from maxpooling_layer import MaxPoolingPaddingValid
from flatten_layer import Flatten
from activation_layer import Relu, Softmax

MIN = 0.0000001


class Net:
	def __init__(self, name='dl_net', class_num = 10):
		self.name = name
		self.class_num = class_num
		self.layers = []

	def add_layer(self, layer):
		self.layers.append(layer)

	def train(self, train_data, train_label, train_batch, valid_data,
	          valid_label, valid_batch, epochs, save_weights_dir, shuffle=True):
		"""
		:param train_data:  (total_num, rows, cols, channels)
		:param train_label: (total_num, nb_class)
		:param train_batch: int 16/32/64/128...
		:param valid_data:  (total_num, rows, cols, channels)
		:param valid_label: (total_num, nb_class)
		:param valid_batch: int 16/32/64/128...
		:param epochs: int
		:param save_weights_dir:
		:param shuffle: bool, whether shuffle train_data
		:return:
		"""

		assert 4 == train_data.ndim
		assert train_data.shape[0] == train_label.shape[0]

		assert 4 == valid_data.ndim
		assert valid_data.shape[0] == valid_label.shape[0]

		train_data = 1.0 * (train_data - np.min(train_data)) / (np.max(train_data) - np.min(train_data))
		valid_data = 1.0 * (valid_data - np.min(valid_data)) / (np.max(valid_data) - np.min(valid_data))

		total_train_num = train_data.shape[0]
		if shuffle:
			index = [i for i in range(total_train_num)]
			np.random.shuffle(index)
			train_data = train_data[index]
			train_label = train_label[index]

		for epoch in range(epochs):
			print 'epoch:{}/{}'.format(epoch, epochs)
			for batch_iter in range(0, total_train_num, train_batch):
				if batch_iter + train_batch < total_train_num:
					self.train_one_batch(train_data[batch_iter:batch_iter+train_batch, :, :, :],
					                     train_label[batch_iter:batch_iter+train_batch, :])
				else:
					self.train_one_batch(train_data[batch_iter:total_train_num, :, :, :],
					                     train_label[batch_iter:total_train_num, :])

			valid_avg_acc, valid_avg_loss = self.validate_model(valid_data, valid_label, valid_batch)
			print 'valid_avg_acc:{}, valid_avg_loss:{}'.format(valid_avg_acc, valid_avg_loss)

		layer_num = len(self.layers)
		for idx_layer in range(layer_num):
			if hasattr(self.layers[idx_layer], 'save_weights'):
				self.layers[idx_layer].save_weights(save_weights_dir)

	def train_one_batch(self, batch_train_data, batch_train_label):
		"""
		:param batch_train_data:(batch, rows, cols, channels)
		:param batch_train_label:(batch, nb_class)
		:return:
		"""
		layer_num = len(self.layers)
		input_tensor = batch_train_data

		for idx_layer in range(layer_num):
			output_tensor = self.layers[idx_layer].forward(input_tensor)
			input_tensor = output_tensor

		idx_max_out = np.argmax(output_tensor, axis=1)
		idx_max_label = np.argmax(batch_train_label, axis=1)
		avg_acc = np.sum(idx_max_out == idx_max_label) / float(batch_train_label.shape[0])
		avg_loss = -np.sum(batch_train_label * np.log(output_tensor + MIN)) / float(batch_train_label.shape[0])

		print 'train_avg_acc:{}, train_avg_loss:{}'.format(avg_acc, avg_loss)

		residual_tensor = batch_train_label
		for idx_layer in range(layer_num-1, -1, -1):
			residual_x = self.layers[idx_layer].backward(residual_tensor)
			residual_tensor = residual_x

	def validate_model(self, valid_data, valid_label, valid_batch):
		"""
		:param valid_data:(total_num, rows, cols, channels)
		:param valid_label:(total_num, nb_class)
		:param valid_batch: int 16/32/64/128...
		:return:
		"""
		layer_num = len(self.layers)
		total_valid_num = valid_data.shape[0]
		avg_loss = 0
		avg_acc = 0

		for v_batch_iter in range(0, total_valid_num, valid_batch):
			if v_batch_iter + valid_batch < total_valid_num:
				batch_valid_data = valid_data[v_batch_iter:v_batch_iter + valid_batch, :, :, :]
				batch_valid_label = valid_label[v_batch_iter:v_batch_iter + valid_batch, :]
				input_tensor = batch_valid_data
				for idx_layer in range(layer_num):
					output_tensor = self.layers[idx_layer].forward(input_tensor)
					input_tensor = output_tensor

				idx_max_out = np.argmax(output_tensor, axis=1)
				idx_max_label = np.argmax(batch_valid_label, axis=1)
				avg_acc += np.sum(idx_max_out == idx_max_label)
				avg_loss += np.sum(batch_valid_label * np.log(output_tensor + MIN))
			else:
				batch_valid_data = valid_data[v_batch_iter:total_valid_num, :, :, :]
				batch_valid_label = valid_label[v_batch_iter:total_valid_num, :]
				input_tensor = batch_valid_data
				for idx_layer in range(layer_num):
					output_tensor = self.layers[idx_layer].forward(input_tensor)
					input_tensor = output_tensor

				idx_max_out = np.argmax(output_tensor, axis=1)
				idx_max_label = np.argmax(batch_valid_label, axis=1)
				avg_acc += np.sum(idx_max_out == idx_max_label)
				avg_loss += np.sum(batch_valid_label * np.log(output_tensor + MIN))

		avg_loss = -float(avg_loss) / total_valid_num
		avg_acc = float(avg_acc) / total_valid_num
		return avg_acc, avg_loss

	def predict(self, predict_data, batch_size=None):
		"""
		:param predict_data: (num, rows, cols, channels)
		:param batch_size:
		:return:classify_ret
		"""

		assert predict_data.ndim == 4
		layer_num = len(self.layers)
		predict_data_num = predict_data.shape[0]
		batch_size = batch_size if batch_size is not None else 1

		classify_ret = np.zeros((predict_data_num, self.class_num))

		for batch_iter in range(0, predict_data_num, batch_size):
			if batch_iter + batch_size < predict_data_num:
				batch_data = predict_data[batch_iter:batch_iter + batch_size, :, :, :]

				input_tensor = batch_data
				for idx_layer in range(layer_num):
					output_tensor = self.layers[idx_layer].forward(input_tensor)
					input_tensor = output_tensor

				idx_max_out = np.argmax(output_tensor, axis=1)
				print 'predict result:{}'.format(idx_max_out)
				classify_ret[batch_iter:batch_iter + batch_size, :] = output_tensor
			else:
				batch_data = predict_data[batch_iter:predict_data_num, :, :, :]
				input_tensor = batch_data
				for idx_layer in range(layer_num):
					output_tensor = self.layers[idx_layer].forward(input_tensor)
					input_tensor = output_tensor

				idx_max_out = np.argmax(output_tensor, axis=1)
				print 'predict result:{}'.format(idx_max_out)
				classify_ret[batch_iter:predict_data_num, :] = output_tensor

		return classify_ret
