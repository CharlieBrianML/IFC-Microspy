#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# sr.py
# Neural Network
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import interfaceTools as it
import cv2
#os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

model = None

def preprocess_image(image_path):
	""" Loads image from path and preprocesses to make it model ready
	  Args:
		image_path: Path to the image file
	"""
	hr_image = tf.image.decode_image(tf.io.read_file(image_path))
	# If PNG, remove the alpha channel. The model only supports
	# images with 3 color channels.
	if hr_image.shape[-1] == 4:
		hr_image = hr_image[...,:-1]
	hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
	hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
	hr_image = tf.cast(hr_image, tf.float32)
	return tf.expand_dims(hr_image, 0)
	
def save_image(image, filename):
	"""
	Saves unscaled Tensor Images.
	Args:
	  image: 3D image tensor. [height, width, channels]
	  filename: Name of the file to save to.
	"""
	if not isinstance(image, Image.Image):
		image = tf.clip_by_value(image, 0, 255)
		image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
	image.save("%s.jpg" % filename)
	print("Saved as %s.jpg" % filename)
		
# Defining helper functions
def downscale_image(image):
	"""
	  Scales down images using bicubic downsampling.
	  Args:
		  image: 3D or 4D tensor of preprocessed image
	"""
	image_size = []
	if len(image.shape) == 3:
		image_size = [image.shape[1], image.shape[0]]
	else:
		raise ValueError("Dimension mismatch. Can work only on single image.")

	image = tf.squeeze(tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8))
	lr_image = np.asarray(Image.fromarray(image.numpy()).resize([image_size[0] // 4, image_size[1] // 4],Image.BICUBIC))
	lr_image = tf.expand_dims(lr_image, 0)
	lr_image = tf.cast(lr_image, tf.float32)
	return lr_image
	
def model_NeuralNetwork(image_path):
	global model
	hr_image = preprocess_image(image_path)
	lr_image = downscale_image(tf.squeeze(hr_image))
	
	# fake_image = model(hr_image)
	# fake_image = tf.squeeze(fake_image)

	fake_image = model(lr_image)
	fake_image = tf.squeeze(fake_image)
	return fake_image, hr_image
	
		
#%matplotlib inline
def plot_image(image, title=""):
	"""
	Plots images from image tensors.
	Args:
	  image: 3D image tensor. [height, width, channels].
	  title: Title to display in the plot.
	"""
	image = np.asarray(image)
	image = tf.clip_by_value(image, 0, 255)
	image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
	plt.imshow(image)
	plt.axis("off")
	plt.title(title)
	
def tensor_to_array(image):
	image = np.asarray(image)
	image = tf.clip_by_value(image, 0, 255)
	image = Image.fromarray(tf.cast(image, tf.uint8).numpy())	
	return image

def nn(dev_img, img_tensor):
	import tifffile
	import interfaceTools as it
	global model
	
	if not(os.path.isdir('training_deconv')) and not(os.path.isdir('training_set')):
		os.mkdir('training_deconv')
		os.mkdir('training_set')
		print(dev_img.shape)
		print(img_tensor.shape)
		di = np.zeros((dev_img.shape[2],dev_img.shape[3],3))
		it = np.zeros((img_tensor.shape[2],img_tensor.shape[3],3))
		
		for c in range(dev_img.shape[1]):
			for z in range(dev_img.shape[0]):
				di[:,:,0] = np.uint8(dev_img[z,c,:,:])
				di[:,:,1] = np.uint8(dev_img[z,c,:,:])
				di[:,:,2] = np.uint8(dev_img[z,c,:,:])
				cv2.imwrite('training_deconv/'+str(c+1)+'_'+str(z+1)+'.jpg', di)
				
		for c in range(img_tensor.shape[0]):
			for z in range(img_tensor.shape[1]):
				it[:,:,0] = np.uint8(img_tensor[c,z,:,:])
				it[:,:,1] = np.uint8(img_tensor[c,z,:,:])
				it[:,:,2] = np.uint8(img_tensor[c,z,:,:])
				cv2.imwrite('training_set/'+str(c+1)+'_'+str(z+1)+'.jpg', it)			
	
	info = np.load('info.npy', allow_pickle=True).item()
	img_output = np.zeros((info['z'], info['c'], info['x'], info['y']))

	imgs_deconv = os.listdir('training_deconv')
	
	# Path model 
	SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

	model = hub.load(SAVED_MODEL_PATH)
	
	if not(os.path.isdir('output_NN')):
		os.mkdir('output_NN')	

	start = time.time()
	for image in imgs_deconv:
		fake_image, hr_image = model_NeuralNetwork('training_deconv/'+image)
		img_array = tensor_to_array(tf.squeeze(fake_image))
		plt.imsave("output_NN/neural_network_"+image.split('.')[0]+'.png', img_array, cmap='gray')
		c = int(image.split('.')[0].split('_')[0])-1
		z = int(image.split('.')[0].split('_')[1])-1
		print('Processing: ', image, '\tc: ',c,'z: ',z)
		img_output[z,c,:,:] = np.asarray(img_array)[:,:,0]
	print("Time Taken: %f" % (time.time() - start))	
	
	import interfaceTools as it
	nnimg = it.NewWindow('Neural Network ')
	it.windows_img.append(nnimg)
	nnimg.desplay_image('Neural Network ', img_output)
