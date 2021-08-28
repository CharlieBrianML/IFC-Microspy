#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# ifc_srm.py
# File main
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
	#SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
	hr_image = preprocess_image(image_path)
	lr_image = downscale_image(tf.squeeze(hr_image))
	#model = hub.load(SAVED_MODEL_PATH)

	# start = time.time()
	# fake_image = model(hr_image)
	# fake_image = tf.squeeze(fake_image)
	# print("Time Taken: %f" % (time.time() - start))

	start = time.time()
	fake_image = model(lr_image)
	fake_image = tf.squeeze(fake_image)
	print("Time Taken: %f" % (time.time() - start))
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
	
	#img_output = np.zeros(dev_img.shape)
	img_output = np.zeros((26, 4, 800, 800))
	'''
	print('imagen a guardar: ', dev_img.shape)
	tifffile.imsave('training_deconv/image.tif', dev_img, imagej=True)					
	'''
	imgs_deconv = os.listdir('training_deconv')
	
	# Declaring Constants
	#IMAGE_PATH = "training_deconv/1_6.jpg"
	SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

	#imgdeconv=cv2.imread(IMAGE_PATH,0)
	#img=cv2.imread('training_set/1_6.jpg',0)
	'''
	from skimage import io
	img = io.imread('training_deconv/image.tif')
	
	print('imagen leida: ', img.shape)
	'''
	# hr_image = preprocess_image(IMAGE_PATH)
	# lr_image = downscale_image(tf.squeeze(hr_image))

	# Plotting Low Resolution Image
	#plot_image(tf.squeeze(lr_image), title="Low Resolution")

	# Plotting Original Resolution image
	#plot_image(tf.squeeze(hr_image), title="Original Image")
	#save_image(tf.squeeze(hr_image), filename="Original Image")	

	model = hub.load(SAVED_MODEL_PATH)

	# # start = time.time()
	# # fake_image = model(hr_image)
	# # fake_image = tf.squeeze(fake_image)
	# # print("Time Taken: %f" % (time.time() - start))

	# start = time.time()
	# fake_image = model(lr_image)
	# fake_image = tf.squeeze(fake_image)
	# print("Time Taken: %f" % (time.time() - start))

	# # Plotting Super Resolution Image
	# plot_image(tf.squeeze(fake_image), title="Super Resolution")
	# save_image(tf.squeeze(fake_image), filename="Super Resolution")
	
	if not(os.path.isdir('output_NN')):
		os.mkdir('output_NN')	

	for image in imgs_deconv:
		print('Processing: ', image)
		fake_image, hr_image = model_NeuralNetwork('training_deconv/'+image)
		img_array = tensor_to_array(tf.squeeze(fake_image))
		plt.imsave("output_NN/neural_network_"+image.split('.')[0]+'.png', img_array, cmap='gray')
		print(type(np.asarray(img_array)))
		c = int(image.split('.')[0].split('_')[0])-1
		z = int(image.split('.')[0].split('_')[1])-1
		print('Channel; ', c)
		print('Z; ', z)
		#img_output[z,c,:,:] = cv2.merge(np.asarray(img_array))
		img_output[z,c,:,:] = np.asarray(img_array)[:,:,0]
		#cv2.imwrite("output_NN/neural_network_"+image.split('.')[0]+'.png', tensor_to_array(tf.squeeze(fake_image)))
	'''
	for c in range(img.shape[3]):
		for z in range(img.shape[0]):
			print('Processing channel '+str(c)+' z: ', str(z))
			fake_image, hr_image = model_NeuralNetwork(img)
			img_array = tensor_to_array(tf.squeeze(fake_image))
			plt.imsave("output_NN/neural_network_"+image.split('.')[0]+'.png', img_array, cmap='gray')
			print(type(np.asarray(img_array)))
			img_output[] = np.asarray(img_array)
	'''	
	#nnimg = it.NewWindow('Neural Network '+it.file.split('/')[len(it.file.split('/'))-1])
	#nnimg.desplay_image('Neural Network '+it.file.split('/')[len(it.file.split('/'))-1], )	
	import interfaceTools as it
	nnimg = it.NewWindow('Neural Network ')
	nnimg.desplay_image('Neural Network ', img_output)			
	
	#fake_image, hr_image = model_NeuralNetwork(IMAGE_PATH)

	# plot_image(tf.squeeze(fake_image), title="Neural Network")
	# # Calculating PSNR wrt Original Image
	# psnr = tf.image.psnr(tf.clip_by_value(fake_image, 0, 255),tf.clip_by_value(hr_image, 0, 255), max_val=255)
	# print("PSNR Achieved: %f" % psnr)

	c=1
	z=0

	# plt.rcParams['figure.figsize'] = [15, 10]
	# fig, axes = plt.subplots(1, 3)
	# fig.tight_layout()
	# plt.subplot(131)
	# #plt.imshow(img_tensor[z,c,:,:], cmap='gray')
	# plt.imshow(img, cmap='gray')
	# plt.axis("off")
	# plt.title('Original')
	# #plot_image(img_tensor[z,c,:,:], title="Original")
	# plt.subplot(132)
	# save_image(tf.squeeze(hr_image), filename='output_NN/SuperResolution')
	# fig.tight_layout()
	# #plt.imshow(dev_img[c,z,:,:], cmap='gray')
	# plt.imshow(imgdeconv, cmap='gray')
	# plt.axis("off")
	# plt.title('Deconvolution')	
	# #plot_image(dev_img[c,z,:,:], "Deconvolution")
	# plt.subplot(133)
	# fig.tight_layout()
	# plot_image(tf.squeeze(fake_image), "Neural Network")
	# plt.savefig("output_NN/NeuralNetwork_c"+str(c)+"_z"+str(z)+".jpg", bbox_inches="tight")
	# print("PSNR: %f" % psnr)
	# plt.show()
