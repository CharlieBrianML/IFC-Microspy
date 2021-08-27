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
import cv2
#os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

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

def nn(dev_img, img_tensor):
	
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
	
	# Declaring Constants
	IMAGE_PATH = "training_deconv/1_7.jpg"
	SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

	imgdeconv=cv2.imread(IMAGE_PATH,0)
	img=cv2.imread('training_set/1_7.jpg',0)
	hr_image = preprocess_image(IMAGE_PATH)
	lr_image = downscale_image(tf.squeeze(hr_image))

	# Plotting Low Resolution Image
	plot_image(tf.squeeze(lr_image), title="Low Resolution")

	# Plotting Original Resolution image
	#plot_image(tf.squeeze(hr_image), title="Original Image")
	#save_image(tf.squeeze(hr_image), filename="Original Image")	

	model = hub.load(SAVED_MODEL_PATH)

	# start = time.time()
	# fake_image = model(hr_image)
	# fake_image = tf.squeeze(fake_image)
	# print("Time Taken: %f" % (time.time() - start))

	start = time.time()
	fake_image = model(lr_image)
	fake_image = tf.squeeze(fake_image)
	print("Time Taken: %f" % (time.time() - start))

	# # Plotting Super Resolution Image
	# plot_image(tf.squeeze(fake_image), title="Super Resolution")
	# save_image(tf.squeeze(fake_image), filename="Super Resolution")

	plot_image(tf.squeeze(fake_image), title="Neural Network")
	# Calculating PSNR wrt Original Image
	psnr = tf.image.psnr(tf.clip_by_value(fake_image, 0, 255),tf.clip_by_value(hr_image, 0, 255), max_val=255)
	print("PSNR Achieved: %f" % psnr)

	c=0
	z=0
	
	if not(os.path.isdir('output_NN')):
		os.mkdir('output_NN')

	plt.rcParams['figure.figsize'] = [15, 10]
	fig, axes = plt.subplots(1, 3)
	fig.tight_layout()
	plt.subplot(131)
	#plt.imshow(img_tensor[z,c,:,:], cmap='gray')
	plt.imshow(img, cmap='gray')
	plt.axis("off")
	plt.title('Original')
	#plot_image(img_tensor[z,c,:,:], title="Original")
	plt.subplot(132)
	save_image(tf.squeeze(hr_image), filename='output_NN/SuperResolution')
	fig.tight_layout()
	#plt.imshow(dev_img[c,z,:,:], cmap='gray')
	plt.imshow(imgdeconv, cmap='gray')
	plt.axis("off")
	plt.title('Deconvolution')	
	#plot_image(dev_img[c,z,:,:], "Deconvolution")
	plt.subplot(133)
	fig.tight_layout()
	plot_image(tf.squeeze(fake_image), "Neural Network")
	plt.savefig("output_NN/NeuralNetwork_c"+str(c)+"_z"+str(z)+".jpg", bbox_inches="tight")
	print("PSNR: %f" % psnr)
	plt.show()
	#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
