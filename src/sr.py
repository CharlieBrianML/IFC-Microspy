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
import cv2
import src.interfaceTools as it

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

def nn(img_tensor):
	import tifffile
	import src.interfaceTools as it
	from .imageFunctions import istiffRGB
	global model
	
	if(len(it.windows_img)>0):
	
		if not(os.path.isdir('src/cache/training_set')):
			os.mkdir('src/cache/training_set')
			
		img_path = 'src/cache/training_set/training_'+it.windows_img[-1].nameWindow
	
		# Creation of the training set
		if not(os.path.isdir(img_path)):
			os.mkdir(img_path)
			print(img_tensor.shape)
			
			if (img_tensor.ndim == 2):
				img = np.zeros((img_tensor.shape[0],img_tensor.shape[1],3))
				img[:,:,0] = np.uint8(img_tensor)
				img[:,:,1] = np.uint8(img_tensor)
				img[:,:,2] = np.uint8(img_tensor)
				cv2.imwrite(img_path+'/1.jpg', img)
			if (img_tensor.ndim == 3):
				if (istiffRGB(img_tensor.shape)):
					cv2.imwrite(img_path+'/1.jpg', img_tensor)
				elif (img_tensor.shape[0]>4):
					img = np.zeros((img_tensor.shape[1],img_tensor.shape[2],3))
					for z in range(img_tensor.shape[0]):
						img[:,:,0] = np.uint8(img_tensor[z,:,:])
						img[:,:,1] = np.uint8(img_tensor[z,:,:])
						img[:,:,2] = np.uint8(img_tensor[z,:,:])			
						cv2.imwrite(img_path+'/'+str(z+1)+'.jpg', img)
				else:
					img = np.zeros((img_tensor.shape[1],img_tensor.shape[2],3))
					for c in range(img_tensor.shape[0]):
						img[:,:,0] = np.uint8(img_tensor[c,:,:])
						img[:,:,1] = np.uint8(img_tensor[c,:,:])
						img[:,:,2] = np.uint8(img_tensor[c,:,:])			
						cv2.imwrite(img_path+'/'+str(c+1)+'.jpg', img)					
					
			if (img_tensor.ndim == 4):
				img = np.zeros((img_tensor.shape[2],img_tensor.shape[3],3))
				
				for c in range(img_tensor.shape[1]):
					for z in range(img_tensor.shape[0]):
						img[:,:,0] = np.uint8(img_tensor[z,c,:,:])
						img[:,:,1] = np.uint8(img_tensor[z,c,:,:])
						img[:,:,2] = np.uint8(img_tensor[z,c,:,:])
						cv2.imwrite(img_path+'/'+str(c+1)+'_'+str(z+1)+'.jpg', img)
		
		#img_output = np.zeros(it.windows_img[-1].tensor_img.shape)
		img_output = np.zeros(img_tensor.shape)
		imgs = os.listdir(img_path)
		
		
		
		# Path model 
		SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
		model = hub.load(SAVED_MODEL_PATH)

		start = time.time()
		display1f = False
		# Neural network processing
		if (img_output.ndim == 2):
			print('Processing: ', imgs[0])
			it.statusBar.configure(text = 'Processing: '+str(imgs[0]))
			fake_image, hr_image = model_NeuralNetwork(img_path+'/'+imgs[0])
			img_array = tensor_to_array(tf.squeeze(fake_image))
			img_output = np.asarray(img_array)[:,:,0]
			display1f = True
			
		if (img_output.ndim == 3):
			if (istiffRGB(img_output.shape)):
				print('Processing: ', imgs[0])
				it.statusBar.configure(text = 'Processing: '+str(imgs[0]))
				fake_image, hr_image = model_NeuralNetwork(img_path+'/'+imgs[0])
				img_array = tensor_to_array(tf.squeeze(fake_image))
				img_output = np.asarray(img_array)
				display1f = True
			elif (img_output.shape[0]>4):
				for image in imgs:
					z = int(image.split('.')[0])-1
					print('Processing: ', image, '\tz: ',z)	
					it.statusBar.configure(text = 'Processing: '+image+'\tz: '+str(z))
					fake_image, hr_image = model_NeuralNetwork(img_path+'/'+image)
					img_array = tensor_to_array(tf.squeeze(fake_image))
					img_output[z,:,:] = np.asarray(img_array)[:,:,0]				
			else:
				for image in imgs:
					c = int(image.split('.')[0])-1
					print('Processing: ', image, '\tc: ',c)	
					it.statusBar.configure(text = 'Processing: '+image+'\tc: '+str(c))		
					fake_image, hr_image = model_NeuralNetwork(img_path+'/'+image)
					img_array = tensor_to_array(tf.squeeze(fake_image))
					img_output[c,:,:] = np.asarray(img_array)[:,:,0]			
		
		if (img_output.ndim == 4):
			for image in imgs:
				c = int(image.split('.')[0].split('_')[0])-1
				z = int(image.split('.')[0].split('_')[1])-1
				print('Processing: ', image, '\tc: ',c,'z: ',z)
				it.statusBar.configure(text = 'Processing: '+image+'\tc: '+str(c)+'\tz: '+str(z))
				fake_image, hr_image = model_NeuralNetwork(img_path+'/'+image)
				img_array = tensor_to_array(tf.squeeze(fake_image))
				img_output[z,c,:,:] = np.asarray(img_array)[:,:,0]
			
		import src.interfaceTools as it
		#print("Time Taken: %f" % (time.time() - start))
		(m,s) = it.getFormatTime(time.time() - start)
		print("Runtime: ",m, "minutes, ",s, "seconds")		
		it.statusBar.configure(text = "Time Taken: " + str(time.time() - start))
		
		nnimg = it.NewWindow('Neural Network: '+it.windows_img[-1].nameWindow, image = True)
		if (display1f):
			nnimg.placeImage(img_output)
		else:	
			nnimg.desplay_image(img_output)
		nnimg.tensor_img = img_output	
		it.windows_img.append(nnimg)	
	else:
		from tkinter import messagebox
		messagebox.showinfo(message='There is no input parameter')
