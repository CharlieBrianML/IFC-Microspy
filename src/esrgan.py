#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# esrgan.py
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
inx0, inx1, inx2, inx3 = (None, None, None, None)
inx0p, inx1p, inx2p, inx3p = (None, None, None, None)
metadata = {}

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

def nn(img_tensor, index, metadataimg):
	import tifffile
	import src.interfaceTools as it
	from shutil import rmtree
	from .imageFunctions import istiffRGB
	from tkinter import messagebox
	import os
	global model
	
	global inx0, inx1, inx2, inx3, inx0p, inx1p, inx2p, inx3p, metadata
	inx0, inx1, inx2, inx3 = (None, None, None, None)
	inx0p, inx1p, inx2p, inx3p = (None, None, None, None)
	
	metadata = metadataimg
	
	if(len(it.windows_img)>0):
	
		if not(os.path.isdir('src/cache/training_set')):
			os.mkdir('src/cache/training_set')
			
		# img_path = 'src/cache/training_set/training_'+it.windows_img[index].nameWindow.replace(" ", "").replace("-", "")
		img_path = 'src/cache/training_set/'
		
		print(img_tensor.shape)
		
		if (it.windows_img[index].metadata['type']=='uint16'):
			img_tensor = np.uint16(img_tensor)*(255/4095)
		
		if (img_tensor.ndim == 2):
			img = np.zeros((img_tensor.shape[0],img_tensor.shape[1],3))
			img[:,:,0] = np.uint8(img_tensor)
			img[:,:,1] = np.uint8(img_tensor)
			img[:,:,2] = np.uint8(img_tensor)
			cv2.imwrite(img_path+'/1.jpg', img)
		if (img_tensor.ndim == 3):
			if (istiffRGB(img_tensor.shape)):
				cv2.imwrite(img_path+'/1.jpg', img_tensor)
			elif ('slices' in metadata):
				img = np.zeros((metadata['X'],metadata['Y'],3))
				for z in range(metadata['slices']['value']):
					updateIndex(metadata['slices']['index'], z)
					img[:,:,0] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2])
					img[:,:,1] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2])
					img[:,:,2] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2])
					cv2.imwrite(img_path+'/'+str(z+1)+'.jpg', img)
			elif ('frames' in metadata):
				img = np.zeros((metadata['X'],metadata['Y'],3))
				for f in range(metadata['frames']['value']):
					updateIndex(metadata['frames']['index'], f)
					img[:,:,0] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2])
					img[:,:,1] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2])
					img[:,:,2] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2])
					cv2.imwrite(str(f+1)+'.jpg', img)		
					cv2.imwrite(img_path+'/'+str(f+1)+'.jpg', img)	
			else:
				img = np.zeros((metadata['X'],metadata['Y'],3))
				for c in range(metadata['channels']['value']):
					updateIndex(metadata['channels']['index'], c)
					img[:,:,0] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2])
					img[:,:,1] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2])
					img[:,:,2] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2])
					cv2.imwrite(img_path+'/'+str(c+1)+'.jpg', img)			
				
		if (img_tensor.ndim == 4):
			img = np.zeros((metadata['X'],metadata['Y'],3))
			if(('channels' in metadata) and ('slices' in metadata)):
				for c in range(metadata['channels']['value']):
					updateIndex(metadata['channels']['index'], c)
					for z in range(metadata['slices']['value']):
						updateIndex(metadata['slices']['index'], z)
						img[:,:,0] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3])
						img[:,:,1] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3])
						img[:,:,2] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3])
						cv2.imwrite(img_path+'/'+str(c+1)+'_'+str(z+1)+'.jpg', img)
			if(('channels' in metadata) and ('frames' in metadata)):
				for c in range(metadata['channels']['value']):
					updateIndex(metadata['channels']['index'], c)
					for f in range(metadata['frames']['value']):
						updateIndex(metadata['frames']['index'], f)
						img[:,:,0] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3])
						img[:,:,1] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3])
						img[:,:,2] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3])
						cv2.imwrite(img_path+'/'+str(c+1)+'_'+str(f+1)+'.jpg', img)
			if(('frames' in metadata) and ('slices' in metadata)):
				for f in range(metadata['frames']['value']):
					updateIndex(metadata['frames']['index'], f)
					for z in range(metadata['slices']['value']):
						updateIndex(metadata['slices']['index'], z)
						img[:,:,0] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3].reshape((metadata['X'],metadata['Y'])) )
						img[:,:,1] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3].reshape((metadata['X'],metadata['Y'])) )
						img[:,:,2] = np.uint8(img_tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3].reshape((metadata['X'],metadata['Y'])) )
						cv2.imwrite(img_path+'/'+str(f+1)+'_'+str(z+1)+'.jpg', img)					
		
		img_output = np.zeros(img_tensor.shape)
		imgs = os.listdir(img_path)
		
		# Path model 
		SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
		model = hub.load(SAVED_MODEL_PATH)

		start = time.time()
		display1f = False
		try:	
			# Neural network processing
			if (img_output.ndim == 2):
				it.printMessage('Processing: '+str(imgs[0]))
				fake_image, hr_image = model_NeuralNetwork(img_path+'/'+imgs[0])
				img_array = tensor_to_array(tf.squeeze(fake_image))
				img_output = np.asarray(img_array)[:,:,0]
				display1f = True
				
			if (img_output.ndim == 3):
				if (istiffRGB(img_output.shape)):
					it.printMessage('Processing: '+str(imgs[0]))
					fake_image, hr_image = model_NeuralNetwork(img_path+'/'+imgs[0])
					img_array = tensor_to_array(tf.squeeze(fake_image))
					img_output = np.asarray(img_array)
					display1f = True
				elif ('slices' in metadata):
					for image in imgs:
						z = int(image.split('.')[0])-1
						it.printMessage('Processing: '+image+'\tz: '+str(z))
						fake_image, hr_image = model_NeuralNetwork(img_path+'/'+image)
						img_array = tensor_to_array(tf.squeeze(fake_image))
						updateIndex(metadata['slices']['index'], z)
						img_output[inx0p:inx0,inx1p:inx1,inx2p:inx2] = np.asarray(img_array)[:,:,0]
				elif ('frames' in metadata):
					for image in imgs:
						f = int(image.split('.')[0])-1
						it.printMessage('Processing: '+image+'\tf: '+str(f))
						fake_image, hr_image = model_NeuralNetwork(img_path+'/'+image)
						img_array = tensor_to_array(tf.squeeze(fake_image))
						updateIndex(metadata['frames']['index'], f)
						img_output[inx0p:inx0,inx1p:inx1,inx2p:inx2] = np.asarray(img_array)[:,:,0]						
				else:
					for image in imgs:
						c = int(image.split('.')[0])-1
						it.printMessage('Processing: '+image+'\tc: '+str(c))		
						fake_image, hr_image = model_NeuralNetwork(img_path+'/'+image)
						img_array = tensor_to_array(tf.squeeze(fake_image))
						updateIndex(metadata['channels']['index'], c)
						img_output[inx0p:inx0,inx1p:inx1,inx2p:inx2] = np.asarray(img_array)[:,:,0]			
			
			if (img_output.ndim == 4):
				if(('channels' in metadata) and ('slices' in metadata)):
					for image in imgs:
						c = int(image.split('.')[0].split('_')[0])-1
						z = int(image.split('.')[0].split('_')[1])-1
						it.printMessage('Processing: '+image+'\tc: '+str(c)+'\tz: '+str(z))
						fake_image, hr_image = model_NeuralNetwork(img_path+'/'+image)
						img_array = tensor_to_array(tf.squeeze(fake_image))
						updateIndex(metadata['channels']['index'], c)
						updateIndex(metadata['slices']['index'], z)
						img_output[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3] = np.asarray(img_array)[:,:,0]
				if(('channels' in metadata) and ('frames' in metadata)):
					for image in imgs:
						c = int(image.split('.')[0].split('_')[0])-1
						f = int(image.split('.')[0].split('_')[1])-1
						it.printMessage('Processing: '+image+'\tc: '+str(c)+'\tf: '+str(f))
						fake_image, hr_image = model_NeuralNetwork(img_path+'/'+image)
						img_array = tensor_to_array(tf.squeeze(fake_image))
						updateIndex(metadata['channels']['index'], c)
						updateIndex(metadata['frames']['index'], f)
						img_output[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3] = np.asarray(img_array)[:,:,0]
				if(('frames' in metadata) and ('slices' in metadata)):
					for image in imgs:
						f = int(image.split('.')[0].split('_')[0])-1
						z = int(image.split('.')[0].split('_')[1])-1
						it.printMessage('Processing: '+image+'\tf: '+str(f)+'\tz: '+str(z))
						fake_image, hr_image = model_NeuralNetwork(img_path+'/'+image)
						img_array = tensor_to_array(tf.squeeze(fake_image))
						updateIndex(metadata['frames']['index'], f)
						updateIndex(metadata['slices']['index'], z)
						img_output[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3] = np.asarray(img_array)[:,:,0].reshape(img_output[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3].shape)
			
			import src.interfaceTools as it
			
			(m,s) = it.getFormatTime(time.time() - start)
			it.printMessage("Runtime: "+str(m)+" minutes, "+str(s)+" seconds")
			
			if (it.windows_img[index].metadata['type']=='uint16'):
				img_output = np.uint16(img_output*(4095/255))
			
			nnimg = it.NewWindow('Neural Network: '+it.windows_img[index].nameWindow, metadata=metadataimg, image = True)
			if (display1f):
				nnimg.placeImage(img_output)
			else:	
				nnimg.desplay_image(img_output)
			nnimg.tensor_img = img_output	
			it.windows_img.append(nnimg)
			
		except ValueError:
			messagebox.showinfo(message='The matrix (x, y) is not the same size or its size is odd')
			if (os.path.isdir('src/cache/training_set')):
				rmtree("src/cache/training_set")
		#except InternalError:
		# except Exception:	
			# messagebox.showinfo(message='CUDA runtime implicit initialization on GPU failed. Status: out of memory')
		finally:	
			if (os.path.isdir('src/cache/training_set')):
				rmtree("src/cache/training_set")
	else:
		messagebox.showinfo(message='There is no input parameter')

def updateIndex(index, pos):
	global inx0, inx1, inx2, inx3, inx0p, inx1p, inx2p, inx3p

	if (index==0):
		inx0 = pos+1
		inx0p = pos
	if (index==1):
		inx1 = pos+1
		inx1p = pos
	if (index==2):
		inx2 = pos+1
		inx2p = pos
	if (index==3):
		inx3 = pos+1
		inx3p = pos			