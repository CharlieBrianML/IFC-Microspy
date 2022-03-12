#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# imageFunctions.py
# Contains all the tools for image processing
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

from ctypes import resize
from locale import normalize
from skimage.exposure import rescale_intensity
from skimage.restoration import denoise_tv_chambolle
import numpy as np
import cv2

inx0, inx1, inx2, inx3 = (None, None, None, None)
inx0p, inx1p, inx2p, inx3p = (None, None, None, None)

def normalizeImg(tensor, metadata):
	"""Normalizes a data matrix"""	
	if (metadata['type']=='uint8'):
		normalized = np.uint8(tensor*(255/tensor.max()))
	if (metadata['type']=='uint16'):
		normalized = np.uint16(tensor*(4096/tensor.max()))
	return normalized

def rescaleSkimage(img):
	"""Implement the function rescale_intensity from skimage"""
	imgRescale = rescale_intensity(img, in_range=(0, 255))
	imgRescale = (imgRescale * 255).astype("uint8")
	return imgRescale

def mostrarImagen(nameWindow, img, close):
	"""Display an image"""
	cv2.imshow(nameWindow, img)
	if (close):
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
def guardarImagen(nameFile, img):
	"""Save an image"""
	cv2.imwrite(nameFile, img)
	
def escalaGrises(img):
	"""Transforms an image to grayscale"""
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
def denoisingTV(img,value):
	"""Applies the total variation filter to an image"""
	return denoise_tv_chambolle(img, weight=value)
	
def tensorDenoisingTV(tensor,value, metadata):
	"""Applies the total variation filter to an image"""
	global inx0, inx1, inx2, inx3, inx0p, inx1p, inx2p, inx3p
	inx0, inx1, inx2, inx3 = (None, None, None, None)
	inx0p, inx1p, inx2p, inx3p = (None, None, None, None)
	
	if(tensor.ndim==2):
		img = denoisingTV(tensor,value)
	if(tensor.ndim==3):
		img = np.zeros(tensor.shape)
		if(istiffRGB(tensor.shape)):
			for r in range(tensor.shape[2]):
				img[:,:,r] = denoisingTV(tensor[:,:,r],value)
		else:
			if('slices' in metadata):
				for z in range(metadata['slices']['value']):
					updateIndex(metadata['slices']['index'], z)
					img[inx0p:inx0,inx1p:inx1,inx2p:inx2] = denoisingTV(tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2],value)
			if('frames' in metadata):
				for f in range(metadata['frames']['value']):
					updateIndex(metadata['frames']['index'], f)
					img[inx0p:inx0,inx1p:inx1,inx2p:inx2] = denoisingTV(tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2],value)					
			if('channels' in metadata):
				for c in range(metadata['channels']['value']):
					updateIndex(metadata['channels']['index'], c)
					img[inx0p:inx0,inx1p:inx1,inx2p:inx2] = denoisingTV(tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2],value)
	if(tensor.ndim==4):
		img = np.zeros(tensor.shape)
				
		if(('channels' in metadata) and ('slices' in metadata)):
			for c in range(metadata['channels']['value']):
				updateIndex(metadata['channels']['index'], c)
				for z in range(metadata['slices']['value']):
					updateIndex(metadata['slices']['index'], z)
					img[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3] = denoisingTV(tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3],value)
		if(('channels' in metadata) and ('frames' in metadata)):
			for c in range(metadata['channels']['value']):
				updateIndex(metadata['channels']['index'], c)
				for f in range(metadata['frames']['value']):
					updateIndex(metadata['frames']['index'], f)
					img[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3] = denoisingTV(tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3],value)
		if(('frames' in metadata) and ('slices' in metadata)):
			for f in range(metadata['frames']['value']):
				updateIndex(metadata['frames']['index'], f)
				for z in range(metadata['slices']['value']):
					updateIndex(metadata['slices']['index'], z)
					img[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3] = denoisingTV(tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3],value)
	
	img = img*(tensor.max()/img.max())
	if (metadata['type']=='uint16'):
		img = np.uint16(img)
	return img
	
def resizeTensor(tensor,x,y, metadata):
	"""change the dimensions (x, y) of a tensor"""
	global inx0, inx1, inx2, inx3, inx0p, inx1p, inx2p, inx3p
	inx0, inx1, inx2, inx3 = (None, None, None, None)
	inx0p, inx1p, inx2p, inx3p = (None, None, None, None)	

	if(tensor.ndim==2):
		img = cv2.resize(tensor, (x,y), interpolation = cv2.INTER_LINEAR)
	if(tensor.ndim==3):
		if(istiffRGB(tensor.shape)):
			img = np.zeros((x,y,tensor.shape[2]))
			for r in range(tensor.shape[2]):
				img[:,:,r] = cv2.resize(tensor[:,:,r], (x,y), interpolation = cv2.INTER_LINEAR)
			img = np.uint8(img)
		else:
			if('slices' in metadata):
				img = np.zeros(getNewShape(tensor, metadata, x,y))
				for z in range(metadata['slices']['value']):
					updateIndex(metadata['slices']['index'], z)
					img[inx0p:inx0,inx1p:inx1,inx2p:inx2] = cv2.resize(tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2].reshape((metadata['X'],metadata['Y'])), (x,y), interpolation = cv2.INTER_LINEAR)
			if('frames' in metadata):
				img = np.zeros(getNewShape(tensor, metadata, x,y))
				for f in range(metadata['frames']['value']):
					updateIndex(metadata['frames']['index'], f)
					img[inx0p:inx0,inx1p:inx1,inx2p:inx2] = cv2.resize(tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2].reshape((metadata['X'],metadata['Y'])), (x,y), interpolation = cv2.INTER_LINEAR)					
			if('channels' in metadata):
				img = np.zeros(getNewShape(tensor, metadata, x,y))
				for c in range(metadata['channels']['value']):
					updateIndex(metadata['channels']['index'], c)
					img[inx0p:inx0,inx1p:inx1,inx2p:inx2] = cv2.resize(tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2].reshape((metadata['X'],metadata['Y'])), (x,y), interpolation = cv2.INTER_LINEAR)
	if(tensor.ndim==4):
		if(('channels' in metadata) and ('slices' in metadata)):
			img = np.zeros(getNewShape(tensor, metadata, x,y))
			for c in range(metadata['channels']['value']):
				updateIndex(metadata['channels']['index'], c)
				for z in range(metadata['slices']['value']):
					updateIndex(metadata['slices']['index'], z)
					img[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3] = cv2.resize(tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3].reshape((metadata['X'],metadata['Y'])), (x,y), interpolation = cv2.INTER_LINEAR)
		if(('channels' in metadata) and ('frames' in metadata)):
			img = np.zeros(getNewShape(tensor, metadata, x,y))
			for c in range(metadata['channels']['value']):
				updateIndex(metadata['channels']['index'], c)
				for f in range(metadata['frames']['value']):
					updateIndex(metadata['frames']['index'], f)
					img[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3] = cv2.resize(tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3].reshape((metadata['X'],metadata['Y'])), (x,y), interpolation = cv2.INTER_LINEAR)
		if(('frames' in metadata) and ('slices' in metadata)):
			img = np.zeros(getNewShape(tensor, metadata, x,y))
			for f in range(metadata['frames']['value']):
				updateIndex(metadata['frames']['index'], f)
				for z in range(metadata['slices']['value']):
					updateIndex(metadata['slices']['index'], z)
					resized = cv2.resize(tensor[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3].reshape((metadata['X'],metadata['Y'])), (x,y), interpolation = cv2.INTER_LINEAR)				
					img[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3] = resized.reshape(img[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3].shape)
	
	if (metadata['type']=='uint16'):
		img = np.uint16(img)
	
	return img

def getNewShape(tensor, metadata, x,y):
	shape = list(tensor.shape)
	indxs = []
	for i in range(len(shape)):
		if(metadata['X']==shape[i] or metadata['Y']==shape[i]):
			indxs.append(i)
	shape[indxs[0]], shape[indxs[1]] = (x,y)
	print('New shape: ', shape)
	return tuple(shape)
	
def imgReadCv2(nameImg):
	return cv2.imread(nameImg)
		
def istiffRGB(tiff):
	if(tiff[len(tiff)-1]==3):
		return True
	else:
		return False
	  
def getMetadataImg(filepath):
	"""Get metadata from a .tif file"""
	matrix = cv2.imread(filepath)
	metadata = {'path':filepath, 'name':filepath.split('/')[-1], 'tensor':matrix, 'type':matrix.dtype,'X':matrix.shape[0],'Y':matrix.shape[1], 'num_aperture':1.35, 'pinhole_radius':(120000/1000)/2, 'magnification': 60.0, 'refr_index':1.47}
	return metadata
	
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