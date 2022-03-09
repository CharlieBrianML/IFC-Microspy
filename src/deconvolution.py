#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# deconvolucion.py
# Gestiona el proceso de deconvolucion
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

from time import time
from time import sleep
import os
import sys
import tifffile
import numpy as np
from progress.bar import Bar, ChargingBar

import src.interfaceTools as it
import src.imageFunctions as imf
from .deconvTF import deconvolveTF
import src.tiff as tif

inx0, inx1, inx2, inx3 = (None, None, None, None)
inx0p, inx1p, inx2p, inx3p = (None, None, None, None)
metadata = {}

def deconvolutionTiff(img,psf,iterations):
	"""Performs deconvolution of a multichannel file"""
	global inx0, inx1, inx2, inx3, inx0p, inx1p, inx2p, inx3p, metadata
	inx0, inx1, inx2, inx3 = (None, None, None, None)
	inx0p, inx1p, inx2p, inx3p = (None, None, None, None)	
	
	deconv_list=np.zeros(img.shape,  dtype=metadata['type'])
	
	if(img.ndim==3):
		if('slices' in metadata):

			for z in range(metadata['slices']['value']):
				updateIndex(metadata['slices']['index'], z)
				deconv = deconvolveTF(img[inx0p:inx0,inx1p:inx1,inx2p:inx2].reshape((metadata['X'],metadata['Y'])),psf[z,:,:],iterations) #Image deconvolution function
				
				it.printMessage('Slice '+str(z+1)+' deconvolved')
				deconv_list[inx0p:inx0,inx1p:inx1,inx2p:inx2]=deconv
		if('frames' in metadata):
			for f in range(metadata['frames']['value']):
				updateIndex(metadata['frames']['index'], f)
				deconv = deconvolveTF(img[inx0p:inx0,inx1p:inx1,inx2p:inx2].reshape((metadata['X'],metadata['Y'])),psf,iterations) #Image deconvolution function
				
				it.printMessage('Frame '+str(f+1)+' deconvolved')
				deconv_list[inx0p:inx0,inx1p:inx1,inx2p:inx2]=deconv
		if('channels' in metadata):
			for c in range(metadata['channels']['value']):
				updateIndex(metadata['channels']['index'], c)
				deconv = deconvolveTF(img[inx0p:inx0,inx1p:inx1,inx2p:inx2],psf[inx0p:inx0,inx1p:inx1,inx2p:inx2],iterations) #Image deconvolution function
				
				it.printMessage('Channel '+str(c+1)+' deconvolved')
				deconv_list[inx0p:inx0,inx1p:inx1,inx2p:inx2]=deconv
	if(img.ndim==4):
		if(('channels' in metadata) and ('slices' in metadata)):
			for c in range(metadata['channels']['value']):
				bar = Bar("\nChannel "+str(c+1)+' :', max=metadata['channels']['value'])
				updateIndex(metadata['channels']['index'], c)
				for z in range(metadata['slices']['value']):
					updateIndex(metadata['slices']['index'], z)
					deconv = deconvolveTF(img[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3],psf[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3], iterations) #Image deconvolution function
					deconv_list[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3]=deconv
					bar.next()
				it.printMessage('Channel '+str(c+1)+' deconvolved')
				bar.finish()
		if(('channels' in metadata) and ('frames' in metadata)):
			for c in range(metadata['channels']['value']):
				bar = Bar("\nChannel "+str(c+1)+' :', max=metadata['channels']['value'])
				updateIndex(metadata['channels']['index'], c)
				for f in range(metadata['frames']['value']):
					updateIndex(metadata['frames']['index'], f)
					deconv = deconvolveTF(img[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3].reshape((metadata['X'],metadata['Y'])),psf[c,:,:], iterations) #Image deconvolution function
					deconv_list[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3]=deconv
					bar.next()
				it.printMessage('Channel '+str(c+1)+' deconvolved')
				bar.finish()
		if(('frames' in metadata) and ('slices' in metadata)):
			for f in range(metadata['frames']['value']):
				bar = Bar("\nFrame "+str(f+1)+' :', max=metadata['frames']['value'])
				updateIndex(metadata['frames']['index'], f)
				for z in range(metadata['slices']['value']):
					updateIndex(metadata['slices']['index'], z)
					deconv = deconvolveTF(img[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3].reshape((metadata['X'],metadata['Y'])),psf[z,:,:], iterations) #Image deconvolution function
					deconv_list[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3]=deconv.reshape(img[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3].shape)
					bar.next()
				it.printMessage('Frame '+str(f+1)+' deconvolved')
				bar.finish()				

	return deconv_list
	
def deconvolutionRGB(img,psf,iterations):
	"""Performs deconvolution of a RGB file"""
	deconv=np.zeros(img.shape)
	for crgb in range(3):
		deconv[:,:,crgb]=deconvolveTF(img[:,:,crgb], psf, iterations) #Image deconvolution function
	return deconv
	
def deconvolution1Frame(img,psf,iterations):
	"""Performs deconvolution of a matrix"""
	deconv=deconvolveTF(img, psf, iterations) #Image deconvolution function
	return deconv
	

def deconvolutionMain(img_tensor,psf_tensor,i, nameFile, metadataimg):
	"""This function is in charge of determining how the provided matrix should be processed together with the psf matrix"""
	global message, metadata
	metadata = metadataimg
	to=time()
	
	print(img_tensor.shape)
	print(psf_tensor.shape)
	it.printMessage('Starting deconvolution')
	if(img_tensor.ndim==2):
		tiffdeconv = deconvolution1Frame(img_tensor,psf_tensor,i)
	if(img_tensor.ndim==3):
		if(imf.istiffRGB(img_tensor.shape)):
			tiffdeconv = deconvolutionRGB(img_tensor,psf_tensor,i)
		else:
			tiffdeconv = deconvolutionTiff(img_tensor,psf_tensor,i)
	if(img_tensor.ndim==4):
		tiffdeconv = deconvolutionTiff(img_tensor,psf_tensor,i)

	if (metadata['type']=='uint16'):
		it.printMessage('converting matrix to uint16')
		deconvolution_matrix = np.uint16(tiffdeconv)
	if (metadata['type']=='uint8'):	
		it.printMessage('converting matrix to uint8')
		deconvolution_matrix = np.uint8(tiffdeconv)	

	it.printMessage('Deconvolution successful, end of execution')

	(m,s) = it.getFormatTime(time() - to)
	it.printMessage("Runtime: "+str(m)+" minutes, "+str(s)+" seconds")
	return deconvolution_matrix

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