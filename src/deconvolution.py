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

message = ''

def deconvolutionTiff(img,psf,iterations):
	"""Performs deconvolution of a multichannel file"""
	deconv_list=np.zeros(img.shape,  dtype="int16")
	
	if(img.ndim==3):
		for c in range(img.shape[0]):
			from skimage import io
			#psf2 = np.uint16(io.imread('/home/charlie/test/psf_0005.tif'))
			psf2 = io.imread('/home/charlie/test/psf_0005.tif')
			print(psf2.shape)
			#psf[c,:,:] = psf2[:,:,c]

			deconv = deconvolveTF(img[c,:,:],psf[c,:,:],iterations) #Image deconvolution function
			deconvN = imf.normalizar(deconv) #The matrix is normalized
			
			print('Channel ',c+1,' deconvolved')
			it.statusBar.configure(text = 'Channel '+str(c+1)+' deconvolved')
			deconv_list[c,:,:]=deconvN
	if(img.ndim==4):
		if(imf.istiffRGB(img.shape)):
			for c in range(img.shape[0]):
				deconv= deconvolutionRGB(img[c,:,:,:],psf[c,:,:], iterations, weight) #Image deconvolution function
				print('Channel ',c+1,' deconvolved')
				it.statusBar.configure(text = 'Channel '+str(c+1)+' deconvolved')
				deconv_list[c,:,:,:]=deconv
		else:
			for c in range(img.shape[1]):
				bar = Bar("\nChannel "+str(c+1)+' :', max=img.shape[0])
				for z in range(img.shape[0]):
					deconv = deconvolveTF(img[z,c,:,:],psf[z,c,:,:], iterations) #Image deconvolution function
					deconvN = imf.normalizar(deconv) #The matrix is normalized
					deconv_list[z,c,:,:]=deconvN
					bar.next()
				it.statusBar.configure(text = 'Channel '+str(c+1)+' deconvolved')	
				bar.finish()
	if(img.ndim==5):
		for c in range(img.shape[0]):
			for z in range(img.shape[1]):
				deconv= deconvolutionRGB(img[z,c,:,:,:],psf[z,c,:,:,:], iterations) #Image deconvolution function
				deconv_list[z,c,:,:,:]=deconv
			print('Channel ',c+1,' deconvolved')	
	return deconv_list
	
def deconvolutionRGB(img,psf,iterations):
	"""Performs deconvolution of a RGB file"""
	deconvN=np.zeros(img.shape)
	for crgb in range(3):
		deconv=deconvolveTF(img[:,:,crgb], psf[:,:,crgb], iterations) #Image deconvolution function
		deconvN[:,:,crgb]=imf.normalizar(deconv)
	return deconvN
	
def deconvolution1Frame(img,psf,iterations):
	"""Performs deconvolution of a matrix"""
	print('psf max:',psf.max())
	deconvN=np.zeros(img.shape,  dtype="int16")
	deconv=deconvolveTF(img, psf, iterations) #Image deconvolution function
	deconvN=imf.normalizar(np.uint8(deconv))
	return deconvN
	

def deconvolutionMain(img_tensor,psf_tensor,i, nameFile, metadata):
	"""This function is in charge of determining how the provided matrix should be processed together with the psf matrix"""
	global message
	to=time()

	path = os.path.dirname(os.path.realpath(sys.argv[0])) #Working directory
	savepath = os.path.join(path,'deconvolutions/Deconvolution_'+nameFile.split('.')[0]+' i-'+str(i)+'.tif')
	#tifffile.imsave(path + '/deconvolutions/'+nameFile.split('.')[0]+'_normalized.tif', np.uint16(img_tensor*(65535/img_tensor.max())), imagej=True)
	
	print(img_tensor.shape)
	print(psf_tensor.shape)
	message = '\nStarting deconvolution'
	print(message)
	it.statusBar.configure(text = 'Starting deconvolution')
	if(img_tensor.ndim==2):
		tiffdeconv = deconvolution1Frame(img_tensor,psf_tensor,i)
	if(img_tensor.ndim==3):
		if(imf.istiffRGB(img_tensor.shape)):
			tiffdeconv = deconvolutionRGB(img_tensor,psf_tensor,i)
		else:
			tiffdeconv = deconvolutionTiff(img_tensor,psf_tensor,i)
	if(img_tensor.ndim==4):
		tiffdeconv = deconvolutionTiff(img_tensor,psf_tensor,i)
		
	deconvolution_matrix = np.uint16(tiffdeconv)
		
	#tifffile.imsave(savepath, deconvolution_matrix, imagej=True)
	message = 'Deconvolution successful, end of execution'
	print(message)
	it.statusBar.configure(text = message)

	(m,s) = it.getFormatTime(time() - to)
	print("Runtime: ",m, "minutes, ",s, "seconds")
	#it.statusbar['text']="Runtime: "+str(tt/60)+"minutes"
	message = ''
	return deconvolution_matrix
