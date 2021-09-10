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

from skimage.exposure import rescale_intensity
from skimage.restoration import denoise_tv_chambolle
import numpy as np
import cv2


def normalizar(data):
	"""Normalizes a data matrix"""
	max=np.amax(data)#Se calcula el valor maximo del vector
	for p in range(data.shape[0]):
		for m in range(data.shape[1]):
			data[p][m]=(data[p][m]*256)/max  #Formula para normalizar los valores de [0, 255]
	return data

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
	
def imgReadCv2(nameImg):
	return cv2.imread(nameImg)

def validatePSF(tiff,psf):
	if(tiff.shape==psf.shape):
		return True
		
def istiffRGB(tiff):
	if(tiff[len(tiff)-1]==3):
		return True
	else:
		return False
	
#Funcion para elegir el canal de la matriz       
def elegirCanal(canal,matrix):
	img = np.zeros((matrix.shape[0], matrix.shape[1],3))
	if(canal=='R' or canal=='r'):
		color=2
	if(canal=='G' or canal=='g'):
		color=1
	if(canal=='B' or canal=='b'):
		color=0
	img[:,:,color]=matrix
	return img
	