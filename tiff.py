#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# tiff.py
# Contains the functions for reading and processing .tif files
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

import cv2
import tifffile
import numpy as np
from skimage import io

dataDeconv = ('images', 'channels', 'slices', 'hyperstack','Info')
datainfo = ('DimensionOrder', 'IsRGB', 'PixelType', 'SizeC','SizeT', 'SizeX', 'SizeY', 'SizeZ', 'ObjectiveLens NAValue', 'PinholeDiameter', 
'[Channel 1 Parameters] ExcitationWavelength', '[Channel 2 Parameters] ExcitationWavelength', '[Channel 3 Parameters] ExcitationWavelength',
'[Channel 4 Parameters] ExcitationWavelength', '[Reference Image Parameter] HeightConvertValue', '[Reference Image Parameter] WidthConvertValue', 
'[Channel 1 Parameters] EmissionWavelength', '[Channel 2 Parameters] EmissionWavelength', '[Channel 3 Parameters] EmissionWavelength','[Channel 4 Parameters] EmissionWavelength',
'[Reference Image Parameter] HeightUnit', '[Axis 3 Parameters Common] PixUnit', '[Axis 3 Parameters Common] EndPosition', '[Axis 3 Parameters Common] StartPosition')

def readTiff(fileTiff):
	"""Function that reads a .tif file"""
	img = io.imread(fileTiff)
	return img
	
def imgtoTiff(imgs,savepath):
	"""Function that converts a multidimensional array to a .tif file"""
	tifffile.imsave(savepath,imgs) #[[[]]] -> .tif

def imgtoMatrix(img_list):
	"""Function that converts images to multidimensional arrays"""
	cnt_num = 0
	for img in img_list: #Estraemos cada imagen de la lista
		new_img = img[np.newaxis, ::] #Convertimos la imagen a una matriz multidimensional
		if cnt_num == 0:
			tiff_list = new_img
		else:
			tiff_list = np.append(tiff_list, new_img, axis=0) #Agregamos a la lista las imagenes convertidas en matrices
		cnt_num += 1
	return tiff_list
	
def metadata_format(metadata):
	"""Formats metadata extracted from a .tif file"""
	metadata_dic = {'Channel 1 Parameters':{'ExcitationWavelength':0.0,'EmissionWavelength':0.0},'Channel 2 Parameters':{'ExcitationWavelength':0.0,'EmissionWavelength':0.0},
	'Channel 3 Parameters':{'ExcitationWavelength':0.0, 'EmissionWavelength':0.0}, 'Channel 4 Parameters':{'ExcitationWavelength':0.0, 'EmissionWavelength':0.0}, 'refr_index': 0.0,
	'num_aperture':0.0,'pinhole_radius':0.0,'magnification':0.0, 'Axis 3 Parameters Common':{'EndPosition':0.0,'StartPosition':0.0}, 'Axis 0 Parameters Common':{'EndPosition':0.0, 'StartPosition':0.0}}

	metadata_dic['Channel 1 Parameters']['ExcitationWavelength'] = float(metadata['[Channel1Parameters]ExcitationWavelength'])
	metadata_dic['Channel 2 Parameters']['ExcitationWavelength'] = float(metadata['[Channel2Parameters]ExcitationWavelength'])
	metadata_dic['Channel 3 Parameters']['ExcitationWavelength'] = float(metadata['[Channel3Parameters]ExcitationWavelength'])
	metadata_dic['Channel 4 Parameters']['ExcitationWavelength'] = float(metadata['[Channel4Parameters]ExcitationWavelength'])
	metadata_dic['Channel 1 Parameters']['EmissionWavelength'] = float(metadata['[Channel1Parameters]EmissionWavelength'])
	metadata_dic['Channel 2 Parameters']['EmissionWavelength'] = float(metadata['[Channel1Parameters]EmissionWavelength'])
	metadata_dic['Channel 3 Parameters']['EmissionWavelength'] = float(metadata['[Channel1Parameters]EmissionWavelength'])
	metadata_dic['Channel 4 Parameters']['EmissionWavelength'] = float(metadata['[Channel1Parameters]EmissionWavelength'])
	metadata_dic['num_aperture'] = float(metadata['ObjectiveLensNAValue'])
	metadata_dic['pinhole_radius'] = (float(metadata['PinholeDiameter']))/2
	metadata_dic['magnification'] = 0.75
	metadata_dic['refr_index'] = 1.5
	metadata_dic['Axis 3 Parameters Common']['EndPosition'] = float(metadata['[Axis3ParametersCommon]EndPosition'])
	metadata_dic['Axis 3 Parameters Common']['StartPosition'] = float(metadata['[Axis3ParametersCommon]StartPosition'])
	metadata_dic['Axis 0 Parameters Common']['EndPosition'] = float(metadata['[ReferenceImageParameter]HeightConvertValue'])
	return metadata_dic

def getMetadata(filename):
	"""Get metadata from a .tif file"""
	metadata = []
	try:
		with tifffile.TiffFile(filename) as tif:
			imagej_hyperstack = tif.asarray()
			imagej_metadata = tif.imagej_metadata #Diccionario con todos los metadatos
		#Se obtienen los metadatos de interes
		for tag in imagej_metadata:
			if tag in dataDeconv:
				metadata.append((tag, imagej_metadata[tag]))
		#Se obtienen los metadatos restantes del atributo info
		metadatainfo = metadata[4][1].split('\n')
		metadata.pop()
		for taginfo in metadatainfo:
			for parameter in datainfo:
				if parameter in taginfo:
					metadata.append(tuple(taginfo.replace(" ", "").split('=')))
	except IndexError:
		print('No metadata found')
	return metadata_format(dict(metadata))