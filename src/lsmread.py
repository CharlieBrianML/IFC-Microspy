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

basicinfo = ('images', 'channels', 'frames','slices', 'hyperstack','Info')
datainfo = ('DimensionOrder', 'IsRGB', 'PixelType', 'SizeC','SizeT', 'SizeX', 'SizeY', 'SizeZ', 'ObjectiveLens NAValue', 'PinholeDiameter', 'Magnification','[Axis 5 Parameters Common] CalibrateValueA',
'[Channel 1 Parameters] ExcitationWavelength', '[Channel 2 Parameters] ExcitationWavelength', '[Channel 3 Parameters] ExcitationWavelength',
'[Channel 4 Parameters] ExcitationWavelength', '[Reference Image Parameter] HeightConvertValue', '[Reference Image Parameter] WidthConvertValue', 
'[Channel 1 Parameters] EmissionWavelength', '[Channel 2 Parameters] EmissionWavelength', '[Channel 3 Parameters] EmissionWavelength','[Channel 4 Parameters] EmissionWavelength',
'[Reference Image Parameter] HeightUnit', '[Axis 3 Parameters Common] PixUnit', '[Axis 3 Parameters Common] EndPosition', '[Axis 3 Parameters Common] StartPosition',
'[Axis 3 Parameters Common] MaxSize', '[Axis 0 Parameters Common] MaxSize', '[Axis 0 Parameters Common] EndPosition')

def readTiff(fileTiff):
	"""Function that reads a .tif file"""
	return io.imread(fileTiff)
	
def imgtoTiff(imgs,savepath):
	"""Function that converts a multidimensional array to a .tif file"""
	tifffile.imsave(savepath,imgs) #[[[]]] -> .tif

def getMetadata(filename):
	"""Get metadata from a .lsm file"""
	metadata = {'path':filename, 'name':filename.split('/')[-1], 'num_aperture':1.35, 'pinhole_radius':(120000/1000)/2, 'magnification': 0.75, 'refr_index':1.45}
	try:
		with tifffile.TiffFile(filename) as tif:
			imagej_metadata = {}
			imagej_hyperstack = tif.asarray()
			for tag in tif.pages[0].tags.values():
				name, value = tag.name, tag.value
				imagej_metadata[name] = value

		metadata.update({'tensor':resizeTensor(imagej_hyperstack)})
		metadata.update({'type':imagej_hyperstack.dtype})
		if('CZ_LSMINFO' in imagej_metadata):
			if ('DimensionChannels' in imagej_metadata['CZ_LSMINFO']):
				if(imagej_metadata['CZ_LSMINFO']['DimensionChannels']>1):
					metadata.update({'channels': {'value':imagej_metadata['CZ_LSMINFO']['DimensionChannels'],'index':getIndexOfTuple(metadata['tensor'].shape,imagej_metadata['CZ_LSMINFO']['DimensionChannels'])}} )
			if ('DimensionZ' in imagej_metadata['CZ_LSMINFO']):
				if(imagej_metadata['CZ_LSMINFO']['DimensionZ']>1):
					metadata.update({'slices': {'value':imagej_metadata['CZ_LSMINFO']['DimensionZ'],'index':getIndexOfTuple(metadata['tensor'].shape,imagej_metadata['CZ_LSMINFO']['DimensionZ'])}} )
			if ('DimensionTime' in imagej_metadata['CZ_LSMINFO']):
				if(imagej_metadata['CZ_LSMINFO']['DimensionTime']>1):
					metadata.update({'frames': {'value':imagej_metadata['CZ_LSMINFO']['DimensionTime'],'index':getIndexOfTuple(metadata['tensor'].shape,imagej_metadata['CZ_LSMINFO']['DimensionTime'])}} )
			if ('DimensionX' in imagej_metadata['CZ_LSMINFO']):
				metadata.update({'X':imagej_metadata['CZ_LSMINFO']['DimensionX']})
			if ('DimensionY' in imagej_metadata['CZ_LSMINFO']):
				metadata.update({'Y':imagej_metadata['CZ_LSMINFO']['DimensionY']})
		print("Information found:",metadata)
		
		return metadata
	except IndexError:
		print('No metadata found')
	
def getIndexOfTuple(shape, tag):
	return shape.index(int(tag))
	
def resizeTensor(tensor):
	shape = list(tensor.shape)
	for ele in shape:
		if 1 in shape:
			shape.remove(1)
	return tensor.reshape(shape)
