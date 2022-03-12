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
	print(metadata)
	"""Formats metadata extracted from a .tif file"""
	metadata_dic = {'Channel 1 Parameters':{'ExcitationWavelength':0.0,'EmissionWavelength':0.0},'Channel 2 Parameters':{'ExcitationWavelength':0.0,'EmissionWavelength':0.0}, 'Axis 5 Parameters Common':{'CalibrateValueA':0.0},
	'Channel 3 Parameters':{'ExcitationWavelength':0.0, 'EmissionWavelength':0.0}, 'Channel 4 Parameters':{'ExcitationWavelength':0.0, 'EmissionWavelength':0.0}, 'refr_index': 0.0,
	'num_aperture':0.0,'pinhole_radius':0.0,'magnification':0.0, 'Axis 3 Parameters Common':{'EndPosition':0.0,'StartPosition':0.0,'MaxSize':0.0}, 'Axis 0 Parameters Common':{'EndPosition':0.0, 'StartPosition':0.0, 'MaxSize':0.0}}

	metadata_dic['Channel 1 Parameters']['ExcitationWavelength'] = float(metadata['[Channel1Parameters]ExcitationWavelength'])
	metadata_dic['Channel 2 Parameters']['ExcitationWavelength'] = float(metadata['[Channel2Parameters]ExcitationWavelength'])
	metadata_dic['Channel 3 Parameters']['ExcitationWavelength'] = float(metadata['[Channel3Parameters]ExcitationWavelength'])
	metadata_dic['Channel 4 Parameters']['ExcitationWavelength'] = float(metadata['[Channel4Parameters]ExcitationWavelength'])
	metadata_dic['Channel 1 Parameters']['EmissionWavelength'] = float(metadata['[Channel1Parameters]EmissionWavelength'])
	metadata_dic['Channel 2 Parameters']['EmissionWavelength'] = float(metadata['[Channel1Parameters]EmissionWavelength'])
	metadata_dic['Channel 3 Parameters']['EmissionWavelength'] = float(metadata['[Channel1Parameters]EmissionWavelength'])
	metadata_dic['Channel 4 Parameters']['EmissionWavelength'] = float(metadata['[Channel1Parameters]EmissionWavelength'])
	metadata_dic['Axis 5 Parameters Common']['CalibrateValueA'] = float(metadata['[Axis5ParametersCommon]CalibrateValueA'])
	metadata_dic['num_aperture'] = float(metadata['ObjectiveLensNAValue'])
	print("NA:", float(metadata['ObjectiveLensNAValue']) )
	metadata_dic['pinhole_radius'] = (float(metadata['PinholeDiameter']))/2
	print("Pinhole:", (float(metadata['PinholeDiameter']))/2 )
	metadata_dic['magnification'] = float(metadata['Magnification'])
	print("Magnification:", float(metadata['Magnification']) )
	metadata_dic['refr_index'] = "{:.2f}".format( float(metadata['ObjectiveLensNAValue']) / np.sin(metadata_dic['Axis 5 Parameters Common']['CalibrateValueA']* np.pi / 180.) )
	print("Refr:indx: ",metadata_dic['refr_index'])
	metadata_dic['Axis 3 Parameters Common']['EndPosition'] = float(metadata['[Axis3ParametersCommon]EndPosition'])
	metadata_dic['Axis 3 Parameters Common']['StartPosition'] = float(metadata['[Axis3ParametersCommon]StartPosition'])
	metadata_dic['Axis 3 Parameters Common']['MaxSize'] = float(metadata['[Axis3ParametersCommon]MaxSize'])
	metadata_dic['Axis 0 Parameters Common']['MaxSize'] = float(metadata['[Axis0ParametersCommon]MaxSize'])
	metadata_dic['Axis 0 Parameters Common']['EndPosition'] = float(metadata['[Axis0ParametersCommon]EndPosition'])
	return metadata_dic

def getMetadata(filename):
	"""Get metadata from a .tif file"""
	metadata = {'path':filename, 'name':filename.split('/')[-1], 'tensor':io.imread(filename), 'num_aperture':1.35, 'pinhole_radius':(120000/1000)/2, 'magnification': 60, 'refr_index':1.47}
	try:	
		with tifffile.TiffFile(filename) as tif:
			imagej_hyperstack = tif.asarray()
			imagej_metadata = tif.imagej_metadata #Diccionario con todos los metadatos
		# print(imagej_metadata)
		#Se obtienen los metadatos de interes
		for tag in imagej_metadata:
			if tag in basicinfo:
				if (tag == 'channels'or tag == 'frames'or tag == 'slices'):
					metadata.update( {tag:{'value': imagej_metadata[tag], 'index': getIndexOfTuple(metadata['tensor'].shape,imagej_metadata[tag]) }} )
				else:
					metadata.update({tag: imagej_metadata[tag]})
		
		x,y = getSizeXY(metadata)
		metadata.update(x)
		metadata.update(y)
		metadata.update({'type': metadata['tensor'].dtype})
		if 'Info' in dict(metadata):
			info = metadata['Info'].split('\n')
			metadata.pop('Info')
			metadatainfo = {}
			for taginfo in info:
				for parameter in datainfo:
					if parameter in taginfo:
						infosplitted = taginfo.replace(" ", "").split('=')
						metadatainfo.update(  {infosplitted[0]:infosplitted[1]}  )
			metadata.update(metadata_format(metadatainfo))
		# print("Information found:",metadata.get())	
		return metadata
	except IndexError:
		print('No metadata found')
	
def getIndexOfTuple(shape, tag):
	return shape.index(int(tag))
	
def getSizeXY(metadata):
	shape = list(metadata['tensor'].shape)
	if ('channels' in metadata):
		shape.remove(metadata['channels']['value'])
	if ('frames' in metadata):
		shape.remove(metadata['frames']['value'])
	if ('slices' in metadata):
		shape.remove(metadata['slices']['value'])
	return {'X': shape[0]},{'Y': shape[1]}
