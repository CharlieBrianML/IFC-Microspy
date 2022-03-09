#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# oibread.py
# Contains the tools to process an .oib file
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

from oiffile import *

def getMetadata(filename):
	"""Gets the metadata of an .oib file"""
	with OifFile(filename) as oib:
		metadata = oib.mainfile
		metadata.update({'tensor':get_matrix_oib(filename)})
		metadata.update(dict(axes=oib.axes,shape=oib.shape,dtype=oib.dtype, num_aperture=1.35, pinhole_radius=(120000/1000)/2, magnification=0.75, refr_index=1.45))
		metadata.update(addBasicInfo(metadata))
	return metadata
	
def get_matrix_oib(filename):
	"""Gets the array associated with the .oib file"""
	return imread(filename)
	
def getIndexOfTuple(shape, tag):
	return shape.index(int(tag))
	
def addBasicInfo(metadata):
	"""Add additional basic information"""
	info = {}
	if ('View Max CH' in metadata['2D Display']):
		if(not(metadata['2D Display']['View Max CH']=='-1')):
			info.update({'channels':{'value': int(metadata['2D Display']['View Max CH']),'index':getIndexOfTuple(metadata['tensor'].shape, metadata['2D Display']['View Max CH'])}})
	if ('T End Pos' in metadata['2D Display']):
		if(not(metadata['2D Display']['T End Pos']=='-1')):
			info.update({'frames':{'value': int(metadata['2D Display']['T End Pos']),'index':getIndexOfTuple(metadata['tensor'].shape, metadata['2D Display']['T End Pos'])}})
	if ('Z End Pos' in metadata['2D Display']):
		if(not(metadata['2D Display']['Z End Pos']=='-1')):
			info.update({'slices':{'value': int(metadata['2D Display']['Z End Pos']),'index':getIndexOfTuple(metadata['tensor'].shape, metadata['2D Display']['Z End Pos'])}})
	ImageHeight , ImageWidth = 	int(metadata['Reference Image Parameter']['ImageHeight']), int(metadata['Reference Image Parameter']['ImageWidth'])
	info.update({'X':ImageHeight})
	info.update({'Y':ImageWidth})
	info.update({'type':metadata['dtype']})
	print("Information found:",info)
	return info