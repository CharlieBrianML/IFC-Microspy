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
		metadata.update(dict(axes=oib.axes,shape=oib.shape,dtype=oib.dtype, num_aperture=1.35, pinhole_radius=(120000/1000)/2, magnification=0.75, refr_index=1.45))
	return metadata
	
def get_matrix_oib(filename):
	"""Gets the array associated with the .oib file"""
	matrix = imread(filename)
	return matrix