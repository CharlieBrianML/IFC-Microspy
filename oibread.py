#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# oibread.py
# Contiene las herramientas para procesar un archivo .oib
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

from oiffile import *

def getMetadata(filename):
	with OifFile(filename) as oib:
		metadata = oib.mainfile
		metadata.update(dict(axes=oib.axes,shape=oib.shape,dtype=oib.dtype, num_aperture=1.35, pinhole_radius=(120000/1000)/2, magnification=0.75, refr_index=1.45))
	return metadata
	
#Obtencion de la matriz del oib	
def get_matrix_oib(filename):
	matrix = imread(filename)
	return matrix