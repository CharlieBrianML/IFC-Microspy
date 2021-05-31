#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# createPSF.py
# Genera una matriz o tensor que contiene una psf dados metadatos de entrada
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

"""Point Spread Function example.

Demonstrate the use of the psf library for calculating point spread functions for fluorescence microscopy.

"""
import numpy as np
import psf


def psf_generator(cmap='hot', savebin=False, savetif=False, savevol=False, plot=False, display=False, psfvol=False, psftype=0, expsf=False, empsf=False, realshape=(0,0), **kwargs):
	"""Calculate and save point spread functions."""

	args = {
		'shape': (50, 50),  # number of samples in z and r direction
		'dims': (5.0, 5.0),   # size in z and r direction in micrometers
		'ex_wavelen': 488.0,  # excitation wavelength in nanometers
		'em_wavelen': 520.0,  # emission wavelength in nanometers
		'num_aperture': 1.2,
		'refr_index': 1.333,
		'magnification': 1.0,
		'pinhole_radius': 0.05,  # in micrometers
		'pinhole_shape': 'round',
	}
	args.update(kwargs)

	if (psftype == 0):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.CONFOCAL, **args)	
		print('psf.GAUSSIAN | psf.CONFOCAL generated')	
	if (psftype == 1):
		psf_matrix = psf.PSF(psf.ISOTROPIC | psf.EXCITATION, **args)
		print('psf.ISOTROPIC | psf.EXCITATION generated')
	if (psftype == 2):
		psf_matrix = psf.PSF(psf.ISOTROPIC | psf.EMISSION, **args)
		print('psf.ISOTROPIC | psf.EMISSION generated')
	if (psftype == 3):
		psf_matrix = psf.PSF(psf.ISOTROPIC | psf.WIDEFIELD, **args)
		print('psf.ISOTROPIC | psf.WIDEFIELD generated')
	if (psftype == 4):
		psf_matrix = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
		print('psf.ISOTROPIC | psf.CONFOCAL generated')
	if (psftype == 5):
		psf_matrix = psf.PSF(psf.ISOTROPIC | psf.TWOPHOTON, **args)
		print('psf.ISOTROPIC | psf.TWOPHOTON generated')
	if (psftype == 6):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.EXCITATION, **args)
		print('psf.GAUSSIAN | psf.EXCITATION generated')
	if (psftype == 7):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.EMISSION, **args)
		print('psf.GAUSSIAN | psf.EMISSION generated')
	if (psftype == 8):
		print('psf.GAUSSIAN | psf.WIDEFIELD generated')
	if (psftype == 9):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.CONFOCAL, **args)
		print('psf.GAUSSIAN | psf.CONFOCAL generated')
	if (psftype == 10):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.TWOPHOTON, **args)
		print('psf.GAUSSIAN | psf.TWOPHOTON generated')
	if (psftype == 11):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.EXCITATION | psf.PARAXIAL, **args)
		print('psf.GAUSSIAN | psf.EXCITATION | psf.PARAXIAL generated')
	if (psftype == 12):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.EMISSION | psf.PARAXIAL, **args)
		print('psf.GAUSSIAN | psf.EMISSION | psf.PARAXIAL generated')
	if (psftype == 13):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.WIDEFIELD | psf.PARAXIAL, **args)
		print('psf.GAUSSIAN | psf.WIDEFIELD | psf.PARAXIAL generated')
	if (psftype == 14):
		print('psf.GAUSSIAN | psf.CONFOCAL | psf.PARAXIAL generated')
	if (psftype == 15):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.TWOPHOTON | psf.PARAXIAL, **args)
		print('psf.GAUSSIAN | psf.TWOPHOTON | psf.PARAXIAL generated')
	
	if empsf:
		psf_matrix = psf_matrix.expsf
	if expsf:
		psf_matrix = psf_matrix.empsf
		
	
	if psfvol:
		psf_matrix = normalize_matrix(psf_matrix.volume())
	else: 
		psf_matrix = normalize_matrix(psf.mirror_symmetry(psf_matrix.data))
	
	if plot:
		import matplotlib.pyplot as plt
		plt.imshow(psf_matrix, cmap=cmap)
		plt.show()
	
	if display:
		import cv2
		cv2.imshow('PSF',psf_matrix)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if savetif:
		# save zr slices to TIFF files
		from tifffile import imsave
		
		imsave('psf_matrix.tif', psf_matrix, metadata = {'axes':'TZCYX'}, imagej=True)

	if savevol:
		# save xyz volumes to files.
		from tifffile import imsave
		
		imsave('psf_matrix_vol.tif', psf_matrix,  metadata = {'axes':'TZCYX'}, imagej=True)
		
	psf_matrix = psf_matrix[:realshape[0],:,:]
	psf_matrix = psf_matrix[:,:realshape[1],:realshape[1]]
		
	print('PSF shape: ', psf_matrix.shape)
	return psf_matrix
	
def normalize_matrix(matrix):
	return np.uint8(255*matrix)