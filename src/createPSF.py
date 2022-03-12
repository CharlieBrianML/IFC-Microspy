#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# createPSF.py
# This module use of the psf library for calculating point spread functions for fluorescence microscopy
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

import numpy as np
import psf

inx0, inx1, inx2, inx3 = (None, None, None, None)
inx0p, inx1p, inx2p, inx3p = (None, None, None, None)


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
		psf_matrix = psf.PSF(psf.ISOTROPIC | psf.EXCITATION, **args)
		print('psf.ISOTROPIC | psf.EXCITATION generated')
	if (psftype == 1):
		psf_matrix = psf.PSF(psf.ISOTROPIC | psf.EMISSION, **args)
		print('psf.ISOTROPIC | psf.EMISSION generated')
	if (psftype == 2):
		psf_matrix = psf.PSF(psf.ISOTROPIC | psf.WIDEFIELD, **args)
		print('psf.ISOTROPIC | psf.WIDEFIELD generated')
	if (psftype == 3):
		psf_matrix = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
		print('psf.ISOTROPIC | psf.CONFOCAL generated')
	if (psftype == 4):
		psf_matrix = psf.PSF(psf.ISOTROPIC | psf.TWOPHOTON, **args)
		print('psf.ISOTROPIC | psf.TWOPHOTON generated')
	if (psftype == 5):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.EXCITATION, **args)
		print('psf.GAUSSIAN | psf.EXCITATION generated')
	if (psftype == 6):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.EMISSION, **args)
		print('psf.GAUSSIAN | psf.EMISSION generated')
	if (psftype == 7):
		print('psf.GAUSSIAN | psf.WIDEFIELD generated')
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.WIDEFIELD, **args)
	if (psftype == 8):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.CONFOCAL, **args)
		print('psf.GAUSSIAN | psf.CONFOCAL generated')
	if (psftype == 9):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.TWOPHOTON, **args)
		print('psf.GAUSSIAN | psf.TWOPHOTON generated')
	if (psftype == 10):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.EXCITATION | psf.PARAXIAL, **args)
		print('psf.GAUSSIAN | psf.EXCITATION | psf.PARAXIAL generated')
	if (psftype == 11):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.EMISSION | psf.PARAXIAL, **args)
		print('psf.GAUSSIAN | psf.EMISSION | psf.PARAXIAL generated')
	if (psftype == 12):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.WIDEFIELD | psf.PARAXIAL, **args)
		print('psf.GAUSSIAN | psf.WIDEFIELD | psf.PARAXIAL generated')
	if (psftype == 13):
		print('psf.GAUSSIAN | psf.CONFOCAL | psf.PARAXIAL generated')
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.CONFOCAL | psf.PARAXIAL, **args)
	if (psftype == 14):
		psf_matrix = psf.PSF(psf.GAUSSIAN | psf.TWOPHOTON | psf.PARAXIAL, **args)
		print('psf.GAUSSIAN | psf.TWOPHOTON | psf.PARAXIAL generated')
	
	if empsf:
		psf_matrix = psf_matrix.expsf
	if expsf:
		psf_matrix = psf_matrix.empsf
		
	
	if psfvol:
		# psf_matrix = normalize_matrix(psf_matrix.volume())
		psf_matrix = psf_matrix.volume()
		psf_matrix = psf_matrix[:realshape[0],:,:]
		psf_matrix = psf_matrix[:,:realshape[1],:realshape[1]]
	else: 
		#psf_matrix = normalize_matrix(psf.mirror_symmetry(psf_matrix.data))
		psf_matrix = psf.mirror_symmetry(psf_matrix.data)
		psf_matrix = psf_matrix[:realshape[1],:realshape[1]]
	
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
		
	print('PSF shape: ', psf_matrix.shape)
	return psf_matrix
	
def normalize_matrix(matrix):
	"""This function normalize a matrix"""
	return np.uint8(255*matrix)
	
def constructpsf(metadata, channel, psf_vol, psftype):
	"""This function defines the parameters to build a psf"""
	if psf_vol:
		shape = (int((metadata['Axis 3 Parameters Common']['MaxSize']/2)+1),int((metadata['Axis 0 Parameters Common']['MaxSize']/2)+1))
		dims = (metadata['Axis 3 Parameters Common']['EndPosition']/1000,metadata['Axis 0 Parameters Common']['EndPosition'])
	else:
		shape = (int((metadata['Axis 0 Parameters Common']['MaxSize']/2)+1),int((metadata['Axis 0 Parameters Common']['MaxSize']/2)+1))
		dims = (metadata['Axis 0 Parameters Common']['EndPosition'],metadata['Axis 0 Parameters Common']['EndPosition'])
	ex_wavelen = metadata['Channel '+str(channel)+' Parameters']['ExcitationWavelength']
	em_wavelen = metadata['Channel '+str(channel)+' Parameters']['EmissionWavelength']
	num_aperture = metadata['num_aperture']
	pinhole_radius = metadata['pinhole_radius']
	magnification = metadata['magnification']
	refr_index = metadata['refr_index']
	print('shape: ', shape)
	print('dims: ', dims)
	print('ex_wavelen: ',ex_wavelen)
	print('em_wavelen: ',em_wavelen)
	print('num_aperture: ',num_aperture)
	print('pinhole_radius: ',pinhole_radius)
	print('magnification: ', magnification)
	print('refr_index: ', refr_index)
	return psf_generator(psfvol=psf_vol, psftype=psftype, shape=shape, dims=dims, ex_wavelen=ex_wavelen, num_aperture=num_aperture, pinhole_radius=pinhole_radius, refr_index=refr_index,
	magnification=magnification, em_wavelen=em_wavelen, realshape=(int(metadata['Axis 3 Parameters Common']['MaxSize']),int(metadata['Axis 0 Parameters Common']['MaxSize'])))	

def shape_psf(tensor, metadata, psftype):
	"""This function defines the dimensions that the psf will have"""
	global inx0, inx1, inx2, inx3, inx0p, inx1p, inx2p, inx3p
	inx0, inx1, inx2, inx3 = (None, None, None, None)
	inx0p, inx1p, inx2p, inx3p = (None, None, None, None)
	
	dimtensor = tensor.ndim
	
	if (dimtensor==2):
		multipsf = constructpsf(metadata, 1, False, psftype)
			
	if (dimtensor==3):
		import src.imageFunctions as imf
		if(imf.istiffRGB(tensor.shape)):
			print('Generating psf: ')
			multipsf = constructpsf(metadata, 1, False, psftype)
		else:	
			if ('slices' in metadata):
				multipsf = constructpsf(metadata, 1, True, psftype)
			if ('frames' in metadata):
				multipsf = constructpsf(metadata, 1, False, psftype)
			if ('channels' in metadata):
				multipsf = np.zeros(tensor.shape)
				for c in range(metadata['channels']['value']):
					print('\nGenerating psf channel: ',c)
					updateIndex(metadata['channels']['index'], c)
					multipsf[inx0p:inx0,inx1p:inx1,inx2p:inx2] = constructpsf(metadata, c+1, False, psftype)
					
	if (dimtensor==4):
		if(('channels' in metadata) and ('slices' in metadata)):
			multipsf = np.zeros(tensor.shape)
			for c in range(metadata['channels']['value']):
				print('\nGenerating psf channel: ',c)
				updateIndex(metadata['channels']['index'], c)
				multipsf[inx0p:inx0,inx1p:inx1,inx2p:inx2,inx3p:inx3] = constructpsf(metadata, c+1, True, psftype)
				
		if(('channels' in metadata) and ('frames' in metadata)):
			multipsf = np.zeros((metadata['channels']['value'],metadata['X'],metadata['Y']))
			for c in range(metadata['channels']['value']):
				print('\nGenerating psf channel: ',c)
				updateIndex(0, c)
				multipsf[inx0p:inx0,inx1p:inx1,inx2p:inx2] = constructpsf(metadata, c+1, False, psftype)
				
		if(('frames' in metadata) and ('slices' in metadata)):
			multipsf = constructpsf(metadata, 1, True, psftype)				
			
	#from tifffile import imsave
	#imsave('psf_matrix.tif', np.uint8(multipsf), metadata = {'axes':'TZCYX'}, imagej=True)
			
	return multipsf

def updateIndex(chnindex, pos):
	global inx0, inx1, inx2, inx3, inx0p, inx1p, inx2p, inx3p

	if (chnindex==0):
		inx0 = pos+1
		inx0p = pos
	if (chnindex==1):
		inx1 = pos+1
		inx1p = pos
	if (chnindex==2):
		inx2 = pos+1
		inx2p = pos
	if (chnindex==3):
		inx3 = pos+1
		inx3p = pos