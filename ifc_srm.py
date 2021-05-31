#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# ifc_srm.py
# Archivo principal
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

import oibread as oibr
import createPSF as cpsf
import deconvolution as dv
import tiff as tif
import numpy as np
import interfaceTools as it

def constructpsf(metadata, channel):
	shape = (int((metadata['Axis 3 Parameters Common']['MaxSize']/2)+1),int((metadata['Axis 0 Parameters Common']['MaxSize']/2)+1))
	dims = (metadata['Axis 3 Parameters Common']['EndPosition']/1000,metadata['Axis 0 Parameters Common']['EndPosition'])
	ex_wavelen = metadata['Channel '+str(channel)+' Parameters']['ExcitationWavelength']
	em_wavelen = metadata['Channel '+str(channel)+' Parameters']['EmissionWavelength']
	num_aperture = 1.35
	pinhole_radius = 85000/1000
	magnification = 0.75
	refr_index = 1.5
	print('shape: ', shape)
	print('dims: ', dims)
	print('ex_wavelen: ',ex_wavelen)
	print('em_wavelen: ',em_wavelen)
	print('num_aperture: ',num_aperture)
	print('pinhole_radius: ',pinhole_radius)
	print('magnification: ', magnification)	
	return cpsf.psf_generator(psfvol=True , shape=shape, dims=dims, ex_wavelen=ex_wavelen, num_aperture=num_aperture, pinhole_radius=pinhole_radius, refr_index=refr_index,
	magnification=magnification, em_wavelen=em_wavelen, realshape=(int(metadata['Axis 3 Parameters Common']['MaxSize']),int(metadata['Axis 0 Parameters Common']['MaxSize'])))

def get_metadata(filepath):
	metadata = oibr.getMetadata(filepath)
	#print(metadata)
	img_tensor = oibr.get_matrix_oib(filepath)
	print(img_tensor.shape) 
	dimtensor = img_tensor.ndim
	print(dimtensor)

	if (dimtensor>2):
		multipsf = np.zeros((img_tensor.shape[0],img_tensor.shape[1],img_tensor.shape[2],img_tensor.shape[3]))
		for i in range(img_tensor.shape[0]):
			multipsf[i,:,:,:] = constructpsf(metadata, i+1)		
		# from tifffile import imsave
		# imsave('psf_vol.tif', np.uint8(multipsf),  metadata = {'axes':'TZCYX'}, imagej=True)
		#dv.deconvolutionMain(img_tensor,multipsf,2,20)
	return metadata
	

entryIterations,entryWeight,dropdownImg, dropdownPSF = (None,None,None,None)

def deconvolution_parameters():
	global entryIterations, entryWeight, dropdownImg, dropdownPSF
	
	metadata = get_metadata(it.file)
	opcDeconv = it.NewWindow('Deconvolution parameters','300x380') #Objeto de la clase NewWindow
	
	opcDeconv.createLabel('Image: ',20,20)
	opcDeconv.createLabel('PSF: ',20,50)
	opcDeconv.createLabel('Iterations: ',20,80)
	opcDeconv.createLabel('Weight: ',20,110)
	opcDeconv.createLabel('Ex_wavelen: ',20,140)
	opcDeconv.createLabel('Em_wavelen: ',20,170)
	opcDeconv.createLabel('Num_aperture: ',20,200)
	opcDeconv.createLabel('Pinhole_radius: ',20,230)
	opcDeconv.createLabel('Magnification: ',20,260)
	
	entryIterations = opcDeconv.createEntry('50',110,80)
	entryWeight = opcDeconv.createEntry('20',110,110)
	
	entryex_wavelen = opcDeconv.createEntry('30',140,140)
	entryex_wavelen.insert(0, metadata['Channel 1 Parameters']['ExcitationWavelength'])
	
	entryem_wavelen = opcDeconv.createEntry('40',140,170)
	entryem_wavelen.insert(0, metadata['Channel 1 Parameters']['EmissionWavelength'])
	
	entrynum_aperture = opcDeconv.createEntry('51',140,200)
	entrynum_aperture.insert(0, 1.35)
	
	entrypinhole_radius = opcDeconv.createEntry('60',140,230)
	entrypinhole_radius.insert(0, 85000/1000)
	
	entrymagnification = opcDeconv.createEntry('70',140,260)
	entrymagnification.insert(0, 0.75)
	
	entryimg = opcDeconv.createEntry(it.file.split('/')[len(it.file.split('/'))-1],110,20, True)
	#entryimg.insert(0, it.file)	
	
	entrypsf = opcDeconv.createEntry('psf_'+it.file.split('/')[len(it.file.split('/'))-1],110,50, True)
	#entrypsf.insert(0, 'psf_'+it.file)
	
	# dropdownImg = opcDeconv.createCombobox(110,20)
	# dropdownPSF = opcDeconv.createCombobox(110,50)
	opcDeconv.createButton('OK', deconvolution_event, 'bottom')

def deconvolution_event():
	global entryIterations, entryWeight, dropdownImg, dropdownPSF
	#dv.deconvolutionMain(img_tensor,multipsf,1,20)
	dv.deconvolutionMain(img_tensor,multipsf,entryIterations.get(),entryWeight.get())

#Se crea la ventana principal del programa
it.createWindowMain()
#Se crea menu desplegable
menu = it.createMenu()
#Se añaden las opciones del menu
opc1 = it.createOption(menu)
it.createCommand(opc1, "Abrir", it.openFile)
it.createCommand(opc1, "Guardar", it.saveFile)
it.createCommand(opc1, "Salir", it.mainWindow.quit)
it.createCascade(menu, 'Archivo', opc1)

opc2 = it.createOption(menu)
it.createCommand(opc2, "Deconvolution", deconvolution_parameters)
#it.createCommand(opc2, "Guardar", it.saveFile)
#it.createCommand(opc2, "Salir", mainWindow.quit)
it.createCascade(menu, 'Image', opc2)

it.createStatusBar()
statusBar = it.createStatusBar()
statusBar['text']=dv.message

it.mainWindow.mainloop()