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

import oibread as oib
import createPSF as cpsf
import deconvolution as dv
import tiff as tif
import numpy as np
import interfaceTools as it
	

entryIterations,entryWeight,dropdownImg, dropdownPSF, metadata = (None,None,None,None, None)
entryex_wavelen, entryem_wavelen, entrynum_aperture, entrypinhole_radius, entrymagnification = (None,None,None,None, None)

def deconvolution_parameters():
	global entryIterations, entryWeight, dropdownImg, dropdownPSF, metadata
	global entryex_wavelen, entryem_wavelen, entrynum_aperture, entrypinhole_radius, entrymagnification

	if (it.file.split('.')[1]=='oib'):
		metadata = oib.getMetadata(it.file)
		print('Hola estoy aqui')
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
	opcDeconv.createButton('Deconvolution', deconvolution_event, 'bottom')
	opcDeconv.createButton('Generate psf', createpsf_event, 'bottom')

def deconvolution_event():
	global entryIterations, entryWeight, dropdownImg, dropdownPSF
	img_tensor = oibr.get_matrix_oib(it.file)
	#dv.deconvolutionMain(img_tensor,multipsf,1,20)
	dv.deconvolutionMain(img_tensor,multipsf,entryIterations.get(),entryWeight.get())
	
def createpsf_event():
	metadata['Channel 1 Parameters']['ExcitationWavelength'] = float(entryex_wavelen.get())
	metadata['Channel 1 Parameters']['EmissionWavelength'] = float(entryem_wavelen.get())
	metadata['num_aperture'] = float(entrynum_aperture.get())
	metadata['pinhole_radius'] = float(entrypinhole_radius.get())
	metadata['magnification'] = float(entrymagnification.get())
	
	cpsf.shape_psf(oib.get_matrix_oib(it.file),metadata)

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