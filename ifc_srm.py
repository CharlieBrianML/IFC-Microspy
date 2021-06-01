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
entryex_wavelen, entryem_wavelen, entrynum_aperture, entrypinhole_radius, entrymagnification, entrydimz, entrydimr = (None,None,None,None,None,None, None)

def deconvolution_parameters():
	global entryIterations, entryWeight, dropdownImg, dropdownPSF, metadata
	global entryex_wavelen, entryem_wavelen, entrynum_aperture, entrypinhole_radius, entrymagnification, entrydimz, entrydimr

	if (it.file.split('.')[1]=='oib'):
		metadata = oib.getMetadata(it.file)
		
	opcDeconv = it.NewWindow('Deconvolution parameters','300x650') #Objeto de la clase NewWindow
	
	opcDeconv.createLabel('Image: ',20,20)
	opcDeconv.createLabel('PSF: ',20,50)
	opcDeconv.createLabel('Iterations: ',20,80)
	opcDeconv.createLabel('Weight TV: ',20,110)
	
	opcDeconv.createLabel('PSF parameters ',20,170)
	opcDeconv.createLabel('Ex_wavelenCh1:                       [nm]',20,200)
	opcDeconv.createLabel('Ex_wavelenCh2:                       [nm]',20,230)
	opcDeconv.createLabel('Ex_wavelenCh3:                       [nm]',20,260)
	opcDeconv.createLabel('Ex_wavelenCh4:                       [nm]',20,290)
	
	opcDeconv.createLabel('Em_wavelenCh1:                     [nm] ',20,320)
	opcDeconv.createLabel('Em_wavelenCh2:                     [nm]',20,350)
	opcDeconv.createLabel('Em_wavelenCh3:                     [nm]',20,380)
	opcDeconv.createLabel('Em_wavelenCh4:                     [nm]',20,410)
	
	opcDeconv.createLabel('Num_aperture:',20,440)
	opcDeconv.createLabel('Pinhole_radius:                         [um]',20,470)
	opcDeconv.createLabel('Magnification:',20,500)
	opcDeconv.createLabel('Refr_index:',20,530)
	opcDeconv.createLabel('Dim_z:                                       [um]',20,560)
	opcDeconv.createLabel('Dim_r:                                       [um]',20,590)
	
	entryimg = opcDeconv.createEntry(it.file.split('/')[len(it.file.split('/'))-1],110,20, 25,True)
	entrypsf = opcDeconv.createEntry('psf_'+it.file.split('/')[len(it.file.split('/'))-1],110,50,25, True)	
	
	entryIterations = opcDeconv.createEntry('',110,80,25)
	entryWeight = opcDeconv.createEntry('',110,110,25)
	
	entryex_wavelench1 = opcDeconv.createEntry(metadata['Channel 1 Parameters']['ExcitationWavelength'],160,200)
	entryex_wavelench2 = opcDeconv.createEntry(metadata['Channel 2 Parameters']['ExcitationWavelength'],160,230)
	entryex_wavelench3 = opcDeconv.createEntry(metadata['Channel 3 Parameters']['ExcitationWavelength'],160,260)
	entryex_wavelench4 = opcDeconv.createEntry(metadata['Channel 4 Parameters']['ExcitationWavelength'],160,290)
	
	entryem_wavelench1 = opcDeconv.createEntry(metadata['Channel 1 Parameters']['EmissionWavelength'],160,320)
	entryem_wavelench2 = opcDeconv.createEntry(metadata['Channel 2 Parameters']['EmissionWavelength'],160,350)
	entryem_wavelench3 = opcDeconv.createEntry(metadata['Channel 3 Parameters']['EmissionWavelength'],160,380)
	entryem_wavelench4 = opcDeconv.createEntry(metadata['Channel 4 Parameters']['EmissionWavelength'],160,410)
	
	entrynum_aperture = opcDeconv.createEntry(metadata['num_aperture'],160,440)
	entrypinhole_radius = opcDeconv.createEntry(metadata['pinhole_radius'],160,470)
	entrymagnification = opcDeconv.createEntry(metadata['magnification'],160,500)
	entrymagnification = opcDeconv.createEntry(metadata['refr_index'],160,530)
	entrydimz = opcDeconv.createEntry(metadata['Axis 3 Parameters Common']['EndPosition']/1000,160,560)
	entrydimr = opcDeconv.createEntry(metadata['Axis 0 Parameters Common']['EndPosition'],160,590)
	
	# dropdownImg = opcDeconv.createCombobox(110,20)
	# dropdownPSF = opcDeconv.createCombobox(110,50)
	opcDeconv.createButtonXY('Deconvolution', deconvolution_event, 100, 140)
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
	metadata['Axis 3 Parameters Common']['EndPosition'] = float(entrydimz.get())
	metadata['Axis 0 Parameters Common']['EndPosition'] = float(entrydimr.get())
	
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