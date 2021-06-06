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

from tkinter import messagebox
import oibread as oib
import createPSF as cpsf
import deconvolution as dv
import tiff as tif
import numpy as np
import interfaceTools as it
	

entryIterations,entryWeight,dropdownImg, dropdownPSF, metadata, multipsf, opcDeconv = (None,None,None,None,None,None, None)
entrynum_aperture, entrypinhole_radius, entrymagnification, entrydimz, entrydimr = (None,None,None,None, None)
entryex_wavelench1, entryem_wavelench1, entryex_wavelench2, entryem_wavelench2, entryex_wavelench3, entryem_wavelench3, entryex_wavelench4, entryem_wavelench4 = (None,None,None,None,None,None,None,None)

def deconvolution_parameters():
	global dropdownImg, dropdownPSF, metadata, opcPsf
	global entryex_wavelen, entryem_wavelen, entrynum_aperture, entrypinhole_radius, entrymagnification, entrydimz, entrydimr
	global entryex_wavelench1, entryem_wavelench1, entryex_wavelench2, entryem_wavelench2, entryex_wavelench3, entryem_wavelench3, entryex_wavelench4, entryem_wavelench4

	if (it.file.split('.')[1]=='oib'):
		metadata = oib.getMetadata(it.file)
		
	opcPsf = it.NewWindow(it.file.split('/')[len(it.file.split('/'))-1],'300x550') #Objeto de la clase NewWindow
	
	opcPsf.createLabel('PSF parameters ',20,70)
	opcPsf.createLabel('Ex_wavelenCh1:                       [nm]',20,100)
	opcPsf.createLabel('Ex_wavelenCh2:                       [nm]',20,130)
	opcPsf.createLabel('Ex_wavelenCh3:                       [nm]',20,160)
	opcPsf.createLabel('Ex_wavelenCh4:                       [nm]',20,190)
	
	opcPsf.createLabel('Em_wavelenCh1:                     [nm] ',20,220)
	opcPsf.createLabel('Em_wavelenCh2:                     [nm]',20,250)
	opcPsf.createLabel('Em_wavelenCh3:                     [nm]',20,280)
	opcPsf.createLabel('Em_wavelenCh4:                     [nm]',20,310)
	
	opcPsf.createLabel('Num_aperture:',20,340)
	opcPsf.createLabel('Pinhole_radius:                         [um]',20,370)
	opcPsf.createLabel('Magnification:',20,400)
	opcPsf.createLabel('Refr_index:',20,430)
	opcPsf.createLabel('Dim_z:                                       [um]',20,460)
	opcPsf.createLabel('Dim_xy:                                       [um]',20,490)
	
	entryex_wavelench1 = opcPsf.createEntry(metadata['Channel 1 Parameters']['ExcitationWavelength'],160,100)
	entryex_wavelench2 = opcPsf.createEntry(metadata['Channel 2 Parameters']['ExcitationWavelength'],160,130)
	entryex_wavelench3 = opcPsf.createEntry(metadata['Channel 3 Parameters']['ExcitationWavelength'],160,160)
	entryex_wavelench4 = opcPsf.createEntry(metadata['Channel 4 Parameters']['ExcitationWavelength'],160,190)
	
	entryem_wavelench1 = opcPsf.createEntry(metadata['Channel 1 Parameters']['EmissionWavelength'],160,220)
	entryem_wavelench2 = opcPsf.createEntry(metadata['Channel 2 Parameters']['EmissionWavelength'],160,250)
	entryem_wavelench3 = opcPsf.createEntry(metadata['Channel 3 Parameters']['EmissionWavelength'],160,280)
	entryem_wavelench4 = opcPsf.createEntry(metadata['Channel 4 Parameters']['EmissionWavelength'],160,310)
	
	entrynum_aperture = opcPsf.createEntry(metadata['num_aperture'],160,340)
	entrypinhole_radius = opcPsf.createEntry(metadata['pinhole_radius'],160,370)
	entrymagnification = opcPsf.createEntry(metadata['magnification'],160,400)
	entrymagnification = opcPsf.createEntry(metadata['refr_index'],160,430)
	entrydimz = opcPsf.createEntry(metadata['Axis 3 Parameters Common']['EndPosition']/1000,160,460)
	entrydimr = opcPsf.createEntry(metadata['Axis 0 Parameters Common']['EndPosition'],160,490)
	
	opcPsf.createLabel('PSF type: ',20,10)
	dropdownPSF = opcPsf.createCombobox(20,40)
	opcPsf.createButton('Generate psf', createpsf_event, 'bottom')

def deconvolution_event():
	global entryIterations, entryWeight, dropdownImg, metadata
	img_tensor = oib.get_matrix_oib(it.file)
	try:
		tensor_deconv = dv.deconvolutionMain(img_tensor,multipsf,int(entryIterations.get()),int(entryWeight.get()), it.file.split('/')[len(it.file.split('/'))-1], metadata)
		deconvimg = it.NewWindow('Deconvolution'+it.file.split('/')[len(it.file.split('/'))-1]+' i:'+entryIterations.get()+' w:'+entryWeight.get())
		deconvimg.desplay_image('Deconvolution '+it.file.split('/')[len(it.file.split('/'))-1]+' i:'+entryIterations.get()+' w:'+entryWeight.get(), tensor_deconv)
	except AttributeError:
		messagebox.showinfo(message='There are empty parameters, please check')
	
def createpsf_event():
	global entryex_wavelench1, entryem_wavelench1, entryex_wavelench2, entryem_wavelench2, entryex_wavelench3, entryem_wavelench3, entryex_wavelench4, entryem_wavelench4
	global multipsf, opcPsf, entryIterations, entryWeight, dropdownPSF, metadata
	metadata['Channel 1 Parameters']['ExcitationWavelength'] = float(entryex_wavelench1.get())
	metadata['Channel 2 Parameters']['ExcitationWavelength'] = float(entryex_wavelench2.get())
	metadata['Channel 3 Parameters']['ExcitationWavelength'] = float(entryex_wavelench3.get())
	metadata['Channel 4 Parameters']['ExcitationWavelength'] = float(entryex_wavelench4.get())
	metadata['Channel 1 Parameters']['EmissionWavelength'] = float(entryem_wavelench1.get())
	metadata['Channel 2 Parameters']['EmissionWavelength'] = float(entryem_wavelench2.get())
	metadata['Channel 3 Parameters']['EmissionWavelength'] = float(entryem_wavelench3.get())
	metadata['Channel 4 Parameters']['EmissionWavelength'] = float(entryem_wavelench4.get())
	metadata['num_aperture'] = float(entrynum_aperture.get())
	metadata['pinhole_radius'] = float(entrypinhole_radius.get())
	metadata['magnification'] = float(entrymagnification.get())
	metadata['Axis 3 Parameters Common']['EndPosition'] = float(entrydimz.get())
	metadata['Axis 0 Parameters Common']['EndPosition'] = float(entrydimr.get())
	psftype = dropdownPSF.current()
	opcPsf.destroy()
	
	multipsf = cpsf.shape_psf(oib.get_matrix_oib(it.file),metadata, psftype)
	
	opcDeconv = it.NewWindow('Deconv parameters','300x200') #Objeto de la clase NewWindow
	
	opcDeconv.createLabel('Image: ',20,20)
	opcDeconv.createLabel('PSF: ',20,50)
	opcDeconv.createLabel('Iterations: ',20,80)
	opcDeconv.createLabel('Weight TV: ',20,110)
	
	entryimg = opcDeconv.createEntry(it.file.split('/')[len(it.file.split('/'))-1],110,20, 25,True)
	entrypsf = opcDeconv.createEntry('psf_'+it.file.split('/')[len(it.file.split('/'))-1].split('.')[0],110,50,25, True)
	
	entryIterations = opcDeconv.createEntry('',110,80,25)
	entryWeight = opcDeconv.createEntry('',110,110,25)
	opcDeconv.createButtonXY('Deconvolution '+entryIterations.get()+' '+entryWeight.get(), deconvolution_event, 100, 140)

#Se crea la ventana principal del programa
it.createWindowMain()
#Se crea menu desplegable
menu = it.createMenu()
#Se añaden las opciones del menu
opc1 = it.createOption(menu)
it.createCommand(opc1, "Open", it.openFile)
it.createCommand(opc1, "Save", it.saveFile)
it.createCommand(opc1, "Exit", it.mainWindow.quit)
it.createCascade(menu, 'File', opc1)

opc2 = it.createOption(menu)
it.createCommand(opc2, "Deconvolution", deconvolution_parameters)
#it.createCommand(opc2, "Guardar", it.saveFile)
#it.createCommand(opc2, "Salir", mainWindow.quit)
it.createCascade(menu, 'Image', opc2)

#it.createStatusBar()
statusBar = it.createStatusBar()
#statusBar['text']=dv.message

it.mainWindow.mainloop()