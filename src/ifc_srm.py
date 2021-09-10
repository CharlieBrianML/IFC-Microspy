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
from shutil import rmtree
from imageFunctions import istiffRGB
import oibread as oib
import createPSF as cpsf
import deconvolution as dv
import tiff as tif
import numpy as np
import interfaceTools as it
import sr
import sys

entryIterations,entryWeight,dropdownImg, dropdownPSF, metadata, multipsf, opcDeconv = (None,None,None,None,None,None, None)
entrynum_aperture, entrypinhole_radius, entrymagnification, entrydimz, entrydimr, tensor_deconv, img_tensor = (None,None,None,None,None,None,None)
entryex_wavelench1, entryem_wavelench1, entryex_wavelench2, entryem_wavelench2, entryex_wavelench3, entryem_wavelench3, entryex_wavelench4, entryem_wavelench4, entryrefr_index = (None,None,None,None,None,None,None,None,None)

metadata_init = {'Channel 1 Parameters':{'ExcitationWavelength':0.0,'EmissionWavelength':0.0},'Channel 2 Parameters':{'ExcitationWavelength':0.0,'EmissionWavelength':0.0},
'Channel 3 Parameters':{'ExcitationWavelength':0.0, 'EmissionWavelength':0.0}, 'Channel 4 Parameters':{'ExcitationWavelength':0.0, 'EmissionWavelength':0.0}, 'refr_index': 0.0,
'num_aperture':0.0,'pinhole_radius':0.0,'magnification':0.0, 'Axis 3 Parameters Common':{'EndPosition':0.0,'StartPosition':0.0}, 'Axis 0 Parameters Common':{'EndPosition':0.0, 'StartPosition':0.0}}

def psf_winmnyparmts():
	global metadata, opcPsf
	global entryex_wavelen, entryem_wavelen, entrynum_aperture, entrypinhole_radius, entrymagnification, entrydimz, entrydimr, entryrefr_index
	global entryex_wavelench1, entryem_wavelench1, entryex_wavelench2, entryem_wavelench2, entryex_wavelench3, entryem_wavelench3, entryex_wavelench4, entryem_wavelench4
	opcPsf.createLabel('PSF parameters ',20,70)
	opcPsf.createLabel('Ex_wavelenCh1:                        [nm]',20,100)
	opcPsf.createLabel('Ex_wavelenCh2:                        [nm]',20,130)
	opcPsf.createLabel('Ex_wavelenCh3:                        [nm]',20,160)
	opcPsf.createLabel('Ex_wavelenCh4:                        [nm]',20,190)
	
	opcPsf.createLabel('Em_wavelenCh1:                      [nm] ',20,220)
	opcPsf.createLabel('Em_wavelenCh2:                      [nm]',20,250)
	opcPsf.createLabel('Em_wavelenCh3:                      [nm]',20,280)
	opcPsf.createLabel('Em_wavelenCh4:                      [nm]',20,310)
	
	opcPsf.createLabel('Num_aperture:',20,340)
	opcPsf.createLabel('Pinhole_radius:                          [um]',20,370)
	opcPsf.createLabel('Magnification:',20,400)
	opcPsf.createLabel('Refr_index:',20,430)
	opcPsf.createLabel('Dim_z:                                        [um]',20,460)
	opcPsf.createLabel('Dim_xy:                                        [um]',20,490)
	
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
	entryrefr_index = opcPsf.createEntry(metadata['refr_index'],160,430)
	entrydimz = opcPsf.createEntry((metadata['Axis 3 Parameters Common']['StartPosition']-metadata['Axis 3 Parameters Common']['EndPosition'])/1000,160,460)
	entrydimr = opcPsf.createEntry(metadata['Axis 0 Parameters Common']['EndPosition'],160,490)
	
def psf_winolyparmts():
	global metadata, opcPsf
	global entryex_wavelen, entryem_wavelen, entrynum_aperture, entrypinhole_radius, entrymagnification, entrydimz, entrydimr, entryrefr_index
	global entryex_wavelench1, entryem_wavelench1, entryex_wavelench2, entryem_wavelench2, entryex_wavelench3, entryem_wavelench3, entryex_wavelench4, entryem_wavelench4
	opcPsf.window.geometry('300x365')
	opcPsf.createLabel('PSF parameters ',20,70)
	opcPsf.createLabel('Ex_wavelenCh1:                         [nm]',20,100)
	opcPsf.createLabel('Em_wavelenCh1:                       [nm] ',20,130)
	
	opcPsf.createLabel('Num_aperture:',20,160)
	opcPsf.createLabel('Pinhole_radius:                           [um]',20,190)
	opcPsf.createLabel('Magnification:',20,220)
	opcPsf.createLabel('Refr_index:',20,250)
	opcPsf.createLabel('Dim_z:                                         [um]',20,280)
	opcPsf.createLabel('Dim_xy:                                         [um]',20,310)
	
	entryex_wavelench1 = opcPsf.createEntry(metadata['Channel 1 Parameters']['ExcitationWavelength'],160,100)
	entryem_wavelench1 = opcPsf.createEntry(metadata['Channel 1 Parameters']['EmissionWavelength'],160,130)
	
	entrynum_aperture = opcPsf.createEntry(metadata['num_aperture'],160,160)
	entrypinhole_radius = opcPsf.createEntry(metadata['pinhole_radius'],160,190)
	entrymagnification = opcPsf.createEntry(metadata['magnification'],160,220)
	entryrefr_index = opcPsf.createEntry(metadata['refr_index'],160,250)
	entrydimz = opcPsf.createEntry((metadata['Axis 3 Parameters Common']['StartPosition']-metadata['Axis 3 Parameters Common']['EndPosition'])/1000,160,280)
	entrydimr = opcPsf.createEntry(metadata['Axis 0 Parameters Common']['EndPosition'],160,310)		

def psf_parameters():
	global dropdownImg, dropdownPSF, metadata, opcPsf
	global entryex_wavelen, entryem_wavelen, entrynum_aperture, entrypinhole_radius, entrymagnification, entrydimz, entrydimr, entryrefr_index
	global entryex_wavelench1, entryem_wavelench1, entryex_wavelench2, entryem_wavelench2, entryex_wavelench3, entryem_wavelench3, entryex_wavelench4, entryem_wavelench4
	
	try:

		if (it.file.split('.')[1]=='oib'):
			metadata = oib.getMetadata(it.file)
		elif (it.file.split('.')[1]=='tif'):
			try:
				metadata = tif.getMetadata(it.file)
				if (len(metadata)==0):
					metadata = metadata_init
				print(metadata)
			except KeyError:
				messagebox.showinfo(message='No metadata found, please fill in the parameters manually')
				metadata = metadata_init
				if (it.windows_img[-1].tensor_img.ndim>2):
					metadata['Axis 3 Parameters Common']['MaxSize'] = it.windows_img[-1].tensor_img.shape[0]
					metadata['Axis 0 Parameters Common']['MaxSize'] = it.windows_img[-1].tensor_img.shape[1]
				else:	
					metadata['Axis 0 Parameters Common']['MaxSize'] = it.windows_img[-1].tensor_img.shape[0]
					metadata['Axis 3 Parameters Common']['MaxSize'] = it.windows_img[-1].tensor_img.shape[1]
		else:
			metadata = metadata_init
			metadata['Axis 0 Parameters Common']['MaxSize'] = it.windows_img[-1].tensor_img.shape[0]
			metadata['Axis 3 Parameters Common']['MaxSize'] = it.windows_img[-1].tensor_img.shape[1]		
			
		opcPsf = it.NewWindow(it.file.split('/')[len(it.file.split('/'))-1],'300x550') #Objeto de la clase NewWindow
		
		#Creation of the psf parameters window
		if (it.windows_img[-1].tensor_img.ndim==4):
			psf_winmnyparmts()
		elif (it.windows_img[-1].tensor_img.ndim==3):
		
			if (istiffRGB(it.windows_img[-1].tensor_img.shape)): 	#Matrix of the form (x,y,r)
				psf_winolyparmts()
			elif (it.windows_img[-1].tensor_img.shape[0]>4): 		#Matrix of the form (z,x,y)
				psf_winolyparmts()
			else: 													#Matrix of the form (c,x,y)
				psf_winmnyparmts()
		else: 
			psf_winolyparmts()

		
		opcPsf.createLabel('PSF type: ',20,10)
		dropdownPSF = opcPsf.createCombobox(20,40)
		opcPsf.createButton('Generate psf', createpsf_event, 'bottom')
	except IndexError:
		messagebox.showinfo(message='No file has been opened')

def deconvolution_event():
	global entryIterations, entryWeight, dropdownImg, metadata, tensor_deconv, img_tensor
	img_tensor = it.windows_img[-1].tensor_img
	try:
		if(int(entryIterations.get())>0):
			tensor_deconv = dv.deconvolutionMain(img_tensor,multipsf,int(entryIterations.get()),int(entryWeight.get()), it.file.split('/')[len(it.file.split('/'))-1], metadata)
			deconvimg = it.NewWindow('Deconvolution'+it.file.split('/')[len(it.file.split('/'))-1]+' i:'+entryIterations.get()+' w:'+entryWeight.get())
			it.windows_img.append(deconvimg)
			if(tensor_deconv.ndim==4):
				deconvimg.desplay_image('Deconvolution '+it.file.split('/')[len(it.file.split('/'))-1]+' i:'+entryIterations.get()+' w:'+entryWeight.get(), tensor_deconv)
			elif(tensor_deconv.ndim==3):
				import imageFunctions as imf
				if(imf.istiffRGB(tensor_deconv.shape)):
					deconvimg.placeImage(np.uint8(tensor_deconv))
				else: 
					deconvimg.desplay_image('Deconvolution '+it.file.split('/')[len(it.file.split('/'))-1]+' i:'+entryIterations.get()+' w:'+entryWeight.get(), tensor_deconv)
			else:
				deconvimg.placeImage(tensor_deconv)
				deconvimg.tensor_img = tensor_deconv
		else:
			messagebox.showinfo(message='Iteration value equal to zero is not accepted')
	except (AttributeError, ValueError):
		messagebox.showinfo(message='There are empty parameters, please check')
	
def createpsf_event():
	global entryex_wavelench1, entryem_wavelench1, entryex_wavelench2, entryem_wavelench2, entryex_wavelench3, entryem_wavelench3, entryex_wavelench4, entryem_wavelench4
	global multipsf, opcPsf, entryIterations, entryWeight, dropdownPSF, metadata, entryrefr_index
	
	entryex = (entryex_wavelench1, entryex_wavelench2, entryex_wavelench3, entryex_wavelench4)
	entryem = (entryem_wavelench1, entryem_wavelench2, entryem_wavelench3, entryem_wavelench4)
	
	# Extracting available metadata
	for ch in range(4):
		if(entryex[ch]!=None):
			try: 
				metadata['Channel '+str(ch+1)+' Parameters']['ExcitationWavelength'] = float(entryex[ch].get())
			except:
				print('Error: ',sys.exc_info()[0])
			
	for ch in range(4):
		if(entryem[ch]!=None):
			try:
				metadata['Channel '+str(ch+1)+' Parameters']['EmissionWavelength'] = float(entryem[ch].get())
			except:
				print('Error: ',sys.exc_info()[0])
	
	metadata['num_aperture'] = float(entrynum_aperture.get())
	metadata['pinhole_radius'] = float(entrypinhole_radius.get())
	metadata['magnification'] = float(entrymagnification.get())
	metadata['refr_index'] = float(entryrefr_index.get())
	metadata['Axis 3 Parameters Common']['EndPosition'] = float(entrydimz.get())
	metadata['Axis 0 Parameters Common']['EndPosition'] = float(entrydimr.get())
		
	try:
		if ((metadata['num_aperture']/metadata['refr_index'])<=1.0):

			psftype = dropdownPSF.current()
			
			multipsf = cpsf.shape_psf(it.windows_img[-1].tensor_img,metadata, psftype)
			opcPsf.destroy()
			
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
		else: 
			messagebox.showinfo(message='Quotient of the numeric aperture ' +str(metadata['num_aperture'])+ ' and refractive index ' +str(metadata['refr_index'])+ ' is greater than 1.0')
	except ZeroDivisionError:
		messagebox.showinfo(message='Error, there are parameters that cannot be equal to zero')		
	
def neural_network_event():
	global tensor_deconv
	from sr import nn
	try:
		nn(it.windows_img[-1].tensor_img)
	except IndexError:
		messagebox.showinfo(message='There is no input parameter')
	
def on_closing():
	import os
	if not((messagebox.askyesno(message="Do you want to save the generated cache?", title="Cache"))):
		if (os.path.isdir('training_set')):
			rmtree("training_set")
	it.mainWindow.destroy()	
		
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
it.createCommand(opc2, "Deconvolution", psf_parameters)
it.createCommand(opc2, "Neural Network", neural_network_event)
#it.createCommand(opc2, "Zoom", mainWindow.quit)
it.createCascade(menu, 'Image', opc2)

#it.createStatusBar()
statusBar = it.createStatusBar()
#statusBar['text']=dv.message

it.mainWindow.protocol("WM_DELETE_WINDOW", on_closing)

it.mainWindow.mainloop()
