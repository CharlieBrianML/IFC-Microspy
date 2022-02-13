#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# ifc_main.py
# Archivo principal
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

from tkinter import messagebox
from shutil import rmtree
import numpy as np
import sys

from .imageFunctions import istiffRGB
import src.oibread as oib
import src.createPSF as cpsf
import src.deconvolution as dv
import src.tiff as tif
import src.interfaceTools as it
import src.esrgan

entryIterations,entryWeight,dropdownImg, dropdownPSF, metadata, multipsf, opcDeconv, opcTVD, entryWeighttvd, entryrefr_index = (None,None,None,None,None,None,None, None, None, None)
entrynum_aperture, entrypinhole_radius, entrymagnification, entrydimz, entrydimr, tensor_deconv, img_tensor, cmbxFile, opcimg = (None,None,None,None,None,None,None,None, None)
entryem_wavelen, entryex_wavelen, opcResize, entryX, entryY = (None, None,None, None, None)
index = -1

metadata_init = {'Channel 1 Parameters':{'ExcitationWavelength':0.0,'EmissionWavelength':0.0},'Channel 2 Parameters':{'ExcitationWavelength':0.0,'EmissionWavelength':0.0},
'Channel 3 Parameters':{'ExcitationWavelength':0.0, 'EmissionWavelength':0.0}, 'Channel 4 Parameters':{'ExcitationWavelength':0.0, 'EmissionWavelength':0.0}, 'refr_index': 0.0,
'num_aperture':0.0,'pinhole_radius':0.0,'magnification':0.0, 'Axis 3 Parameters Common':{'EndPosition':0.0,'StartPosition':0.0}, 'Axis 0 Parameters Common':{'EndPosition':0.0, 'StartPosition':0.0}}

def psf_winparmts(numch=1):
	global metadata, opcPsf, entryem_wavelen, entryex_wavelen
	global entrynum_aperture, entrypinhole_radius, entrymagnification, entrydimz, entrydimr, entryrefr_index
	
	posy = 100
	entryem_wavelen = []
	entryex_wavelen = []
	opcPsf.createLabel('PSF parameters ',20,70)
	for ch in range(numch):
		opcPsf.createLabel('Ex_wavelenCh'+str(ch+1)+':                        [nm]',20,posy)
		entryex_wavelen.append(opcPsf.createEntry(metadata['Channel '+str(ch+1)+' Parameters']['ExcitationWavelength'],160,posy))
		posy = posy + 30

	for ch in range(numch):
		opcPsf.createLabel('Em_wavelenCh'+str(ch+1)+':                      [nm] ',20,posy)
		entryem_wavelen.append(opcPsf.createEntry(metadata['Channel '+str(ch+1)+' Parameters']['EmissionWavelength'],160,posy))
		posy = posy + 30
		
	opcPsf.window.geometry('300x'+str(posy+210))	
	
	opcPsf.createLabel('Num_aperture:',20,posy)
	opcPsf.createLabel('Pinhole_radius:                          [um]',20,posy+30)
	opcPsf.createLabel('Magnification:',20,posy+60)
	opcPsf.createLabel('Refr_index:',20,posy+90)
	opcPsf.createLabel('Dim_z:                                        [um]',20,posy+120)
	opcPsf.createLabel('Dim_xy:                                        [um]',20,posy+150)
	
	entrynum_aperture = opcPsf.createEntry(metadata['num_aperture'],160,posy)
	entrypinhole_radius = opcPsf.createEntry(metadata['pinhole_radius'],160,posy+30)
	entrymagnification = opcPsf.createEntry(metadata['magnification'],160,posy+60)
	entryrefr_index = opcPsf.createEntry(metadata['refr_index'],160,posy+90)
	entrydimz = opcPsf.createEntry((metadata['Axis 3 Parameters Common']['StartPosition']-metadata['Axis 3 Parameters Common']['EndPosition'])/1000,160,posy+120)
	entrydimr = opcPsf.createEntry(metadata['Axis 0 Parameters Common']['EndPosition'],160,posy+150)

def psf_parameters(flg=True):
	global dropdownImg, dropdownPSF, metadata, opcPsf, opcimg, index
	global entryex_wavelen, entryem_wavelen, entrynum_aperture, entrypinhole_radius, entrymagnification, entrydimz, entrydimr, entryrefr_index
	global entryex_wavelench1, entryem_wavelench1, entryex_wavelench2, entryem_wavelench2, entryex_wavelench3, entryem_wavelench3, entryex_wavelench4, entryem_wavelench4
	
	opcimg = "Deconvolution"
	try:
		if(len(it.windows_img)>1 and flg):
			selectFile()
		else: 
			if (len(it.windows_img)==1):
				index = -1
			nameFile = it.windows_img[index].nameFile
			pathFile = it.windows_img[index].path

			if (nameFile.split('.')[-1]=='oib'):
				try: 
					metadata = oib.getMetadata(pathFile)
				except KeyError:
					messagebox.showinfo(message='No metadata found, please fill in the parameters manually')	
					metadata = metadata_init
					metadata['Axis 3 Parameters Common']['MaxSize'] = it.windows_img[index].tensor_img.shape[0]
					metadata['Axis 0 Parameters Common']['MaxSize'] = it.windows_img[index].tensor_img.shape[1]					
			elif (nameFile.split('.')[-1]=='tif'):
				try:
					metadata = tif.getMetadata(pathFile)
					if (len(metadata)==0):
						metadata = metadata_init
					print(metadata)
				except (KeyError, TypeError):
					messagebox.showinfo(message='No metadata found, please fill in the parameters manually')
					metadata = metadata_init
					if (it.windows_img[index].tensor_img.ndim>2):
						metadata['Axis 3 Parameters Common']['MaxSize'] = it.windows_img[index].tensor_img.shape[0]
						metadata['Axis 0 Parameters Common']['MaxSize'] = it.windows_img[index].tensor_img.shape[1]
					else:	
						metadata['Axis 0 Parameters Common']['MaxSize'] = it.windows_img[index].tensor_img.shape[0]
						metadata['Axis 3 Parameters Common']['MaxSize'] = it.windows_img[index].tensor_img.shape[1]
			else:
				metadata = metadata_init
				metadata['Axis 0 Parameters Common']['MaxSize'] = it.windows_img[index].tensor_img.shape[0]
				metadata['Axis 3 Parameters Common']['MaxSize'] = it.windows_img[index].tensor_img.shape[1]		
				
			opcPsf = it.NewWindow(it.windows_img[index].nameWindow,'300x550') #Objeto de la clase NewWindow
			
			#Creation of the psf parameters window
			if (it.windows_img[index].tensor_img.ndim==4):
				psf_winparmts(numch = it.windows_img[index].tensor_img.shape[1])
			elif (it.windows_img[index].tensor_img.ndim==3):
			
				if (istiffRGB(it.windows_img[index].tensor_img.shape)): 	#Matrix of the form (x,y,r)
					psf_winparmts()
				elif (it.windows_img[index].tensor_img.shape[0]>4): 		#Matrix of the form (z,x,y)
					psf_winparmts()
				else: 												#Matrix of the form (c,x,y)
					psf_winparmts(numch = it.windows_img[index].tensor_img.shape[0])
			else: 
				psf_winparmts()

			
			opcPsf.createLabel('PSF type: ',20,10)
			dropdownPSF = opcPsf.createCombobox(20,40)
			opcPsf.createButton('Generate psf', createpsf_event, 'bottom')
	except IndexError:
		messagebox.showinfo(message='No file has been opened')	
		
def tvd_parameters(flg=True):
	global entryWeighttvd, opcTVD, opcimg, index
	opcimg = 'TV Denoising'
	try: 
		if(len(it.windows_img)>1 and flg):
			selectFile()
		else: 
			if (len(it.windows_img)==1):
				index = -1
			name = it.windows_img[index].nameFile
			opcTVD = it.NewWindow('Total variation denoising','300x120') #Objeto de la clase NewWindow
			
			opcTVD.createLabel('File: ',20,20)
			opcTVD.createLabel('Weight TV: ',20,50)
			
			entryimgtvd = opcTVD.createEntry(name,110,20, 25,True)
			entryWeighttvd = opcTVD.createEntry('',110,50,25)
			opcTVD.createButtonXY('Start', tvd_event, 110, 80)
	except IndexError: 
		messagebox.showinfo(message='No file has been opened')	
		
def tvd_event():
	import src.imageFunctions as imf
	try: 
		w = float(entryWeighttvd.get())
		if (w>0):
			opcTVD.destroy()
			img_tensor = it.windows_img[index].tensor_img
			it.printMessage('Starting processing with weight equal to: '+str(w))
			output = imf.tensorDenoisingTV(img_tensor, w)
			output_img = it.NewWindow('TVD: '+it.windows_img[index].nameWindow+' w:'+str(w), image = True)
			it.windows_img.append(output_img)
			if(output.ndim==4):
				output_img.desplay_image(output)
			elif(output.ndim==3):
				if(imf.istiffRGB(output.shape)):
					print(output.max())
					output = imf.normalizar(output)
					output_img.placeImage(np.uint8(output))
				else:
					output_img.desplay_image(output)
			else:
				output_img.placeImage(output)
				output_img.tensor_img = output			
		else:
			messagebox.showinfo(message='Weight value equal to zero is not accepted')	
	except ValueError:
		messagebox.showinfo(message='The parameter is empty, please check')

def deconvolution_event():
	global entryIterations, dropdownImg, metadata, tensor_deconv, img_tensor, opcDeconv
	img_tensor = it.windows_img[index].tensor_img
	try:
		if(int(entryIterations.get())>0):
			i = int(entryIterations.get())
			opcDeconv.destroy()
			tensor_deconv = dv.deconvolutionMain(img_tensor,multipsf,i, it.windows_img[index].nameFile, metadata)
			deconvimg = it.NewWindow('Deconvolution '+it.windows_img[index].nameWindow+' i:'+str(i), image = True)
			it.windows_img.append(deconvimg)
			if(tensor_deconv.ndim==4):
				deconvimg.desplay_image(tensor_deconv)
			elif(tensor_deconv.ndim==3):
				import src.imageFunctions as imf
				if(imf.istiffRGB(tensor_deconv.shape)):
					deconvimg.placeImage(np.uint8(tensor_deconv))
					deconvimg.tensor_img = tensor_deconv
				else: 
					deconvimg.desplay_image(tensor_deconv)
			else:
				deconvimg.placeImage(tensor_deconv)
				deconvimg.tensor_img = tensor_deconv
		else:
			messagebox.showinfo(message='Iteration value equal to zero is not accepted')
	except (AttributeError, ValueError):
		messagebox.showinfo(message='There are empty parameters, please check')
	
def createpsf_event():
	global entryex_wavelen, entryem_wavelen
	global multipsf, opcPsf, entryIterations, dropdownPSF, metadata, entryrefr_index, opcDeconv
	
	# Extracting available metadata
	for ch in range(len(entryex_wavelen)):
		if(entryex_wavelen[ch]!=None):
			try: 
				metadata['Channel '+str(ch+1)+' Parameters']['ExcitationWavelength'] = float(entryex_wavelen[ch].get())
			except:
				print('Error: ',sys.exc_info()[0])
			
	for ch in range(len(entryem_wavelen)):
		if(entryem_wavelen[ch]!=None):
			try:
				metadata['Channel '+str(ch+1)+' Parameters']['EmissionWavelength'] = float(entryem_wavelen[ch].get())
			except:
				print('Error: ',sys.exc_info()[0])
	
	metadata['num_aperture'] = float(entrynum_aperture.get())
	metadata['pinhole_radius'] = float(entrypinhole_radius.get())
	metadata['magnification'] = float(entrymagnification.get())
	metadata['refr_index'] = float(entryrefr_index.get())
	metadata['Axis 3 Parameters Common']['EndPosition'] = float(entrydimz.get())
	metadata['Axis 0 Parameters Common']['EndPosition'] = float(entrydimr.get())
	#Quitar esta linea 
	#ValueError: could not broadcast input array from shape (2,2) into shape (512,512) - Al leer archivo .lsm
	metadata['Axis 0 Parameters Common']['MaxSize'] = it.windows_img[index].tensor_img.shape[-2]
		
	try:
		if ((metadata['num_aperture']/metadata['refr_index'])<=1.0):

			psftype = dropdownPSF.current()
			
			multipsf = cpsf.shape_psf(it.windows_img[index].tensor_img, metadata, psftype)
			opcPsf.destroy()
			
			opcDeconv = it.NewWindow('Richardson-Lucy Deconvolution','300x150') #Objeto de la clase NewWindow
			
			opcDeconv.createLabel('File: ',20,20)
			opcDeconv.createLabel('PSF: ',20,50)
			opcDeconv.createLabel('Iterations: ',20,80)
			
			entryimg = opcDeconv.createEntry(it.windows_img[index].nameFile,110,20, 25,True)
			entrypsf = opcDeconv.createEntry('psf_'+it.windows_img[index].nameWindow,110,50,25, True)
			
			entryIterations = opcDeconv.createEntry('',110,80,25)
			opcDeconv.createButtonXY('Start', deconvolution_event, 100, 110)	
		else: 
			messagebox.showinfo(message='Quotient of the numeric aperture ' +str(metadata['num_aperture'])+ ' and refractive index ' +str(metadata['refr_index'])+ ' is greater than 1.0')
	except (ZeroDivisionError, TypeError):
		messagebox.showinfo(message='Error, there are parameters that cannot be equal to zero')
	except ValueError:
		messagebox.showinfo(message='Matrix (x, y) is not the same size')	
	
def neural_network_event(flg=True):
	global tensor_deconv, opcimg, index
	from .esrgan import nn
	opcimg = 'Neural Network'
	try: 
		if(len(it.windows_img)>1 and flg):
			selectFile()
		else: 
			if (len(it.windows_img)==1):
				index = -1		
			nn(it.windows_img[index].tensor_img, index)
	except IndexError:
		messagebox.showinfo(message='There is no input parameter')
		
def resize_parameters(flg=True):
	global opcimg, index, opcResize, entryX, entryY
	opcimg = 'Reshape'
	if(len(it.windows_img)>0):
		if(len(it.windows_img)>1 and flg):
			selectFile()
		else: 
			if (len(it.windows_img)==1):
				index = -1
			name = it.windows_img[index].nameFile
			opcResize = it.NewWindow('Resize','230x140') #Objeto de la clase NewWindow
			
			opcResize.createLabel('File: ',20,20)
			opcResize.createLabel('X: ',20,50)
			opcResize.createLabel('Y: ',20,80)
			
			entryimg = opcResize.createEntry(name,80,20, 15,True)
			entryX = opcResize.createEntry('',80,50,15)
			entryY = opcResize.createEntry('',80,80,15)
			opcResize.createButton('Resize', resize_event, 'bottom')				
	else:
		messagebox.showinfo(message='There is no input parameter')	
		
def resize_event():
	import src.imageFunctions as imf
	try:
		x, y = (int(entryX.get()),int(entryY.get()))
		if(x>50 and y>50):
			opcResize.destroy()
			it.printMessage('Starting rescaled: ')
			oldSize = it.windows_img[index].tensor_img.shape[-2]
			it.windows_img[index].tensor_img = np.uint8(imf.resizeTensor(it.windows_img[index].tensor_img, x,y))
			it.windows_img[index].updatePanel(oldSize)
			it.printMessage('Completed: size ('+str(x)+','+str(y)+')')
		else: 
			messagebox.showinfo(message='Values not accepted, you must enter a minimum value of 50')	
	except ValueError:
		messagebox.showinfo(message='There is no input parameter')		
		
def selectFile():
	"""This function select a file"""
	global cmbxFile, opcSF
	if(len(it.windows_img)>0):
		opcSF = it.NewWindow('Open files','300x100')
		opcSF.createLabel('Choose a file',20,20)
		windows_img_names = it.getNamesWindows()
		cmbxFile = opcSF.createCombobox2(windows_img_names,20,50)
		opcSF.createButton('Select', selectFile_event, 'bottom')
	else: 
		messagebox.showinfo(message='No file has been opened')	
		
def selectFile_event():
	"""This function select a file"""
	global index
	index = cmbxFile.current()
	print('index: ', index)
	opcSF.destroy()
	if(opcimg=='Deconvolution'):
		psf_parameters(flg=False)
	if(opcimg=='Neural Network'):
		neural_network_event(flg=False)
	if(opcimg=='TV Denoising'):
		tvd_parameters(flg=False)
	if(opcimg=='Reshape'):
		resize_parameters(flg=False)		
		
def about_event():
	import cv2
	about_win = it.NewWindow('About IFC Microscopy') #Objeto de la clase NewWindow
	img = cv2.imread('src/icon/About.png')
	about_win.placeImage(img)
	about_win.createLabel('IFC Microscopy v0.6.25 ',115,30)
	
def close_windows_event():
	for i in range(len(it.windows_img)):
		it.windows_img[-1].on_closing()
	
def on_closing():
	import os
	if (os.path.isdir('src/cache/training_set')):
		rmtree("src/cache/training_set")
	it.mainWindow.destroy()	
		
def interface():
	#The main program window is created
	it.createWindowMain()
	#Drop-down menu is created
	menu = it.createMenu()
	#Menu options are added
	opc1 = it.createOption(menu)
	it.createCommand(opc1, "Open", it.openFile)
	it.createCommand(opc1, "Save", it.saveFile)
	it.createCommand(opc1, "Close windows", close_windows_event)
	it.createCommand(opc1, "Exit", it.mainWindow.quit)
	it.createCascade(menu, 'File', opc1)
	
	opc2 = it.createOption(menu)
	it.createCommand(opc2, "Resize", resize_parameters)
	#it.createCommand(opc2, "Zoom", mainWindow.quit)
	it.createCascade(menu, 'Edit', opc2)

	opc3 = it.createOption(menu)
	it.createCommand(opc3, "Deconvolution", psf_parameters)
	it.createCommand(opc3, "Neural Network", neural_network_event)
	it.createCommand(opc3, "TV Denoising", tvd_parameters)
	it.createCascade(menu, 'Image', opc3)
	
	opc4 = it.createOption(menu)
	it.createCommand(opc4, "About IFC Microscopy", about_event)
	it.createCascade(menu, 'Help', opc4)	

	it.statusBar = it.createStatusBar()

	it.mainWindow.protocol("WM_DELETE_WINDOW", on_closing)

	it.mainWindow.mainloop()
