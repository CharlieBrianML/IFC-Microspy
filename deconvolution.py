#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# deconvolucion.py
# Gestiona el proceso de deconvolucion
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

from time import time
from time import sleep
import interfaceTools as it
from progress.bar import Bar, ChargingBar
import imageFunctions as imf
from deconvTF import deconvolveTF
import tiff as tif
import os
import sys
import tifffile
import numpy as np

message = ''

def deconvolutionTiff(img,psf,iterations,weight):
	deconv_list=np.zeros(img.shape)
	
	if(img.ndim==3):
		for c in range(img.shape[0]):
			if(weight!=0):
				img_denoised = imf.denoisingTV(img[c,:,:], weight)
			else:
				img_denoised = img[c,:,:]
			deconv = deconvolveTF(img_denoised,psf[c,:,:],iterations) #Funcion de deconvolucion de imagenes
			deconvN = imf.normalizar(deconv) #Se normaliza la matriz 
			print('Channel ',c+1,' deconvolved')
			deconv_list[c,:,:]=deconvN
	if(img.ndim==4):
		if(imf.istiffRGB(img.shape)):
			for c in range(img.shape[0]):
				deconv= deconvolutionRGB(img[c,:,:,:],psf[c,:,:], iterations, weight) #Funcion de deconvolucion de imagenes
				print('Channel ',c+1,' deconvolved')
				deconv_list[c,:,:,:]=deconv
		else:
			deconv_list=np.zeros((img.shape[0],img.shape[1],img.shape[2],img.shape[3]), dtype="int16")
			for c in range(img.shape[0]):
				bar = Bar("\nChannel "+str(c+1)+' :', max=img.shape[0])
				for z in range(img.shape[1]):
					img_denoised = imf.denoisingTV(img[c,z,:,:], weight)
					deconv= deconvolveTF(img_denoised,psf[c,z,:,:], iterations) #Funcion de deconvolucion de imagenes
					deconvN = imf.normalizar(deconv) #Se normaliza la matriz 
					deconv_list[c,z,:,:]=deconvN
					bar.next()
				bar.finish()
	if(img.ndim==5):
		for c in range(img.shape[0]):
			for z in range(img.shape[1]):
				deconv= deconvolutionRGB(img[z,c,:,:,:],psf[z,c,:,:,:], iterations) #Funcion de deconvolucion de imagenes
				deconv_list[z,c,:,:,:]=deconv
			print('Channel ',c+1,' deconvolved')	
	return deconv_list
	
def deconvolutionRGB(img,psf,iterations,weight):
	imgG=imf.escalaGrises(img)
	img_denoised = imf.denoisingTV(imgG,weight)
	deconv=deconvolveTF(img_denoised, psf, iterations) #Funcion de deconvolucion de imagenes
	deconvN=imf.normalizar(deconv)
	deconvC=imf.elegirCanal('r',deconv)
	return deconvC
	
def deconvolution1Frame(img,psf,iterations):
	img_denoised = imf.denoisingTV(imgG,weight)
	deconv=deconvolveTF(img_denoised, psf, iterations) #Funcion de deconvolucion de imagenes
	deconvN=imf.normalizar(deconv)
	return deconvN
	

def deconvolutionMain(img_tensor,psf_tensor,i,weight, nameFile, metadata):
	global message
	to=time()
	
	# psf_tensor_aux = tif.readTiff('oib_files/psf_0005.tif')
	# for i in range(psf_tensor.shape[0]):
		# psf_tensor[i,:,:]=psf_tensor_aux[:,:,i]
	psf_tensor = imf.normalizar(psf_tensor)
	print('max psf: ',psf_tensor.max())

	path = os.path.dirname(os.path.realpath(sys.argv[0])) #Direcctorio donde se almacenara el resultado
	savepath = os.path.join(path,'deconvolutions/Deconvolution_'+nameFile.split('.')[0]+' i-'+str(i)+' w-'+str(weight)+'.tif')
	
	if(img_tensor.ndim>2):
		print(img_tensor.shape)
		print(psf_tensor.shape)
		if(imf.validatePSF(img_tensor,psf_tensor)):
			message = '\nStarting deconvolution'
			print(message)
			if(img_tensor.ndim==2):
				tiffdeconv = deconvolution1Frame(img_tensor,psf_tensor,i,weight)
			if(img_tensor.ndim==3):
				if(imf.istiffRGB(img_tensor.shape)):
					tiffdeconv = deconvolutionRGB(img_tensor,psf_tensor,i,weight)
				else:
					tiffdeconv = deconvolutionTiff(img_tensor,psf_tensor,i,weight)
			if(img_tensor.ndim==4):
				tiffdeconv = deconvolutionTiff(img_tensor,psf_tensor,i,weight)
		else:
			message = 'Wrong psf dimention, please enter a valid psf'
			print(message)
			exit()
			
		deconvolution_matrix = np.uint16(tiffdeconv)
			
		tifffile.imsave(savepath, deconvolution_matrix, imagej=True)
		message = 'Deconvolution successful, end of execution'
		print(message)
		
	else:
		if(extImage=='.jpg' or extImage=='.png' or extImage=='.bmp'):
			if(extPSF=='.jpg' or extPSF=='.png' or extPSF=='.bmp'):
				img = imf.imgReadCv2(imgpath) #Leemos la imagen a procesar 
				psf = imf.imgReadCv2(psfpath) #Leemos la psf de la imagen
				psf=imf.escalaGrises(psf)
				message = '\nFiles are supported\nStarting deconvolution'
				print(message)
				it.statusbar['text']=message
				sleep(1)
				message = "\nProcessing: "+nameFile+extImage
				it.statusbar['text']=message
				sleep(1)
				bar = Bar(message, max=1)
				print('\n')
				if(img.ndim>1):
					#warnings.filterwarnings('ignore', '.*',)
					deconv=deconvolutionRGB(img,psf,i,weight)
					bar.next()
					bar.finish()
				else:
					deconv=deconvolution1Frame(img,psf,i)
				#imf.guardarImagen(os.path.join(savepath,'\Deconvolution_'+nameFile+'.bmp'),deconv)
				imf.guardarImagen(os.getcwd()+'\Deconvolutions\Deconvolution_'+nameFile+'.bmp',deconv)
				#print(savepath,'\Deconvolution_'+nameFile+'.bmp')
				#bar.finish()
				message = 'Deconvolution successful, end of execution'
				print(message)
				it.statusbar['text']=message
				sleep(1)
		else:
			message = 'The file extension is not valid'
			print(message)
	tf=time()
	tt=tf-to
	print("Runtime: ",tt/60, "minutes")
	#it.statusbar['text']="Runtime: "+str(tt/60)+"minutes"
	sleep(1)
	message = ''
	return deconvolution_matrix