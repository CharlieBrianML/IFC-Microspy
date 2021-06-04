#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# interfaceTools.py
# Contiene todas las herramientas para la creacion de una interfaz
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

from tkinter import *    # Carga módulo tk (widgets estándar)
from tkinter import filedialog as fd
from tkinter import ttk  # Carga ttk (para widgets nuevos 8.5+)
from tkinter import messagebox
from PIL import ImageTk, Image

import cv2 
import os
import oibread as oib

# Define la ventana principal de la aplicación
mainWindow = Tk() 
psftypes = ['psf.ISOTROPIC | psf.EXCITATION','psf.ISOTROPIC | psf.EMISSION','psf.ISOTROPIC | psf.WIDEFIELD','psf.ISOTROPIC | psf.CONFOCAL',
'psf.ISOTROPIC | psf.TWOPHOTON','psf.GAUSSIAN | psf.EXCITATION','psf.GAUSSIAN | psf.EMISSION','psf.GAUSSIAN | psf.WIDEFIELD','psf.GAUSSIAN | psf.CONFOCAL',
'psf.GAUSSIAN | psf.TWOPHOTON','psf.GAUSSIAN | psf.EXCITATION | psf.PARAXIAL','psf.GAUSSIAN | psf.EMISSION | psf.PARAXIAL','psf.GAUSSIAN | psf.WIDEFIELD | psf.PARAXIAL',
'psf.GAUSSIAN | psf.CONFOCAL | psf.PARAXIAL','psf.GAUSSIAN | psf.TWOPHOTON | psf.PARAXIAL']
file = ''
filesName = []
filesPath = []
statusbar = None
tensor_img = None
panelImg = None

def openFile():
	global file, tensor_img, panelImg
	file = fd.askopenfilename(initialdir = os.getcwd(), title = 'Seleccione archivo', defaultextension = '*.*', filetypes = (('oib files','*.oib'),('tif files','*.tif')))
	if(len(file)>0):
		filesPath.append(file)
		nameFile = file.split('/')[len(file.split('/'))-1]
		if(os.path.splitext(nameFile)[1]=='.tif'):
			print('Archivo .tif')
			venImg = NewWindow(nameFile)
			scrollbar = Scrollbar(venImg.window, orient=HORIZONTAL, takefocus=10)
			scrollbar.pack(side="top", fill="x")
			scrollbar.config(command=scrollImage)
			venImg.placeImage(file)
		elif(os.path.splitext(nameFile)[1]=='.oib'):	
			print('Archivo .oib')
			tensor_img = oib.get_matrix_oib(file)
			print(tensor_img.shape)
			#from tkinter import messagebox
			#messagebox.showinfo(message='file '+ file +' has been opened', title="File")
			print(nameFile)
			
			venImg = NewWindow(nameFile)
			tensor_img = venImg.desplay_image(nameFile, tensor_img)
			
		else:
			venImg = NewWindow(nameFile)
			#newVen = venImg.createNewWindow(file.split('/')[len(file.split('/'))-1])
			venImg.placeImage(file)
	#venImg = createNewWindow(file)
	#placeImage(venImg, file)
	
def saveFile():
	global file
	savepath = fd.asksaveasfilename(initialdir = '/',title = 'Seleccione archivo', defaultextension = '.png',filetypes = (('png files','*.png'),('jpg f|iles','*.jpg'),('bmp files','*.bmp'),('tif files','*.tif')))
	cv2.imwrite(savepath,cv2.imread(file))
	
def createWindowMain():
	# Define la ventana principal de la aplicación
	#mainWindow = Tk() 
	mainWindow.geometry('500x50') # anchura x altura
	# Asigna un color de fondo a la ventana. 
	mainWindow.configure(bg = 'beige')
	# Asigna un título a la ventana
	mainWindow.title('IFC SuperResolution')
	mainWindow.resizable(width=False,height=False)
	#return mainWindow
	
#def createMenu(mainWindow):
def createMenu():
	#Barra superior
	menu = Menu(mainWindow)
	mainWindow.config(menu=menu)
	return menu
	
def createOption(menu):
	opc = Menu(menu, tearoff=0)
	return opc
	
def createCommand(opc, labelName, commandName):
	opc.add_command(label=labelName, command = commandName)
	
def createCascade(menu, labelName, option):
	menu.add_cascade(label=labelName, menu=option)
	
def createButton(text, command, side):
	ttk.Button(mainWindow, text=text, command=command).pack(side=side)
	
def createEntry(stringVar,x,y):
	entry = ttk.Entry(mainWindow, textvariable=stringVar)
	entry.place(x=x, y=y)
	return entry

def createLabel(text,x,y):
	label = Label(mainWindow, text=text, font=("Arial", 12)).place(x=x, y=y)
	
def createStringVar():
	nombre = StringVar()
	return nombre
	
def createStatusBar():
	global statusbar
	statusbar = Label(mainWindow, text='IFC SuperResolution v0.0.10', bd=1, relief=SUNKEN, anchor=W)
	statusbar.pack(side=BOTTOM, fill=X)
	return statusbar
	
class NewWindow:
	
	def __init__(self,nameWindow,size = None):
		self.nameWindow = nameWindow
		self.window = Toplevel(mainWindow)
		self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
		self.window.geometry(size) # anchura x altura
		#self.window.configure(bg = 'beige')
		self.window.resizable(width=False,height=False)
		self.window.title(self.nameWindow)
		self.img = None
		self.axisz_max = 0
		self.axisc_max = 0
		self.posz = 0
		self.posc = 0
		self.tensor_img = None
		
	def on_closing(self):
		print('Se cerro: ', self.nameWindow)
		self.window.destroy()
		if (self.nameWindow in filesName):
			filesName.remove(self.nameWindow)
			
	def destroy(self):
		self.window.destroy()
		
	def placeImage(self,file):
		#global img
		filesName.append(file.split('/')[len(file.split('/'))-1])
		#from PIL import ImageTk, Image
		#fileImage=Image.open(file)
		self.img = ImageTk.PhotoImage(Image.open(file))
		#self.img = PhotoImage(Image.open(file))
		self.panel = Label(self.window, image = self.img)
		self.panel.image = self.img
		self.panel.pack()

	def placeImageTensor(self,img):
		
		# resize image
		resized = self.resize_image_percent(img, 60)
		
		self.img = ImageTk.PhotoImage(image=Image.fromarray(resized))
		self.panel = Label(self.window, image = self.img)
		self.panel.image = self.img
		self.panel.pack()
		return self.panel
		
	def createButton(self,text, command, side):
		ttk.Button(self.window, text=text, command=command).pack(side=side)	

	def createButtonXY(self,text, command, x, y):
		ttk.Button(self.window, text=text, command=command).place(x=x,y=y)	
		
	def createLabel(self,text,x,y):
		#Label(self.window, text=text).pack(anchor=CENTER)
		label = Label(self.window, text=text, font=("Arial", 12)).place(x=x, y=y)

	def createEntry(self,stringVar,x,y, width=10,disabled=False):
		if disabled:
			entry = ttk.Entry(self.window, width=width)
			entry.insert(0,stringVar)
			entry.configure(state=DISABLED)
		else:
			entry = ttk.Entry(self.window, width=width)
		entry.insert(0, stringVar)
		entry.place(x=x, y=y)
		return entry
		
	def createCombobox(self,x,y):
		global files
		dropdown = ttk.Combobox(self.window, state="readonly",values = psftypes, width=40)
		dropdown.place(x=x, y=y)
		if (len(filesName)>0):
			dropdown.current(0)
		dropdown.current(8)
		return dropdown
		
	def scrollbarz(self, maxpos):
		self.scrollbarz = Scrollbar(self.window, orient=HORIZONTAL, command=self.scrollImagez)
		self.scrollbarz.pack(side=BOTTOM,fill=X)
		
		self.listboxz = Listbox(self.window, yscrollcommand=self.scrollbarz.set)
		for i in range(10+maxpos):
			self.listboxz.insert("end", '')
		self.listboxz.place(x=50,y=50)
		return self.scrollbarz
		
	def scrollbarc(self, maxpos, zscroll = True):
		if zscroll:
			self.scrollbarc = Scrollbar(self.window, orient=HORIZONTAL, command=self.scrollImagec)
			self.scrollbarc.pack(side=BOTTOM,fill=X)
		else:
			self.scrollbarc = Scrollbar(self.window, orient=HORIZONTAL, command=self.scrollImagecs)
			self.scrollbarc.pack(side=BOTTOM,fill=X)		
		
		self.listboxc = Listbox(self.window, yscrollcommand=self.scrollbarc.set)
		for i in range(10+maxpos):
			self.listboxc.insert("end", '')
		self.listboxc.place(x=50,y=50)
		return self.scrollbarc
		
	def createStatusBar(self):
		self.text = 'c:'+str(self.posc+1)+'/'+str(self.axisc_max)+' z:'+str(self.posz+1)+'/'+str(self.axisz_max)
		self.statusbar = Label(self.window, text=self.text, bd=1, relief=SUNKEN, anchor=W)
		self.statusbar.pack(side=TOP, fill=X)
		return self.statusbar
		
	def update_axes(self, axisc_max, axisz_max):
		self.axisz_max = axisz_max
		self.axisc_max = axisc_max
		
	def resize_image_percent(self, img, percent):
		import cv2
		import numpy as np
		import imageFunctions as imf
		width = int(img.shape[1] * percent / 100)
		height = int(img.shape[0] * percent / 100)
		dim = (width, height)	
		
		# resize image
		print(type(img[0,0]))
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
		print(resized.max())
		
		return imf.normalizar(resized)
		#return resized
		
	def scrollImagez(self, *args):
		#global panelImg
		print(args)

		if (int(args[1]) == -1 and self.posz > 0):
			self.posz = self.posz - 1
		if (int(args[1]) == 1 and self.posz < self.axisz_max-1):
			self.posz = self.posz + 1
			
		print(self.posc,self.posz)
		self.text = 'c:'+str(self.posc+1)+'/'+str(self.axisc_max)+' z:'+str(self.posz+1)+'/'+str(self.axisz_max)
		self.statusbar.configure(text = self.text)
		resized = self.resize_image_percent(tensor_img[self.posc,self.posz,:,:],60)
		self.img = ImageTk.PhotoImage(image=Image.fromarray(resized))

		self.panelImg['image'] = self.img
		self.panelImg.image = self.img
		self.scrollbarz.config(command=self.listboxz.yview(self.posz))	
		
	def scrollImagec(self, *args):
		#global panelImg
		print(args)

		if (int(args[1]) == -1 and self.posc > 0):
			self.posc = self.posc - 1
		if (int(args[1]) == 1 and self.posc < self.axisc_max-1):
			self.posc = self.posc + 1
			
		print(self.posc,self.posz)
		self.text = 'c:'+str(self.posc+1)+'/'+str(self.axisc_max)+' z:'+str(self.posz+1)+'/'+str(self.axisz_max)
		self.statusbar.configure(text = self.text)
		resized = self.resize_image_percent(tensor_img[self.posc,self.posz,:,:],60)
		self.img = ImageTk.PhotoImage(image=Image.fromarray(resized))

		self.panelImg['image'] = self.img
		self.panelImg.image = self.img			
		self.scrollbarc.config(command=self.listboxc.yview(self.posc))	
		
	def scrollImagecs(self, *args):
		#global panelImg
		print(args)

		if (int(args[1]) == -1 and self.posc > 0):
			self.posc = self.posc - 1
		if (int(args[1]) == 1 and self.posc < self.axisc_max-1):
			self.posc = self.posc + 1
			
		print(self.posc,self.posz)
		self.text = 'c:'+str(self.posc+1)+'/'+str(self.axisc_max)
		self.statusbar.configure(text = self.text)
		resized = self.resize_image_percent(self.tensor_img[self.posc,:,:],60)
		self.img = ImageTk.PhotoImage(image=Image.fromarray(resized))

		self.panelImg['image'] = self.img
		self.panelImg.image = self.img			
		self.scrollbarc.config(command=self.listboxc.yview(self.posc))		
		
	def desplay_image(self, nameFile, tensor_img):
		self.tensor_img = tensor_img
		if tensor_img.ndim == 4:
			self.update_axes(tensor_img.shape[0],tensor_img.shape[1])
			self.createStatusBar()
			scrollz = self.scrollbarz(tensor_img.shape[1]-1)
			scrollc = self.scrollbarc(tensor_img.shape[0]-1)
			self.panelImg = self.placeImageTensor(tensor_img[0,0,:,:])
		
		if tensor_img.ndim == 3:
			self.update_axes(tensor_img.shape[0],0)
			self.createStatusBar()
			scrollc = self.scrollbarc(tensor_img.shape[0]-1, zscroll=False)
			self.panelImg = self.placeImageTensor(tensor_img[0,:,:])
		return tensor_img		