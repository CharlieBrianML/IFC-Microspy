#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# interfaceTools.py
# Contains all the tools for creating an interface
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

from locale import normalize
from tkinter import *    # Carga módulo tk (widgets estándar)
from tkinter import filedialog as fd
from tkinter import ttk  # Carga ttk (para widgets nuevos 8.5+)
from tkinter import messagebox
from tkinter.font import Font
from PIL import ImageTk, Image

import cv2 
import os
import src.oibread as oib
import src.tiff as tif
import src.lsmread as lsm
from .imageFunctions import istiffRGB

# Define la ventana principal de la aplicación
mainWindow = Tk() 
psftypes = ['psf.ISOTROPIC | psf.EXCITATION','psf.ISOTROPIC | psf.EMISSION','psf.ISOTROPIC | psf.WIDEFIELD','psf.ISOTROPIC | psf.CONFOCAL',
'psf.ISOTROPIC | psf.TWOPHOTON','psf.GAUSSIAN | psf.EXCITATION','psf.GAUSSIAN | psf.EMISSION','psf.GAUSSIAN | psf.WIDEFIELD','psf.GAUSSIAN | psf.CONFOCAL',
'psf.GAUSSIAN | psf.TWOPHOTON','psf.GAUSSIAN | psf.EXCITATION | psf.PARAXIAL','psf.GAUSSIAN | psf.EMISSION | psf.PARAXIAL','psf.GAUSSIAN | psf.WIDEFIELD | psf.PARAXIAL',
'psf.GAUSSIAN | psf.CONFOCAL | psf.PARAXIAL','psf.GAUSSIAN | psf.TWOPHOTON | psf.PARAXIAL']
file = ''
filesName = []
filesPath = []
statusBar = None
tensor_img = None
panelImg = None
cmbxFile, opcSF = (None, None)
windows_img = []
infoFile = {}
currentDir = os.getcwd()

def openFile():
	"""This function open files of type .oib .tif and .bmp"""
	global file, tensor_img, panelImg, currentDir
	filepath = fd.askopenfilename(initialdir = currentDir, title = 'Select a file', defaultextension = '*.*', filetypes = (('oib files','*.oib'),('lsm files','*.lsm'),('tif files','*.tif'),('bmp files','*.bmp'),('png files','*.png'),('jpg files','*.jpg')))
	currentDir = filepath
	if(len(filepath)>0):
		try:
			import src.imageFunctions as imf
			nameFile = filepath.split('/')[-1]
			if(os.path.splitext(nameFile)[1]=='.tif'):
				metadata = tif.getMetadata(filepath)
				if(not(istiffRGB(metadata['tensor'].shape))):
					print('File: ', nameFile)
					print('Shape: ', metadata['tensor'].shape)
					
					if(metadata['tensor'].ndim==4 or metadata['tensor'].ndim==3):
						venImg = NewWindow(filepath, metadata = metadata, image = True)
						venImg.desplay_image(metadata['tensor'])
						windows_img.append(venImg)	
					else:
						venImg = NewWindow(filepath, metadata = metadata, image = True)
						venImg.placeImage(metadata['tensor'])
						venImg.tensor_img = metadata['tensor']
						windows_img.append(venImg)
			elif(os.path.splitext(nameFile)[1]=='.lsm'):
				metadata = lsm.getMetadata(filepath)
				if(not(istiffRGB(metadata['tensor'].shape))):
					print('File: ', nameFile)
					print('Shape: ', metadata['tensor'].shape)
					
					if(metadata['tensor'].ndim==4 or metadata['tensor'].ndim==3):
						venImg = NewWindow(filepath, metadata = metadata, image = True)
						venImg.desplay_image(metadata['tensor'])
						windows_img.append(venImg)	
					else:
						venImg = NewWindow(filepath, metadata = metadata, image = True)
						venImg.placeImage(metadata['tensor'])
						venImg.tensor_img = metadata['tensor']
						windows_img.append(venImg)				
			
			elif(os.path.splitext(nameFile)[1]=='.oib'):
				metadata = oib.getMetadata(filepath)
				print('File: ', nameFile)
				print('Shape: ', metadata['tensor'].shape)
				
				venImg = NewWindow(filepath, metadata = metadata,image = True)
				venImg.desplay_image(metadata['tensor'])
				windows_img.append(venImg)
			else:
				import cv2
				print('File: ', nameFile)
				metadata = imf.getMetadataImg(filepath)
				venImg = NewWindow(filepath, metadata=metadata, image = True)
				venImg.placeImage(metadata['tensor'])
				venImg.tensor_img = metadata['tensor']
				windows_img.append(venImg)
		except IndexError:
			messagebox.showinfo(message='Format not supported')			
	
def saveFile():
	"""This function save files of type .oib .tif and .bmp"""
	global cmbxFile, opcSF
	if(len(windows_img)>0):
		opcSF = NewWindow('Save File','300x100')
		opcSF.createLabel('What image do you want to save?',20,20)
		windows_img_names = getNamesWindows()
		cmbxFile = opcSF.createCombobox2(windows_img_names,20,50)
		opcSF.createButton('Save', saveFileEvent, 'bottom')
	else: 
		messagebox.showinfo(message='No file has been opened')
	
def saveFileEvent():
	global cmbxFile, opcSF, currentDir
	import tifffile
	import os
	import numpy as np
	from src.imageFunctions import istiffRGB
	selected_file = cmbxFile.current()
	image = windows_img[selected_file].tensor_img
	namewin = windows_img[selected_file].nameWindow
	
	try:
		if(image.ndim==4):
			savepath = fd.asksaveasfilename(initialdir = currentDir,title = 'Select a file', defaultextension = '.tif', initialfile = namewin, filetypes = (('tif files','*.tif'),))
			currentDir = filepath
			if (savepath!=''):
				tifffile.imsave(savepath, np.uint16(image*(65535/image.max())), imagej=True)
				printMessage('Saved file: '+savepath)
		if(image.ndim==3):
			savepath = fd.asksaveasfilename(initialdir = currentDir,title = 'Select a file', defaultextension = '.tif', initialfile = namewin, filetypes = (('tif files','*.tif'),('png files','*.png'),('jpg files','*.jpg'),('bmp files','*.bmp')))
			currentDir = filepath
			if(not(istiffRGB(image.shape))):			
				tifffile.imsave(savepath, np.uint16(image*(65535/image.max())), imagej=True)
				printMessage('Saved file: '+savepath)
			else: 
				cv2.imwrite(savepath, image)
				printMessage('Saved file: '+savepath)	
		if(image.ndim==2):	
			savepath = fd.asksaveasfilename(initialdir = currentDir,title = 'Select a file', defaultextension = '.png', initialfile = namewin, filetypes = (('png files','*.png'),('jpg files','*.jpg'),('bmp files','*.bmp')))
			currentDir = filepath
			cv2.imwrite(savepath, image)
			printMessage('Saved file: '+savepath)
		opcSF.destroy()	
	except:
		messagebox.showinfo(message='Error when trying to save the file, try again')
		print("Error: ", sys.exc_info()[0])
		
def printMessage(message):
	print(message)
	statusBar.configure(text = message)

def getNamesWindows():
	names = []
	for window_object in windows_img:
		names.append(window_object.nameWindow)
	return names
	
def getFormatTime(time):
	print(time)
	minutes = int(time/60)
	seconds = int(time%60)
	return (minutes, seconds)
	
def createWindowMain():
	"""Definition of the main window"""
	# Define la ventana principal de la aplicación
	#mainWindow = Tk() 
	mainWindow.geometry('500x50') # anchura x altura
	# Asigna un color de fondo a la ventana. 
	mainWindow.configure(bg = 'beige')
	# Asigna un título a la ventana
	mainWindow.title('IFC Microspy')
	#mainWindow.iconbitmap('icon/ifc.ico')
	mainWindow.tk.call('wm', 'iconphoto', mainWindow._w, PhotoImage(file='src/icon/ifc.png'))
	mainWindow.resizable(width=False,height=False)
	#return mainWindow
	
#def createMenu(mainWindow):
def createMenu():
	"""This function creates a menu"""
	#Barra superior
	menu = Menu(mainWindow)
	mainWindow.config(menu=menu)
	return menu
	
def createOption(menu):
	"""This function creates a menu option"""
	opc = Menu(menu, tearoff=0)
	return opc
	
def createCommand(opc, labelName, commandName):
	"""This function creates a command"""
	opc.add_command(label=labelName, command = commandName)
	
def createCascade(menu, labelName, option):
	"""This function creates a tab main"""
	menu.add_cascade(label=labelName, menu=option)
	
def createButton(text, command, side):
	"""This function creates a button"""
	ttk.Button(mainWindow, text=text, command=command).pack(side=side)
	
def createEntry(stringVar,x,y):
	"""This function creates a entry"""
	entry = ttk.Entry(mainWindow, textvariable=stringVar)
	entry.place(x=x, y=y)
	return entry

def createLabel(text,x,y, family = "Helvetica", size = 11, weight = "normal", slant = "roman", underline=0):
	"""This function creates a label"""
	font = Font(family = family,size = size,weight = weight)
	label = Label(mainWindow, text=text, font=font).place(x=x, y=y)	
	
def createStringVar():
	"""This function creates a StringVar"""
	nombre = StringVar()
	return nombre
	
def createStatusBar():
	"""This function creates a status bar"""
	global statusbar
	v = os.popen('git tag').read().split('\n')
	statusbar = Label(mainWindow, text='IFC Microspy '+v[0], bd=1, relief=SUNKEN, anchor=W)
	statusbar.pack(side=BOTTOM, fill=X)
	return statusbar
	
class NewWindow:
	"""This class contains the functions to define a window"""
	
	def __init__(self,nameWindow,size = None, metadata = None, image = False):
		if image:
			self.metadata = metadata
			self.nameFile = nameWindow.split('/')[-1]
			self.nameWindow = self.nameFile.split('.')[0]
			self.path = nameWindow
			self.text = '\t'
			self.img, self.tensor_img = (None, None)
			self.inx0, self.inx1, self.inx2, self.inx3 = (1, 1, 1, 1)
			self.inx0p, self.inx1p, self.inx2p, self.inx3p = (None, None, None, None) #previous index
			self.posz, self.posc, self.post, self.percent = (0, 0, 0, 100)
			self.normalize = False
		else: 
			self.nameWindow = nameWindow
			self.nameFile, self.path = (None, None)
			
		self.window = Toplevel(mainWindow)
		self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
		self.window.geometry(size) # (width, height)
		#self.window.configure(bg = 'beige')
		self.window.tk.call('wm', 'iconphoto', self.window._w, PhotoImage(file='src/icon/ifc.png'))
		self.window.resizable(width=False,height=False)
		self.window.title(self.nameWindow)
				
		
	def on_closing(self):
		print('Closed: ', self.nameWindow)
		if (self.nameWindow in filesName):
			filesName.remove(self.nameWindow)
		if (self in windows_img):
			windows_img.remove(self)
		self.window.destroy()	
			
	def destroy(self):
		self.window.destroy()
		
	def placeImage(self,img):
		self.tensor_img = img
		self.validateSize()
		self.createStatusBar()
		self.createCheakButton('normalize',275,0)
		resized = self.resize_image_percent(img, self.percent)
		imageTk = ImageTk.PhotoImage(image=Image.fromarray(resized))
		self.panelImg = Label(self.window, image = imageTk)
		self.panelImg.image = imageTk
		self.panelImg.pack()
		ttk.Button(self.window, text='+', command=self.increase_size).place(x=0,y=0,width=20,height=20)
		ttk.Button(self.window, text='-', command=self.decrease_size).place(x=25,y=0,width=20,height=20)

	def placeImageTensor(self,img):

		# resize image
		resized = self.resize_image_percent(img, self.percent)
		
		imageTk = ImageTk.PhotoImage(image=Image.fromarray(resized))
		self.panel = Label(self.window, image = imageTk)
		self.panel.image = imageTk
		self.panel.pack()
		return self.panel

	def placeImageAbout(self,img):
		
		imageTk = ImageTk.PhotoImage(image=Image.fromarray(img))
		self.panel = Label(self.window, image = imageTk)
		self.panel.image = imageTk
		self.panel.pack()
		return self.panel		
		
	def createButton(self,text, command, side):
		ttk.Button(self.window, text=text, command=command).pack(side=side)	

	def createButtonXY(self,text, command, x, y):
		ttk.Button(self.window, text=text, command=command).place(x=x,y=y)	
		
	def createLabel(self,text,x,y, family = "Helvetica", size = 11, weight = "normal", slant = "roman", underline=0):
		font = Font(family = family,size = size,weight = weight)
		label = Label(self.window, text=text, font=font).place(x=x, y=y)

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
		dropdown.current(13)
		return dropdown		
		
	def createCombobox2(self,values,x,y):
		dropdown = ttk.Combobox(self.window, state="readonly",values = values, width=40)
		dropdown.place(x=x, y=y)
		if (len(values)>0):
			dropdown.current(0)
		return dropdown

	def createCheakButton(self, text, x, y):
		self.checkvar = IntVar()
		Checkbutton(self.window, text=text, command=self.normalizeImgPanel, variable=self.checkvar, onvalue=1, offvalue=0, height=1, width=6).place(x=x, y=y)
		
	def normalizeImgPanel(self):
		if(self.checkvar.get()):
			print('Normalizeing ',self.nameFile)
			self.normalize = True
			self.updatePanel()
		else:
			self.normalize = False
			self.updatePanel()

	def scrollbarz(self, maxpos):
		self.scrollbarz = Scrollbar(self.window, orient=HORIZONTAL, command=self.scrollImagez)
		self.scrollbarz.pack(side=BOTTOM,fill=X)
		
		self.listboxz = Listbox(self.window, yscrollcommand=self.scrollbarz.set)
		for i in range(10+maxpos):
			self.listboxz.insert("end", '')
		self.listboxz.place(x=50,y=50)
		return self.scrollbarz
		
	def scrollbart(self, maxpos):
		self.scrollbart = Scrollbar(self.window, orient=HORIZONTAL, command=self.scrollImaget)
		self.scrollbart.pack(side=BOTTOM,fill=X)
		
		self.listboxt = Listbox(self.window, yscrollcommand=self.scrollbart.set)
		for i in range(10+maxpos):
			self.listboxt.insert("end", '')
		self.listboxt.place(x=50,y=50)
		return self.scrollbart			
		
	def scrollbarc(self, maxpos):
		self.scrollbarc = Scrollbar(self.window, orient=HORIZONTAL, command=self.scrollImagec)
		self.scrollbarc.pack(side=BOTTOM,fill=X)	
		
		self.listboxc = Listbox(self.window, yscrollcommand=self.scrollbarc.set)
		for i in range(10+maxpos):
			self.listboxc.insert("end", '')
		self.listboxc.place(x=50,y=50)
		return self.scrollbarc
		
	def createStatusBar(self):
		self.text = self.text + '  (' +str(self.metadata['X'])+ 'x' +str(self.metadata['Y'])+ ')' +' '+str(self.percent)+'%' + '  type: '+ str(self.metadata['type'])
		self.statusbar = Label(self.window, text=self.text, bd=1, relief=SUNKEN, anchor=W)
		self.statusbar.pack(side=TOP, fill=X)
		return self.statusbar
		
	def update_axes(self):
		if ('channels' in self.metadata):
			self.scrollbarc(self.metadata['channels']['value']-1)
			self.text = self.text + ' c:'+str(self.posc+1)+'/'+str(self.metadata['channels']['value'])
		if ('frames' in self.metadata):
			self.scrollbart(self.metadata['frames']['value']-1)
			self.text = self.text + ' t:'+str(self.post+1)+'/'+str(self.metadata['frames']['value'])
		if ('slices' in self.metadata):
			self.scrollbarz(self.metadata['slices']['value']-1)
			self.text = self.text + ' z:'+str(self.posz+1)+'/'+str(self.metadata['slices']['value'])
			
	def update_text(self):
		self.text = '\t'
		if ('channels' in self.metadata):
			self.text = self.text + ' c:'+str(self.posc+1)+'/'+str(self.metadata['channels']['value'])
		if ('frames' in self.metadata):
			self.text = self.text + ' t:'+str(self.post+1)+'/'+str(self.metadata['frames']['value'])
		if ('slices' in self.metadata):
			self.text = self.text + ' z:'+str(self.posz+1)+'/'+str(self.metadata['slices']['value'])
		self.text = self.text + '  (' +str(self.metadata['X'])+ 'x' +str(self.metadata['Y'])+ ')' +' '+str(self.percent)+'%' + '  type: '+ str(self.metadata['type'])
		
	def resize_image_percent(self, img, percent):
		import cv2
		import numpy as np
		import src.imageFunctions as imf
		width = int(img.shape[1] * percent / 100)
		height = int(img.shape[0] * percent / 100)
		dim = (width, height)	

		if self.normalize:
			img = imf.normalizeImg(img, self.metadata)

		# resize image
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)

		if (resized.dtype=='uint16'):
			resized	= resized*(255/4095)
		return resized
		
	def scrollImagez(self, *args):
		if ('-1' in args and self.posz > 0):
			self.posz = self.posz - 1
		
		if ('1' in args and self.posz < self.metadata['slices']['value']-1):
			self.posz = self.posz + 1

		self.update_text()
		self.statusbar.configure(text = self.text)
		self.updatePositionS(self.posz)
		print('New Posaxis-z: ',self.posz+1)
		if (self.tensor_img.ndim==4):
			resized = self.resize_image_percent(self.tensor_img[self.inx0p:self.inx0,self.inx1p:self.inx1,self.inx2p:self.inx2,self.inx3p:self.inx3].reshape((self.metadata['X'],self.metadata['Y'])), self.percent)
		if (self.tensor_img.ndim==3):
			resized = self.resize_image_percent(self.tensor_img[self.inx0p:self.inx0,self.inx1p:self.inx1,self.inx2p:self.inx2].reshape((self.metadata['X'],self.metadata['Y'])), self.percent)
		imageTk = ImageTk.PhotoImage(image=Image.fromarray(resized))

		self.panelImg['image'] = imageTk
		self.panelImg.image = imageTk
		self.scrollbarz.config(command=self.listboxz.yview(self.posz))
		
	def scrollImaget(self, *args):
		if ('-1' in args and self.post > 0):
			self.post = self.post - 1
		
		if ('1' in args and self.post < self.metadata['frames']['value']-1):
			self.post = self.post + 1

		self.update_text()
		self.statusbar.configure(text = self.text)
		self.updatePositionF(self.post)
		print('New Posaxis-t: ',self.post+1)
		if (self.tensor_img.ndim==4):
			resized = self.resize_image_percent(self.tensor_img[self.inx0p:self.inx0,self.inx1p:self.inx1,self.inx2p:self.inx2,self.inx3p:self.inx3].reshape((self.metadata['X'],self.metadata['Y'])), self.percent)
		if (self.tensor_img.ndim==3):
			resized = self.resize_image_percent(self.tensor_img[self.inx0p:self.inx0,self.inx1p:self.inx1,self.inx2p:self.inx2].reshape((self.metadata['X'],self.metadata['Y'])), self.percent)
		imageTk = ImageTk.PhotoImage(image=Image.fromarray(resized))

		self.panelImg['image'] = imageTk
		self.panelImg.image = imageTk
		self.scrollbart.config(command=self.listboxt.yview(self.post))
		
	def scrollImagec(self, *args):
		if ('-1' in args and self.posc > 0):
			self.posc = self.posc - 1
		
		if ('1' in args and self.posc < self.metadata['channels']['value']-1):
			self.posc = self.posc + 1

		self.update_text()
		self.statusbar.configure(text = self.text)
		self.updatePositionC(self.posc)
		print('New Posaxis-c: ',self.posc+1)
		if (self.tensor_img.ndim==4):
			resized = self.resize_image_percent(self.tensor_img[self.inx0p:self.inx0,self.inx1p:self.inx1,self.inx2p:self.inx2,self.inx3p:self.inx3].reshape((self.metadata['X'],self.metadata['Y'])), self.percent)
		if (self.tensor_img.ndim==3):
			resized = self.resize_image_percent(self.tensor_img[self.inx0p:self.inx0,self.inx1p:self.inx1,self.inx2p:self.inx2].reshape((self.metadata['X'],self.metadata['Y'])), self.percent)
		imageTk = ImageTk.PhotoImage(image=Image.fromarray(resized))

		self.panelImg['image'] = imageTk
		self.panelImg.image = imageTk	
		self.scrollbarc.config(command=self.listboxc.yview(self.posc))
		
	def desplay_image(self, tensor_img):
		self.tensor_img = tensor_img
		if tensor_img.ndim == 4:
			self.validateSize()
			self.update_axes()
			self.createStatusBar()
			self.updateIndex()
			self.createCheakButton('normalize',275,0)
			self.panelImg = self.placeImageTensor( tensor_img[None:self.inx0,None:self.inx1,None:self.inx2,None:self.inx3].reshape((self.metadata['X'],self.metadata['Y'])) )
			ttk.Button(self.window, text='+', command=self.increase_size).place(x=0,y=0,width=20,height=20)
			ttk.Button(self.window, text='-', command=self.decrease_size).place(x=25,y=0,width=20,height=20)
		
		if tensor_img.ndim == 3:
			self.validateSize()
			self.update_axes()
			self.createStatusBar()
			self.updateIndex()
			self.createCheakButton('normalize',275,0)		
			self.panelImg = self.placeImageTensor( tensor_img[None:self.inx0,None:self.inx1,None:self.inx2].reshape((self.metadata['X'],self.metadata['Y'])) )
			ttk.Button(self.window, text='+', command=self.increase_size).place(x=0,y=0,width=20,height=20)
			ttk.Button(self.window, text='-', command=self.decrease_size).place(x=25,y=0,width=20,height=20)

	def updatePositionC(self, position):
		if (self.metadata['channels']['index']==0):
			self.inx0 = position+1
			self.inx0p = position
		if (self.metadata['channels']['index']==1):
			self.inx1 = position+1
			self.inx1p = position
		if (self.metadata['channels']['index']==2):
			self.inx2 = position+1
			self.inx2p = position
		if (self.metadata['channels']['index']==3):
			self.inx3 = position+1
			self.inx3p = position
			
	def updatePositionF(self, position):
		if (self.metadata['frames']['index']==0):
			self.inx0 = position+1
			self.inx0p = position
		if (self.metadata['frames']['index']==1):
			self.inx1 = position+1
			self.inx1p = position
		if (self.metadata['frames']['index']==2):
			self.inx2 = position+1
			self.inx2p = position
		if (self.metadata['frames']['index']==3):
			self.inx3 = position+1
			self.inx3p = position
			
	def updatePositionS(self, position):
		if (self.metadata['slices']['index']==0):
			self.inx0 = position+1
			self.inx0p = position
		if (self.metadata['slices']['index']==1):
			self.inx1 = position+1
			self.inx1p = position
		if (self.metadata['slices']['index']==2):
			self.inx2 = position+1
			self.inx2p = position
		if (self.metadata['slices']['index']==3):
			self.inx3 = position+1
			self.inx3p = position			
			
	def updateIndex(self):
		indexX = self.metadata['tensor'].shape.index(self.metadata['X'])
		indexY = self.metadata['tensor'].shape.index(self.metadata['Y'], indexX+1)
		if (indexX==0):
			self.inx0 = None
		if (indexX==1):
			self.inx1 = None
		if (indexX==2):
			self.inx2 = None
		if (indexX==3):
			self.inx3 = None
		if (indexY==0):
			self.inx0 = None
		if (indexY==1):
			self.inx1 = None
		if (indexY==2):
			self.inx2 = None
		if (indexY==3):
			self.inx3 = None			
			
	def resize(self):
		self.update_text()
		self.statusbar.configure(text = self.text)
		if(self.tensor_img.ndim==2):
			resized = self.resize_image_percent(self.tensor_img,self.percent)	
		if(self.tensor_img.ndim==3):

			if(istiffRGB(self.tensor_img.shape)):
				resized = self.resize_image_percent(self.tensor_img,self.percent)
			else:
				resized = self.resize_image_percent(self.tensor_img[self.inx0p:self.inx0,self.inx1p:self.inx1,self.inx2p:self.inx2].reshape((self.metadata['X'],self.metadata['Y'])), self.percent)
		if(self.tensor_img.ndim==4):
			resized = self.resize_image_percent(self.tensor_img[self.inx0p:self.inx0,self.inx1p:self.inx1,self.inx2p:self.inx2,self.inx3p:self.inx3].reshape((self.metadata['X'],self.metadata['Y'])), self.percent)		
		return resized
		
	def increase_size(self):
		self.percent = self.percent+10
		print('Percent',self.percent)
		resized = self.resize()
		
		imageTk = ImageTk.PhotoImage(image=Image.fromarray(resized))
		self.panelImg['image'] = imageTk
		self.panelImg.image = imageTk
		
	def decrease_size(self):
		self.percent = self.percent-10
		print('Percent',self.percent)
		resized = self.resize()
		
		imageTk = ImageTk.PhotoImage(image=Image.fromarray(resized))
		self.panelImg['image'] = imageTk
		self.panelImg.image = imageTk
		
	def updatePanel(self, oldSize=0, new_percent=False):
		import src.imageFunctions as imf
		if new_percent:
			update_percent = int(self.metadata['Y']*(100/oldSize))
			print('New percent: ', update_percent)
			self.percent = update_percent
			self.update_text()
			self.statusbar.configure(text = self.text)
		else:
			update_percent=self.percent
		
		if(self.tensor_img.ndim==2):		
			imageTk = ImageTk.PhotoImage(image=Image.fromarray( self.resize_image_percent(self.tensor_img,update_percent) ))	
		if(self.tensor_img.ndim==3):
			if(istiffRGB(self.tensor_img.shape)):		
				imageTk = ImageTk.PhotoImage(image=Image.fromarray( self.resize_image_percent(self.tensor_img,update_percent) ))
			else:	
				imageTk = ImageTk.PhotoImage(image=Image.fromarray( self.resize_image_percent(self.tensor_img[self.inx0p:self.inx0,self.inx1p:self.inx1,self.inx2p:self.inx2].reshape((self.metadata['X'],self.metadata['Y'])),update_percent) ))
		if(self.tensor_img.ndim==4):
			imageTk = ImageTk.PhotoImage(image=Image.fromarray( self.resize_image_percent(self.tensor_img[self.inx0p:self.inx0,self.inx1p:self.inx1,self.inx2p:self.inx2,self.inx3p:self.inx3].reshape((self.metadata['X'],self.metadata['Y'])), update_percent) ))
				
		self.panelImg['image'] = imageTk
		self.panelImg.image = imageTk
		
	def getResolution(self):
		import platform
		size = (None, None)
		sistem = platform.system()
		if(sistem=='Windows'):
			import ctypes
			user32 = ctypes.windll.user32
			user32.SetProcessDPIAware()
			size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
		if(sistem=='Linux'):
			args = ["xrandr", "-q", "-d", ":0"]
			proc = subprocess.Popen(args,stdout=subprocess.PIPE)
			for line in proc.stdout:
				if isinstance(line, bytes):
					line = line.decode("utf-8")
					if "Screen" in line:
						size = (int(line.split()[7]),  int(line.split()[9][:-1]))
		print('Screen resolution: ',size)
		return size
	
	def validateSize(self):
		screen_resolution = self.getResolution()
		if(screen_resolution[1]<self.metadata['Y']):
			screen70percent = 70*(screen_resolution[1]/100)
			self.percent= int(screen70percent*(100/self.metadata['Y']))
