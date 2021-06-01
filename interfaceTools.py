from tkinter import *    # Carga módulo tk (widgets estándar)
from tkinter import filedialog as fd
from tkinter import ttk  # Carga ttk (para widgets nuevos 8.5+)
from PIL import ImageTk, Image
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

import cv2 
import os

# Define la ventana principal de la aplicación
mainWindow = Tk() 
#img = ''
file = ''
filesName = []
filesPath = []
statusbar = None

def openFile():
	global file
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
			from tkinter import messagebox
			messagebox.showinfo(message='file '+ file +' has been opened', title="File")
			# venImg = NewWindow(nameFile)
			# scrollbar = Scrollbar(venImg.window, orient=HORIZONTAL, takefocus=10)
			# scrollbar.pack(side="top", fill="x")
			# scrollbar.config(command=scrollImage)
			# venImg.placeImage(file)
		else:
			venImg = NewWindow(nameFile)
			#newVen = venImg.createNewWindow(file.split('/')[len(file.split('/'))-1])
			venImg.placeImage(file)
	#venImg = createNewWindow(file)
	#placeImage(venImg, file)
	
def scrollImage(*args):
	print('Estoy desplazando la imagen')
	
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
	statusbar = Label(mainWindow, text='IFC SuperResolution v0.0.1', bd=1, relief=SUNKEN, anchor=W)
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
		
	def on_closing(self):
		print('Se cerro: ', self.nameWindow)
		self.window.destroy()
		if (self.nameWindow in filesName):
			filesName.remove(self.nameWindow)
		
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
		#dropdown = ttk.Combobox(self.window, state="readonly",values = ["Python", "C", "C++", "Java"])
		dropdown = ttk.Combobox(self.window, state="readonly",values = filesName)
		dropdown.place(x=x, y=y)
		if (len(filesName)>0):
			dropdown.current(0)
		return dropdown