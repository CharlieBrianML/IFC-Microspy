#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ## ###############################################
#
# deconvTF.py
# Deconvoluciona una imagen haciendo uso de tensorflow
#
# Autor: Charlie Brian Monterrubio Lopez
# License: MIT
#
# ## ###############################################

from flowdec import data as fd_data
from flowdec import restoration as fd_restoration

# Run the deconvolution process and note that deconvolution initialization is best kept separate from 
# execution since the "initialize" operation corresponds to creating a TensorFlow graph, which is a 
# relatively expensive operation and should not be repeated across multiple executions

def deconvolveTF(img,kernel,iteration):
	imgInit = fd_restoration.RichardsonLucyDeconvolver(img.ndim).initialize()
	deconv = imgInit.run(fd_data.Acquisition(data=img, kernel=kernel), niter=iteration).data
	return deconv