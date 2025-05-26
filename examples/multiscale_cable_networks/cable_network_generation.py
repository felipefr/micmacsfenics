#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 19:31:09 2025

@author: frocha
"""

import numpy as np
import os, sys
from timeit import default_timer as timer

sys.path.append('/home/frocha/sources/pyola/')
sys.path.append('/home/frocha/sources/netfibGen/src/')

import copy
from fibresLib import *


np.random.seed(5)

net = Network()

maxit = 99999
maxnochange = 100
dJmin = 0.01
Jtol = 1.0e-10
ampP0 = 0.01
alphaP = 0.9
gammaP = 0.95 # damping max coordinate, for min and max. 0 means no damping (fixed window) and 1 means maximum damping (unit window).
ampA0 = 0.01
maxA = 1.1
minA = 0.9
alphaA = 0.9
pertPoint = 0.01 # 0.0 for regular
alphaPert = 0.5
restartPert = 4
ksmooth = 200
omegaSmooth = 0.5
timesSmooth = 4
Jtol2 = 1.0e-15 

setParamOpt = [maxit, maxnochange, dJmin, Jtol, ampP0, alphaP, gammaP, ampA0, alphaA, maxA, minA, pertPoint, alphaPert, restartPert, omegaSmooth, timesSmooth, Jtol2]

pertPoint = 0.0 # 0.0 for regular

net.asymFac = -2 # nx - ny
net.nFibPrevision = 90

net.createNetwork()
net.removeVertHoriFibers()

net.setFlagsAndConnectivity()
net.set_lfa(2,[1.00,0.0])
net.setAf(2,[0.1,0.0])


Preg1 = copy.deepcopy(net.P)
net.addPertubation(pertPoint)
net.correctPoints(0.99,0.01)

net.set_af_Lf_Vf()
net.setNormalAndAbar()

#~ print net.Abar
#~ print net.normal

AfOld = copy.deepcopy(net.Af)

colour = 'black'
writeFigNetwork(net,c= colour, figNum = 2 , filename = 'networkNotOptimised.pdf')

#~ net.optimize(setParamOpt,functionalNBCfib)
#~ net.optimize(setParamOpt,functionalNBCnormal)
net.optimize(setParamOpt,functionalNBCBoth)

#~ writeFigNetwork(net,figNum = 2, filename = 'networkOptimised.png')

#Ndof = 6
#Nsubsteps = 6
#Nparam = 29
# net.writeNetwork(Ndof,Nsubsteps,Nparam,fignum = 1,opInifile = 'Piola', opIncludeTri = 0, addAuxNodes = 2, addAuxNodeOnFibres = 1)


import meshio

# two triangles and one quad
points = net.P
cells = [
    ("line", net.ElemFib),
]

mesh = meshio.Mesh(
    net.P,
    cells,
    cell_data={"A": [net.Af + 0.1*np.random.rand(len(net.Af))]},
)
mesh.write("cable_network.vtk")
