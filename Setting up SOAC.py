import numpy as np
import matplotlib.pyplot as plt


totL = 100000      #total length of the domain [m]
dx = 100           #grid size [m]
nx = int(totL/dx)      #stepsize
dx = totL/nx

xaxis = np.linspace(0,totL,nx,False) #+ dx*0.5

g = 9.8                     #[ms-2]
zeta = np.zeros(nx)         #surface elevation
dzdx = np.zeros(nx)         #gradient of zeta
