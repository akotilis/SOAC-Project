import numpy as np
import matplotlib.pyplot as plt

# Gaussian profile
def bottom_gauss(H0,sill_height,sill_position,sill_width):
    H = np.ones(Ndx)*H0
    gauss = sill_height*np.exp(-(np.linspace(-1,1,Ndx)-sill_position)**2/(2*sill_width**2))
    return H - gauss

L = 200e3
a,b = 0,L #Lower and upper boundaries of the domain
Ndx = 200 #Number of grid steps
dx = (b-a)/Ndx

x = np.linspace(a,b,Ndx)

H = bottom_gauss(500,450,-1/2,0.1)

plt.figure()
plt.plot(x/1000,-H,'k')
plt.axhline(y=0, color='r', linestyle='--')
plt.fill(x/1000,-H)
plt.xlim([x[0]/1000,x[-1]/1000])
plt.ylim([-500,10])
plt.xlabel(r'$x$ (km)')
plt.ylabel('depth (m)')
plt.grid()
plt.show()
plt.savefig('depth_profile.png',dpi=300)