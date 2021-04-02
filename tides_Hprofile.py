import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib import animation
import time

g = 9.81
tau = 0
H = 10
rho0 = 1000

A = 0.5
sigma = 1.4*10**(-4)
c = np.sqrt(g*H)
lambd = 0.
cd = 0.0025

Include_Nonlinear = True

#%% ===========================================================================
#   SPATIAL DIFFERENTATION
#   ===========================================================================

def diff(S, dx):
    return (S[2:] - S[:-2]) / (2 * dx)
    
#%% ===========================================================================
#   BOUNDARY CONDITIONS
#   ===========================================================================

def BCu(Su,Szeta, dx,n):
    if Include_Nonlinear:
        return -g*(-3*Szeta[0]+4*Szeta[1]-Szeta[2])/(2*dx) - (cd * abs(Su[0]) * Su[0]/H[0]) - Su[0] * (-3*Su[0]+4*Su[1]-Su[2])/(2*dx)
    else:
        return -g*(-3*Szeta[0]+4*Szeta[1]-Szeta[2])/(2*dx)
    
def BCzeta(Su,Szeta, dx,n):
    if Include_Nonlinear:
        return -H[-1]*(3*(Szeta[-1]+H[-1])*Su[-1]-4*(Szeta[-2]+H[-2])*Su[-2]+(Szeta[-3]+H[-3])*Su[-3])/(2*dx)
    else:
        return -H[-1]*(3*Su[-1]-4*Su[-2]+Su[-3])/(2*dx)
    
#%% ===========================================================================
#   SIMULATION PARAMETERS
#   ===========================================================================

L = 200e3
a,b = 0,L #Lower and upper boundaries of the domain
Ndx = 100 #Number of grid steps
dx = (b-a)/Ndx

dt = 0.5 * dx / np.sqrt(g * H)
Ndt = 20000 #Number of time steps

x = np.linspace(a,b,Ndx)
t = np.arange(Ndt)*dt

CFL = c*dt/dx
print('Courant number =',CFL)

#%% ===========================================================================
#   BOTTOM TOPOGRAPHY
#   ===========================================================================

# Linear profile
#H = np.ones(Ndx)*H + 3*np.linspace(b,a,Ndx)/L

# Step profile
#H = np.ones(Ndx)*H
#H[0:int(Ndx/2)] = H[0:int(Ndx/2)]/2

# Arctan profile
# H = np.ones(Ndx)*H
# w = 10
# profile = 2*np.arctan(np.linspace(-w/2,w/2,Ndx))/np.pi
# H = H - 2*profile

# Gaussian profile
H = np.ones(Ndx)*H
sill_height = 7
sill_position = -3/4
sill_width = 0.1
gauss = sill_height*np.exp(-(np.linspace(-1,1,Ndx)-sill_position)**2/(2*sill_width**2))
H = H - gauss

#%% ===========================================================================
#   TIME-EVOLUTION FUNCTION
#   ===========================================================================

u_init = np.zeros(Ndx)
zeta_init = np.zeros(Ndx)

def fu(S1,S2,BCu,n): #S1 = Su, S2 = Szeta
    Fu = np.zeros(Ndx)
    if Include_Nonlinear:
        Fu[1:-1] = -g * diff(S2, dx) - cd * abs(S1[1:-1]) * S1[1:-1]/H[1:-1] - S1[1:-1] * diff(S1,dx)
    else:
        Fu[1:-1] = -g * diff(S2, dx)
    Fu[0] = BCu(S1,S2,dx,n)
    return Fu

def fzeta(S1,S2,BCzeta,n): #S1 = Szeta, S2 = Su
    Fzeta = np.zeros(Ndx)
    if Include_Nonlinear:
        Fzeta[1:-1] = -H[1:-1] * diff((S1+H)*S2, dx)
    else:
        Fzeta[1:-1] = -H[1:-1] * diff(S2, dx)
    Fzeta[-1] = BCzeta(S2,S1,dx,n)
    return Fzeta

def execute_rk4():
    Su = np.zeros([Ndt+1,Ndx])
    Szeta = np.zeros([Ndt+1,Ndx])
    Su[0] = u_init
    Szeta[0] = zeta_init
    for n in range(Ndt):
        
        Szeta[n][0] = A*np.sin(2*np.pi*sigma*t[n])
        Su[n][-1] = 0
        
        k1u = dt*fu(Su[n],Szeta[n],BCu,n)
        k1zeta = dt*fzeta(Szeta[n],Su[n],BCzeta,n)
        k2u = dt*fu(Su[n]+k1u/2,Szeta[n]+k1zeta/2,BCu,n)
        k2zeta = dt*fzeta(Szeta[n]+k1zeta/2,Su[n]+k1u/2,BCzeta,n)
        k3u = dt*fu(Su[n]+k2u/2,Szeta[n]+k2zeta/2,BCu,n)
        k3zeta = dt*fzeta(Szeta[n]+k2zeta/2,Su[n]+k2u/2,BCzeta,n)
        k4u = dt*fu(Su[n]+k3u,Szeta[n]+k3zeta,BCu,n)
        k4zeta = dt*fzeta(Szeta[n]+k3zeta,Su[n]+k3u,BCzeta,n)
        
        Su[n+1] = Su[n] + k1u/6 + k2u/3 + k3u/3 + k4u/6
        Szeta[n+1] = Szeta[n] + k1zeta/6 + k2zeta/3 + k3zeta/3 + k4zeta/6
        
    return Su, Szeta

tstart = time.time()
Su, Szeta = execute_rk4()
elapsed = time.time() - tstart
print('Run time:', elapsed,'s')

#%% ===========================================================================
#   ANIMATION
#   ===========================================================================

fac = 10

def execute_animation():

    fig = plt.figure(figsize=(6,6))

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.set_xlim(a/1000, b/1000)
    ax1.set_ylim(-2, 2)
    ax2.set_xlim(a/1000, b/1000)
    ax2.set_ylim(-2, 2)
    ax3.set_xlim(a/1000, b/1000)
    ax3.set_ylim(-15, 0)
    ax3.set_xlabel('$x$ (m)')
    ax1.set_ylabel(r'$\zeta$ (m)')
    ax2.set_ylabel(r'$u$ (m/s)')
    ax3.set_ylabel(r'depth (m)')
    Szeta_rk4, = ax1.plot([], [], '-r',markersize=6)
    Su_rk4, = ax2.plot([], [], '-b',markersize=6)
    ax3.plot(x/1000,-H)
    ax1.grid()
    ax2.grid()
    ax3.grid()

    def init():
        Szeta_rk4.set_data([], [])
        Su_rk4.set_data([], [])
        return Szeta_rk4, Su_rk4

    def animate(i):
        Szeta_rk4.set_data(x/1000, Szeta[fac*i])
        Su_rk4.set_data(x/1000, Su[fac*i])
        return Szeta_rk4, Su_rk4

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1, repeat = True)
    return anim

anim = execute_animation()

#%% ===========================================================================
#   FIGURES AND DATA EXPORT
#   ===========================================================================

plt.figure()
plt.plot(t, Szeta[:-1,0])
plt.plot(t, Szeta[:-1,-1])
plt.xlim([Ndt*dt*(19/20),Ndt*dt])
plt.show()

np.save('Su.npy',Su)
np.save('Szeta.npy',Szeta)

#%% ===========================================================================
#   FOURIER ANALYSIS
#   ===========================================================================

Fourier_position = -4
Fourier = np.fft.rfft(Szeta[:-1,Fourier_position])
Freq = np.fft.rfftfreq(Ndt,dt)
Power = abs(Fourier)**2

peaks = find_peaks(Power)
peaks = peaks[0][np.where(Power[peaks[0]] > np.max(Power)/10)]

plt.figure()
plt.plot(Freq,Power)
plt.scatter(Freq[peaks],Power[peaks])
plt.xlim([0,1e-3])
plt.show()

print('Peak frequencies:',Freq[peaks],'Hz')


