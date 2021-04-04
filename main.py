import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal
from matplotlib import animation

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
#   TIME-EVOLUTION FUNCTION
#   ===========================================================================

def main():

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

    Su, Szeta = execute_rk4()
    
    return Su, Szeta
    
#%% ===========================================================================
#   SIMULATION PARAMETERS
#   ===========================================================================

g = 9.81
H0 = 500
rho0 = 1000

A = 0.5
sigma = 1.4*10**(-4)
c = np.sqrt(g*H0)
lambd = 0.
cd = 0.0025

Include_Nonlinear = True

L = 200e3
a,b = 0,L #Lower and upper boundaries of the domain
Ndx = 100 #Number of grid steps
dx = (b-a)/Ndx

dt = 0.05 * dx / np.sqrt(g * H0)
Ndt = 20000 #Number of time steps

x = np.linspace(a,b,Ndx)
t = np.arange(Ndt)*dt

CFL = c*dt/dx
#print('Courant number =',CFL,'\n')

#%% ===========================================================================
#   BOTTOM TOPOGRAPHY
#   ===========================================================================

# Gaussian profile
def bottom_gauss(H0,sill_height,sill_position,sill_width):
    H = np.ones(Ndx)*H0
    gauss = sill_height*np.exp(-(np.linspace(-1,1,Ndx)-sill_position)**2/(2*sill_width**2))
    return H - gauss

#%% ===========================================================================
#   SIMULATIONS
#   ===========================================================================

sill_height = np.linspace(0,460,2)
#sill_position = np.linspace(-3/4,3/4,4)
#sill_width = np.linspace(0.1,0.5,5)
Nsim = len(sill_height)

Su = np.zeros([Nsim,Ndt+1,Ndx])
Szeta = np.zeros([Nsim,Ndt+1,Ndx])

print('Simulations started\nCourant number =',CFL,'\n')

for i in range(Nsim):
    tstart = time.time()
    H = bottom_gauss(H0,sill_height[i],-3/4,0.1)
    #H = bottom_gauss(H0,5,sill_position[i],0.1)
    #H = bottom_gauss(H0,40,0,sill_width[i])
    Su[i],Szeta[i] = main()
    elapsed = time.time() - tstart
    print('Simulation %d/%d completed' % (i+1,Nsim))
    print('Run time:', elapsed,'s\n')
    
print('Executing FFT analysis...')

#%% ===========================================================================
#   FOURIER ANALYSIS
#   ===========================================================================

def FFT(S):
    Fourier = np.fft.rfft(S)
    Freq = np.fft.rfftfreq(Ndt,dt)
    Power = abs(Fourier)**2
    FFT_theo = 2.0 * np.abs(Fourier/Ndt)
    Phase = np.arctan2(Fourier.imag, Fourier.real) * 180/np.pi

    peaks = signal.find_peaks(Power)
    peaks = peaks[0][np.where(Power[peaks[0]] > np.max(Power)/10)]
    
    return Freq, Fourier, Phase, Power, FFT_theo, peaks

Amp_zeta = np.zeros([Nsim,Ndx])
Amp_u = np.zeros([Nsim,Ndx])
#Phase_zeta = np.zeros([Nsim,Ndx])

fig, ax=plt.subplots(2,1, figsize = (8,6), sharex = True)

for j in range(Nsim):
    for i in range(Ndx):
        freq_zeta, fourier_zeta, phase_zeta, power_zeta, amp_zeta, peaks_zeta = FFT(Szeta[j,:,i])
        freq_u, fourier_u, phase_u, power_u, amp_u, peaks_u = FFT(Su[j,:,i])
    
        Amp_zeta[j,i] = max(amp_zeta)
        Amp_u[j,i] = max(amp_u)
     
    plt.sca(ax[0])
    plt.plot(x/1000, Amp_zeta[j], label=round(sill_height[j],2))
    plt.ylabel("$|\zeta$| (m)")  #elevation amplitude
    plt.xlim([x[0]/1000,x[-1]/1000])
    plt.sca(ax[1])
    plt.plot(x/1000, Amp_u[j])
    plt.ylabel("$u$ (m/s)")
    plt.xlabel("$x$ (km)")
    plt.xlim([x[0]/1000,x[-1]/1000])
ax[0].legend()
plt.tight_layout() 

plt.figure()
plt.axhline(y=0, color='r', linestyle='--')
plt.plot(x/1000,-H)
plt.xlim([x[0]/1000,x[-1]/1000])
plt.show()

#%% ===========================================================================
#   ANIMATION
#   ===========================================================================

fac = 10

def execute_animation(Nsim):

    fig = plt.figure(figsize=(6,6))

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.set_xlim(a/1000, b/1000)
    ax1.set_ylim(-2, 2)
    ax2.set_xlim(a/1000, b/1000)
    ax2.set_ylim(-0.15, 0.15)
    ax3.set_xlim(a/1000, b/1000)
    ax3.set_ylim(-H0, 0)
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
        Szeta_rk4.set_data(x/1000, Szeta[Nsim-1,fac*i])
        Su_rk4.set_data(x/1000, Su[Nsim-1,fac*i])
        return Szeta_rk4, Su_rk4

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1, repeat = True)
    return anim

anim = execute_animation(2)

