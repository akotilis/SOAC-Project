import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import fftpack, signal
import scipy
import pandas as pd

                                           


def freq(sample, sample_rate, frequency):
  
    
    N = len(sample) #number of samples
    period = 1.0/frequency
    t = np.arange(N)*sample_rate #time domain
    
   
    FFT = np.fft.fft(sample)   #FFT on the array
    freqs = np.fft.fftfreq(sample.size)   #, sample_rate)   #Getting the related frequcncies
    mask = freqs>0
    
    FFT_theo = 2.0 * np.abs(FFT/N)  #FFT theoritical(amplitude)
    phase = np.arctan2(FFT.imag, FFT.real) * 180/np.pi   #phase
    L = np.arange(0, np.floor(N/2), dtype = 'int')           #Create the first half of the axis
    
    # # Create plots
    # fig, ax = plt.subplots(3,1, figsize=(8,6))  
    
    # plt.sca(ax[0])
    # plt.plot(t, sample)
    # plt.xlim(t[0],t[-1])
    # plt.ylim(min(sample)-0.5, max(sample)+0.5)
    # plt.ylabel("Signal")
    
    # plt.sca(ax[1])
    # plt.plot(freqs[mask], FFT_theo[mask])
    # plt.xlim(freqs[L[0]], freqs[L[-1]])
    # plt.ylim(FFT_theo[mask[0]], FFT_theo[mask[-1]])
    # plt.ylabel("Amplitude")
    
    # plt.sca(ax[2])
    # plt.plot(t, phase)
    # plt.xlim(t[0], t[L[-1]])
    # plt.ylim(-200, 200)
    # plt.ylabel("Phase")
    # plt.show()
    
    return freqs, FFT, phase, FFT_theo



#============== NPY files ================
Szeta = np.load('Szeta.npy')
Su = np.load('Su.npy')
w = 2*np.pi /44700    

L = 100000
a,b = 0,L #Lower and upper boundaries of the domain
Ndx = 50 #Number of grid steps
dx = (b-a)/Ndx
x = np.linspace(a,b,Ndx)


amp_z = []
amp_vel = []

for i in range(0,50):
    freq_zeta, FFT_zeta, phase_zeta, amp_zeta = freq(Szeta[:,i], 1, w)
    freq_u, FFT_u, phase_u, amp_u = freq(Su[:,i], 1, w)
    
    amp_z.append(max(amp_zeta))
    amp_vel.append(max(amp_u))
    
    
fig, ax=plt.subplots(2,1, figsize = (8,6), sharex = True)
plt.sca(ax[0])
plt.plot(x/L, amp_z)
plt.ylabel("z")
plt.sca(ax[1])
plt.plot(x/L, amp_vel)
plt.ylabel("u")
plt.xlabel("x[m]")
plt.tight_layout() 


