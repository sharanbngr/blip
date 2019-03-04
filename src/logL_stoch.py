from __future__ import division
import numpy as np
import os, pdb
#import matplotlib.pyplot as plt

def gwPSD(Omega0, alpha,freqs, f0, R1, R2, R3):

    '''
    Script to calcualte the GW power in a detector for a given powerlaw
    The TDI channels used are the A, E and T channels, implemented as described
    in http://iopscience.iop.org/article/10.1088/0264-9381/18/17/308

    Only power law injections are possible for now

    omega(f) = omega0 * (f/1mHz)^(alpha)

    Input:
    Omega0: The energy density at 1mHz
    alpha :  Power law index

    fs: sampling frequency of the desired noise

    dur: duration

    Output:

    SA_gw : A channel
    SE_gw : E channel
    ST_gw : T Channel
    '''

    H0 = 2.2*10**(-18)
    Omegaf = Omega0*(freqs/(1e-3))**alpha

    # Spectrum of the SGWB
    Sgw = Omegaf*(3/(4*freqs**3))*(H0/np.pi)**2

    # Spectrum of the SGWB signal as seen in LISA data, ie convoluted with the
    # detector response tensor.
    SA_gw = Sgw*R1
    SE_gw = Sgw*R2
    ST_gw = Sgw*R3

    return SA_gw, SE_gw, ST_gw

def noisePSD(Np, Na, fs, seglen,freqs, f0 ):

    '''
    Calculate noise power spectra for A, E and T channels of a stationary equal arm lisa. 
    '''

    # Convert acceleraation and position noise, converted to phase
    Sp, Sa = Np, Na*(1 + 16e-8/freqs**2)*(1.0/(2*np.pi*freqs)**4)

    # arm length
    L = 2.5e9

    if Na > 1e-35 or Np > 1e-35:
        raise ValueError('Unusually loud values of noise provided')


    ## Noise spectra of the TDI Channels
    SAA = (16.0/3.0) * ((np.sin(2*f0))**2) * Sp*(np.cos(2*f0) + 2)/4 \
        + (16.0/3.0) * ((np.sin(2*f0))**2) * Sa*(4*np.cos(2*f0) + 2*np.cos(4*f0) + 6)/4


    SEE = (16.0/3.0) * ((np.sin(2*f0))**2) * Sp*(2 + np.cos(2*f0))/4 \
        + (16.0/3.0) * ((np.sin(2*f0))**2) * Sa*(4 + 4*np.cos(2*f0) +  4*(np.cos(2*f0))**2 )/4

    STT = (16.0/3.0) * ((np.sin(2*f0))**2) * Sp*(1 - np.cos(2*f0))/4 \
        + (16.0/3.0) * ((np.sin(2*f0))**2) * Sa*(2 - 4*np.cos(2*f0) + 2*(np.cos(2*f0))**2)/4

    
    return SAA, SEE, STT



def isgwb_logL(self, theta):

    '''
    Calculate the isgwb bayesian likelihood. Arguments are the lisa class and the parameter samples. 
    '''

    # unpack priors
    alpha, log_omega0, log_Np, log_Na  = theta

    Np, Na =  10**(log_Np), 10**(log_Na)

    # Modelled Noise PSD
    SAA, SEE, STT = noisePSD(Np, Na, self.params['fs'], self.params['seglen'], self.fdata, self.f0)

    # Modelled signal PSD
    SA_gw, SE_gw, ST_gw = gwPSD(10**log_omega0, alpha, self.fdata, self.f0, self.R1, self.R2, self.R3)

    ## We will assume that the covariance matrix is diagonal and will only calcualte those terms. 
    ## This is true for an equal arm stationary lisa. 


    SA_net, SE_net, ST_net = SAA + SA_gw, SEE +  SE_gw, STT + ST_gw

    SA_net = np.repeat(SA_net.reshape(SA_net.size, 1), self.r1.shape[1], axis=1)
    ST_net = np.repeat(ST_net.reshape(ST_net.size, 1), self.r2.shape[1], axis=1)
    SE_net = np.repeat(SE_net.reshape(SE_net.size, 1), self.r3.shape[1], axis=1)

    Loglike  = -0.5*np.sum( (np.abs(self.r1)**2)/SA_net + (np.abs(self.r2)**2)/SE_net + np.log(2*np.pi*SA_net) + np.log(2*np.pi*SE_net) )

    return Loglike
