from __future__ import division
import numpy as np
from src.inj.addISGWB import calDetectorResponse
import os, pdb
#import matplotlib.pyplot as plt

def gwPSD(Omega0, alpha,freqs, f0):

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

    if not os.path.isfile('detector_response.txt'):
        RA, RE, RT = calDetectorResponse(f0, 'TDI')
        np.savetxt('detector_response.txt', (RA, RE, RT))
    else:
        RA, RE, RT = np.loadtxt('detector_response.txt')

    if RA.shape != freqs.shape:
        RA, RE, RT = calDetectorResponse(f0, 'TDI')
        np.savetxt('detector_response.txt', (RA, RE, RT))



    H0 = 2.2*10**(-18)
    Omegaf = Omega0*(freqs/(1e-3))**alpha

    # Spectrum of the SGWB
    Sgw = Omegaf*(3/(4*freqs**3))*(H0/np.pi)**2

    # Spectrum of the SGWB signal as seen in LISA data, ie convoluted with the
    # detector response tensor.
    SA_gw = Sgw*RA
    SE_gw = Sgw*RE
    ST_gw = Sgw*RT

    return SA_gw, SE_gw, ST_gw

def noisePSD(Np, Na, fs, seglen,freqs, f0 ):

    # Convert acceleraation and position noise, converted to phase
    Sp, Sa = Np, Na*(1 + 16e-8/freqs**2)*(1.0/(2*np.pi*freqs)**4)

    # arm length
    L = 2.5e9

    if Na/(4*L**2) > 1e-35 or Np/(4*L**2) > 1e-35:
        raise ValueError('Unusually loud values of noise provided')

    ## Noise spectra of the TDI Channels
    SAA  = (4/9)*(np.sin(2*f0))**2*(np.cos(2*f0)*(12*Sp) + 24*Sp )+ \
    (16/9)*(np.sin(2*f0))**2*(np.cos(2*f0)*12*Sa + np.cos(4*f0)*(6*Sa) + 18*Sa)

    SEE  = (4/3)*(np.sin(2*f0))**2*(4*Sp + (4 + 4*np.cos(2*f0))*Sp) +\
    (16/3)*(np.sin(2*f0))**2*(4*Sa + 4*Sa*np.cos(2*f0) +  4*Sa*(np.cos(2*f0))**2 )

    STT = (4/9)*(np.sin(2*f0))**2* (12 - 12*np.cos(2*f0))*Sp + \
    (16/9)*(np.sin(2*f0))**2*(6 - 12*np.cos(2*f0) + 6*(np.cos(2*f0))**2)*Sa

    return SAA, SEE, STT



def isgwb_logL(lisa, theta):

    '''
    Calculate the isgwb bayesian likelihood. Arguments are the lisa class and the parameter samples. 
    '''

    # unpack priors
    alpha, log_omega0, log_Np, log_Na  = theta

    #alpha, omega0 = 0.65, 1e-7
    Np, Na =  10**(log_Np), 10**(log_Na)

    # Modelled Noise PSD
    SAA, SEE, STT = noisePSD(Np, Na, lisa.params['fs'], lisa.params['seglen'], lisa.fdata, lisa.f0)

    # Modelled signal PSD
    SA_gw, SE_gw, ST_gw = gwPSD(10**log_omega0, alpha, lisa.fdata, lisa.f0)

    ## We will assume that the covariance matrix is diagonal and will only calcualte those terms. 
    ## This is true for an equal arm stationary lisa. 


    SA_net, SE_net, ST_net = SAA + SA_gw, SEE +  SE_gw, STT + ST_gw

    SA_net = np.repeat(SA_net.reshape(SA_net.size, 1), lisa.r1.shape[1], axis=1)
    ST_net = np.repeat(ST_net.reshape(ST_net.size, 1), lisa.r2.shape[1], axis=1)
    SE_net = np.repeat(SE_net.reshape(SE_net.size, 1), lisa.r3.shape[1], axis=1)

    Loglike  = -0.5*np.sum( (np.abs(lisa.r1)**2)/SA_net + (np.abs(lisa.r2)**2)/SE_net + np.log(2*np.pi*SA_net) + np.log(2*np.pi*SE_net) )

    return Loglike
