from __future__ import division
import numpy as np
from src.inj.addISGWB import calDetectorResponse
import os, pdb
#import matplotlib.pyplot as plt

def gwPSD(Omega0, alpha,freqs):

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

    # speed of light
    cspeed = 3e8 #m/s

    # Length of the arms.
    L = 2.5e9

    #Charactersitic frequency
    fstar = cspeed/(2*np.pi*L)

    # define f0 = f/2f*
    f0 = freqs/(2*fstar)

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

def noisePSD(Np, Na, fs, seglen, freqs ):

    if Na > 1e-35 or Np > 1e-35:
        raise ValueError('Unusually loud values of noise provided')

    # Position noise, converted to phase
    Sp = Np*(1 + ((2e-3)/freqs)**4)
    # Acceleration noise converted to phase
    Sa = Na*(1+ ((4e-4)/(freqs))**2)*(1+ (freqs/(8e-3))**4)*(1/(2*np.pi*freqs)**4)

    # arm length
    L = 2.5e9
    cspeed = 3e8 #m/s

    #Charactersitic frequency
    fstar = cspeed/(2*np.pi*L)

    # define f0 = f/2f*
    f0 = freqs/(2*fstar)

    ## Noise spectra of the TDI Channels
    SAA  = (4/9)*(np.sin(2*f0))**2*(np.cos(2*f0)*(12*Sp) + 24*Sp )+ \
    (16/9)*(np.sin(2*f0))**2*(np.cos(2*f0)*12*Sa + np.cos(4*f0)*(6*Sa) + 18*Sa)

    SEE  = (4/3)*(np.sin(2*f0))**2*(4*Sp + (4 + 4*np.cos(2*f0))*Sp) +\
    (16/3)*(np.sin(2*f0))**2*(4*Sa + 4*Sa*np.cos(2*f0) +  4*Sa*(np.cos(2*f0))**2 )

    STT = (4/9)*(np.sin(2*f0))**2* (12 - 12*np.cos(2*f0))*Sp + \
    (16/9)*(np.sin(2*f0))**2*(6 - 12*np.cos(2*f0) + 6*(np.cos(2*f0))**2)*Sa

    return SAA, SEE, STT



def logL(rA, rE, rT, fdata, config, theta):
    # Check to see if we are getting data
    if fdata.size == 0:
        pdb.set_trace()


    # unpack priors
    alpha, log_omega0, log_Np, log_Na  = theta

    #alpha, omega0 = 0.65, 1e-7
    Np, Na =  10**(log_Np), 10**(log_Na)

    # For Now we will work only with the A channel and the <AA> correlation
    delf = 1/config['seglen']

    # Nyquist freq
    fmax =  config['fs']/2

    # Number of segmants
    nsegs = int(np.floor(config['dur']/config['seglen']))

    # Modelled Noise PSD
    SAA, SEE, STT = noisePSD(Np, Na, config['fs'], config['seglen'], fdata)

    # Modelled signal PSD
    SA_gw, SE_gw, ST_gw = gwPSD(10**log_omega0, alpha, fdata)

    # Covariance matric
    CIJ = np.zeros((rA.shape[0], 3, 3 ))
    CIJ[:, 0, 0] = (SAA + SA_gw)
    CIJ[:, 1, 1] = (SEE + SE_gw)
    CIJ[:, 2, 2] = (STT + ST_gw)

    # Inverse and determinant
    invCIJ = np.linalg.inv(CIJ)
    detC = np.linalg.det(CIJ)
    #detC = (SAA + SA_gw)*(SEE + SE_gw)*(STT + ST_gw)


    # Initialize Loglike
    Loglike = 0


    # Calcualate likelihood, for single detector it is easy
    # Note that as of now, the exponent needs to have a factor for 0.5
    # in it. Some references show 2, but that is incorrect.

    for ii in range(0, nsegs):

    #    # Create column vector
        rbar = np.stack((rA[:, ii], rE[:, ii], rT[:, ii]), axis=1)
        rmat = np.zeros(invCIJ.shape, dtype='complex')
        rmat[:, :, 0] = rbar
        rmat[:, :, 1] = rbar
        rmat[:, :, 2] = rbar

        Loglike = Loglike +  np.real(np.sum(- 0.5*np.conj(rmat)*rmat*invCIJ) - np.sum(0.5*np.log(2*np.pi*detC)))

    return Loglike
