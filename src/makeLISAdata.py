from __future__ import division
import numpy as np
import scipy.signal as sg
from src.inj.addISGWB import injectISGWB
from tools.makeGaussianData import gaussianData

from scipy.interpolate import interp1d as intrp
import os, pdb


class LISAdata(Antennapatterns):

    '''
    Class for lisa data. Includes methods for generation of gaussian instrumental noise, and generation 
    of isotropic stochastic background. Any eventually signal models should be added as methods here. This
    has the Antennapatterns class as a super class. 
    '''

    def __init__(self, params, inj):
        self.params = params
        self.inj = inj
        self.armlength = 2.5e9 ## armlength in meters

    ## Method for reading frequency domain spectral data if given in an npz file
    def read_spectrum(self):
        if os.path.isfile(self.params['input_spectrum']) and not self.params['doPreProc']:
            print("loading freq domain data from input file")

            data = np.load(self.params['input_spectrum'])
            rA    = data['rA']
            rE    = data['rE']
            rT    = data['rT']
            fdata = data['fdata']

            return rA, rE, rT, fdata

    ## Method for reading time domain data from an ascii file. This is useful for the mldc
    def read_time_series(self):
        if os.path.isfile(self.params['simfile']):
            hoft = np.loadtxt(self.params['simfile'])
            times, hX, hY, hZ = hoft[:, 0], hoft[:, 1], hoft[:, 2], hoft[:, 3]

    def gen_noise(self):
        ## Call noise TDI and generate gaussian detector noise. 
          
        # --------------------- Generate Fake Data + Noise -----------------------------
        print("Simulating isgwb data for analysis ...")

       # speed of light
        cspeed = 3e8 #m/s
        delf  = 1.0/self.params['dur']
        frange = np.arange(self.params['fmin'], self.params['fmax'], delf) # in Hz

        # To implement TDI we need time shifts of multiples of L.
        tlag  = self.armlength/cspeed


        # Using 2017 Lisa proposal noise estimations, but treating them as white in position and acceleration respectively. 

        Sp = 1.6e-43
        Sa = 1.44e-48*(1.0/(2.0*np.pi*frange)**4)

        # Generate data
        np12 = gaussianData(Sp, frange, self.params['fs'], self.params['dur'])
        np21 = gaussianData(Sp, frange, self.params['fs'], self.params['dur'])
        np13 = gaussianData(Sp, frange, self.params['fs'], self.params['dur'])
        np31 = gaussianData(Sp, frange, self.params['fs'], self.params['dur'])
        np23 = gaussianData(Sp, frange, self.params['fs'], self.params['dur'])
        np32 = gaussianData(Sp, frange, self.params['fs'], self.params['dur'])

        na12 = gaussianData(Sa, frange, self.params['fs'], self.params['dur'])
        na21 = gaussianData(Sa, frange, self.params['fs'], self.params['dur'])
        na13 = gaussianData(Sa, frange, self.params['fs'], self.params['dur'])
        na31 = gaussianData(Sa, frange, self.params['fs'], self.params['dur'])
        na23 = gaussianData(Sa, frange, self.params['fs'], self.params['dur'])
        na32 = gaussianData(Sa, frange, self.params['fs'], self.params['dur'])
    

        # time array and time shift array
        time_arr = np.linspace(0, self.params['dur'] , num=np12.size, endpoint=False)
        delt = tarr[2] - tarr[1]

        if 1.0/delt != self.params['fs']:
            raise ValueError('Time series generated not consistant with the sampling frequency')
            

        # shifted time series
        tshift = tarr - tlag

        ## The three dopper channels for each arms
        h12  = np12 - na12 + np.interp(tshift, tarr, na21, left=na21[0])
        h21  = np21 + na21 - np.interp(tshift, tarr, na12, left=na12[0])

        h23  = np23 - na23 + np.interp(tshift, tarr, na32, left=na32[0])
        h32  = np32 + na32 - np.interp(tshift, tarr, na23, left=np23[0])

        h31  = np31 - na31 + np.interp(tshift, tarr, na13, left=na13[0])
        h13  = np13 + na13 - np.interp(tshift, tarr, na31, left=na31[0])


        # The michelson channels are formed from the doppler channels
        h1 = np.interp(tshift, tarr, h12, left=h12[0]) + h21 -\
        np.interp(tshift, tarr, h13, left=h13[0]) - h31

        h2 = np.interp(tshift, tarr, h23, left=h23[0]) + h32 -\
        np.interp(tshift, tarr, h21, left=h21[0]) - h12

        h3 = np.interp(tshift, tarr, h31, left=h31[0]) + h13 -\
        np.interp(tshift, tarr, h32, left=h32[0]) - h23

        return h1, h2, h3

    def gen_isgwb(self):
        
        '''
        Generate time series mock LISA data and take ffts. The output are the ffts.

        Michelson channels are manipulated to generate the A, E and T channels.
        We implement TDI as described in

        http://iopscience.iop.org/article/10.1088/0264-9381/18/17/308
        '''

        # --------------------- Generate Fake Data + Noise -----------------------------
        print("Simulating isgwb data for analysis ...")

        # speed of light
        cspeed = 3e8 #m/s


        delf  = 1.0/self.params['dur']
        freqs = np.arange(self.params['fmin'], self.params['fmax'], delf)

        #Charactersitic frequency
        fstar = cspeed/(2*np.pi*self.armlength)

        # define f0 = f/2f*
        f0 = freqs/(2*fstar)



        ## There are the responses for the three arms
        ## Naturally RIJ = RJI

        R12, R23, R31 = self.AET_response(f0)

        H0 = 2.2*10**(-18)
        Omegaf = Omega0*(freqs/(1e-3))**alpha

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*freqs**3))*(H0/np.pi)**2

        # Generate time series data for the doppler channels
        wht_data = np.random.normal(size=int(fs*dur))
        f_wht, wht_psd = sg.welch(wht_data, fs=fs, nperseg=int(fs*dur))


        # Spectrum of the SGWB signal as seen in LISA data, ie convoluted with the
        # detector response tensor, and interpolated to the psd frequencies.
        S12_gw = np.interp(f_wht,freqs, Sgw*R12)
        S23_gw = np.interp(f_wht,freqs,Sgw*R23)
        S31_gw = np.interp(f_wht,freqs,Sgw*R31)

        S21_gw = np.interp(f_wht,freqs,Sgw*R12)
        S32_gw = np.interp(f_wht,freqs,Sgw*R23)
        S13_gw = np.interp(f_wht,freqs,Sgw*R31)

    # PSD of band-limited white gaussian noise
    N02 = 1.0/(f_wht[-1] - f_wht[0])

    ## Do an rrft
    wht_rfft = np.fft.rfft(wht_data)

    # Generate colored ffts
    h12_fft = wht_rfft*np.sqrt(S12_gw)/np.sqrt(N02)
    h23_fft = wht_rfft*np.sqrt(S23_gw)/np.sqrt(N02)
    h31_fft = wht_rfft*np.sqrt(S31_gw)/np.sqrt(N02)
    h21_fft = wht_rfft*np.sqrt(S21_gw)/np.sqrt(N02)
    h32_fft = wht_rfft*np.sqrt(S32_gw)/np.sqrt(N02)
    h13_fft = wht_rfft*np.sqrt(S13_gw)/np.sqrt(N02)

    h12_gw = np.fft.irfft(h12_fft, n=wht_data.size)
    h23_gw = np.fft.irfft(h23_fft, n=wht_data.size)
    h31_gw = np.fft.irfft(h31_fft, n=wht_data.size)
    h21_gw = np.fft.irfft(h21_fft, n=wht_data.size)
    h32_gw = np.fft.irfft(h32_fft, n=wht_data.size)
    h13_gw = np.fft.irfft(h13_fft, n=wht_data.size)



    

    # To generate michelson channels we need time shifts of multiples of L.
    tlag  = L/cspeed

    # time array and time shift array
    tarr = np.linspace(0, dur , num= h12_gw.size, endpoint=False)
    delt = tarr[2] - tarr[1]

    if 1.0/delt != fs:
        pdb.set_trace()

    # shifted time series
    tshift = tarr - tlag

    '''
    h1_gw = np.interp(tshift, tarr, h12_gw, left=h12_gw[0]) + h21_gw -\
    np.interp(tshift, tarr, h31_gw, left=h31_gw[0]) - h13_gw

    h2_gw = np.interp(tshift, tarr, h23_gw, left=h23_gw[0]) + h32_gw -\
    np.interp(tshift, tarr, h21_gw, left=h21_gw[0]) - h12_gw

    h3_gw = np.interp(tshift, tarr, h31_gw, left=h31_gw[0]) + h13_gw -\
    np.interp(tshift, tarr, h32_gw, left=h32_gw[0]) - h23_gw
    '''

    h1_gw, h2_gw, h3_gw = h12_gw - h13_gw, h23_gw - h21_gw,  h31_gw - h32_gw 
















    def do_tdi(self):
    # Implementing TDI ... We assume that h1, h2 and h3 are Michelson Channels.

    # speed of light
    cspeed = 3e8 #m/s
    # Length of the arms.
    L = 2.5e9
    # To implement TDI we need time shifts of multiples of L.
    tlag  = L/cspeed

    # time array and time shift array
    tarr = np.linspace(0, self.params['dur'] , num= h1.size, endpoint=False)
    delt = tarr[2] - tarr[1]

    if 1.0/delt != self.params['fs']:
        pdb.set_trace()

    # shifted time series
    tshift = tarr - 2*tlag


    hX = h1 - np.interp(tshift, tarr, h1, left=h1[0])
    hY = h2 - np.interp(tshift, tarr, h2, left=h2[0])
    hZ = h3 - np.interp(tshift, tarr, h3, left=h3[0])

    hA = (1.0/3.0)*(2*hX - hY - hZ)
    hE = (1.0/np.sqrt(3.0))*(hZ - hY)
    hT = (1.0/3.0)*(hX + hY + hZ)

    ## -------------------------------------------------- Run some diagnostics on TDI if necessary -----------------------------------------------------------------------
    doDiag=1
    if doDiag:
        diagTDI(hA, hE, hT, self.params['fs'], self.params['dur'], inj)

    #---------------------------- TIME SERIES DATA ----------------------------------------------------------------------------------------------------


    print "Calculating power spectra... "
    #---------------------------  Calc PSD Begins -----------------------------------------------------------------------------------------------------

    # Number of segmants
    nsegs = 2*int(np.floor(self.params['dur']/self.params['seglen'])) - 1

    Nperseg=int(self.params['fs']*self.params['seglen'])

    # High pass filter
    order = 8
    zz, pp, kk = sg.butter(order, 0.5*self.params['fmin']/(self.params['fs']/2), btype='highpass', output='zpk')
    sos = sg.zpk2sos(zz, pp, kk)

    hA = sg.sosfiltfilt(sos, hA)
    hE = sg.sosfiltfilt(sos, hE)
    hT = sg.sosfiltfilt(sos, hT)

    # Map of spectrum
    rA1 = np.zeros((1 + int(Nperseg/2), nsegs), dtype='complex')
    rE1 = np.zeros((1 + int(Nperseg/2), nsegs), dtype='complex')
    rT1 = np.zeros((1 + int(Nperseg/2), nsegs), dtype='complex')

    # Hann Window
    hwin = np.hanning(Nperseg)

    # We will use 50% overlapping segments
    for ii in range(0, nsegs):

        idxmin = int(0.5*ii*Nperseg)
        idxmax = idxmin + Nperseg


        fftA =   np.fft.fft(hwin*hA[idxmin:idxmax])
        fftE =   np.fft.fft(hwin*hE[idxmin:idxmax])
        fftT =   np.fft.fft(hwin*hT[idxmin:idxmax])

        # Select only non-negetive frequencies
        rA1[:, ii] = fftA[0:int(Nperseg/2)+1]
        rE1[:, ii] = fftE[0:int(Nperseg/2)+1]
        rT1[:, ii] = fftT[0:int(Nperseg/2)+1]


    # "Cut" to desired frequencies
    fftfreqs = (self.params['fs']/2)*np.linspace(0, 1,  1 + Nperseg/2)
    idx = np.logical_and(fftfreqs >=  self.params['fmin'] , fftfreqs <=  self.params['fmax'])

    # Output arrays
    fdata = fftfreqs[idx]

    # Get desired frequencies only
    # We want to normalize ffts so thier square give the psd
    # 0.375 is to adjust for hann windowing, sqrt(2) for single sided
    rA = np.sqrt(2.0/0.375)*rA1[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))
    rE = np.sqrt(2.0/0.375)*rE1[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))
    rT = np.sqrt(2.0/0.375)*rT1[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))


    if (np.sum(np.isnan(fdata)) !=0) | (np.sum(np.isnan(rA)) !=0) | (np.sum(np.isnan(rE)) !=0) | (np.sum(np.isnan(rT)) !=0):
        pdb.set_trace()
        raise ValueError('Frequencies range requested is above or below the allowed range')


    np.savez(input_spectrum, rT=rT,rE=rE, rA=rA, fdata=fdata)
    return rA, rE, rT, fdata





### --------------------- Diagnostic function for TDI -----------------------------------------------------------------


def diagTDI(hA, hE, hT, fs, dur, inj):

    # Some code for testing if this ever breaks
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    import pdb
    from src.inj.addISGWB import calDetectorResponse

    freqs = np.arange(1e-4, 1e-1, 1e-5) # in Hz

    Np, Na = 4e-42, 3.6e-49
    # Position noise, converted to phase
    Sp = Np*(1 + ((2e-3)/freqs)**4)
    # Acceleration noise converted to phase
    Sa = Na*(1+ ((4e-4)/(freqs))**2)*(1+ (freqs/(8e-3))**4)*(1/(2*np.pi*freqs)**4)

    #Charactersitic frequency
    fstar = 3.0e8/(2*np.pi*2.5e9)

    # define f0 = f/2f*
    f0 = freqs/(2*fstar)

    ## Noise spectra of the TDI Channels
    SAA  = (4/9)*(np.sin(2*f0))**2*(np.cos(2*f0)*(12*Sp) + 24*Sp )+ \
    (16/9)*(np.sin(2*f0))**2*(np.cos(2*f0)*12*Sa + np.cos(4*f0)*(6*Sa) + 18*Sa)
    SEE  = (4/3)*(np.sin(2*f0))**2*(4*Sp + (4 + 4*np.cos(2*f0))*Sp) +\
    (16/3)*(np.sin(2*f0))**2*(4*Sa + 4*Sa*np.cos(2*f0) +  4*Sa*(np.cos(2*f0))**2 )
    STT = (4/9)*(np.sin(2*f0))**2* (12 - 12*np.cos(2*f0))*Sp + \
    (16/9)*(np.sin(2*f0))**2*(6 - 12*np.cos(2*f0) + 6*(np.cos(2*f0))**2)*Sa


    ## Add noise spectra if necessary
    if inj['doInj']:
        
        if not os.path.isfile('detector_response.txt'):
            RA, RE, RT = calDetectorResponse(f0, 'TDI')
        else:
            RA, RE, RT = np.loadtxt('detector_response.txt')

        if RA.shape != freqs.shape:
            RA, RE, RT = calDetectorResponse(f0, 'TDI')


        Omega0, alpha = 10**(inj['ln_omega0']), inj['alpha']

        H0 = 2.2*10**(-18)
        Omegaf = Omega0*(freqs/(1e-3))**alpha

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*freqs**3))*(H0/np.pi)**2

        # Spectrum of the SGWB signal as seen in LISA data, ie convoluted with the
        # detector response tensor.
        SA_gw = Sgw*RA
        SE_gw = Sgw*RE
        ST_gw = Sgw*RT
        
        #SAA, SEE, STT  = SAA + SA_gw, SEE + SE_gw, STT + ST_gw
        SAA, SEE, STT = SA_gw, SE_gw, ST_gw
    ### ---------------------------------- Check data PSD now --------------------------------------
    delT =5e4
    nsegs = int(dur/delT)
    num = int(fs*delT)
    PSDA = np.zeros((int(1+ num/2.0),nsegs))
    PSDE = np.zeros((int(1+ num/2.0),nsegs))
    PSDT = np.zeros((int(1+ num/2.0),nsegs))

    for ii in range(nsegs):
        idxmin = ii*num
        idxmax = idxmin + num

        fPSD, PSDA[:, ii] = welch(hA[idxmin:idxmax], fs=fs, nperseg=num)
        fPSD, PSDE[:, ii] = welch(hE[idxmin:idxmax], fs=fs, nperseg=num)
        fPSD, PSDT[:, ii] = welch(hT[idxmin:idxmax], fs=fs, nperseg=num)

    fmin, fmax = 2e-4, 1e-1
    ymin, ymax = 1e-43, 0.8e-38

    plt.loglog(freqs, SAA, label='required')
    plt.loglog(fPSD, np.mean(PSDA, axis=1),label='PSDA')
    plt.xlim(fmin, fmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig('psdA.png', dpi=125)
    plt.close() 

    plt.loglog(freqs, SEE, label='required')
    plt.loglog(fPSD, np.mean(PSDE, axis=1),label='PSDE')
    plt.xlim(fmin, fmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig('psdE.png', dpi=125)
    plt.close() 

    plt.loglog(freqs, STT, label='required')
    plt.loglog(fPSD, np.mean(PSDT, axis=1),label='PSDT')
    plt.xlim(fmin, fmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig('psdT.png', dpi=125)
    plt.close() 

    pdb.set_trace()