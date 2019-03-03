from __future__ import division
import numpy as np
import scipy.signal as sg
from tools.makeGaussianData import gaussianData
from src.det_response import Antennapatterns
from scipy.interpolate import interp1d as intrp
import os


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

    def gen_michelson_noise(self):
        
        '''
        Generate interferometric michelson noise
        '''
          
        # --------------------- Generate Fake Noise -----------------------------
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
        tarr = np.linspace(0, self.params['dur'] , num=np12.size, endpoint=False)
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

    def gen_aet_noise(self):
        
        cspeed = 3e8 #m/s

        # michelson channels
        hm1, hm2, hm3 = self.gen_michelson_noise()

        ## In time domain we need to do some interpolation to get to the appropriate TDI channels
        tarr = np.linspace(0, self.params['dur'] , num= hm1.size, endpoint=False)

        # shifted time series
        tshift = tarr - 2*self.armlength/cspeed

        hX = hm1 - np.interp(tshift, tarr, hm1, left=hm1[0])
        hY = hm2 - np.interp(tshift, tarr, hm2, left=hm2[0])
        hZ = hm3 - np.interp(tshift, tarr, hm3, left=hm3[0])

        h1_noi = (1.0/3.0)*(2*hX - hY - hZ)
        h2_noi = (1.0/np.sqrt(3.0))*(hZ - hY)
        h3_noi = (1.0/3.0)*(hX + hY + hZ)

        return h1_noi, h2_noi, h3_noi

    def gen_aet_isgwb(self):
        
        '''
        Generate time series mock LISA data. The output are time domain TDI data. Michelson 
        channels are manipulated to generate the A, E and T channels. We implement TDI as described in

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
        RAA, REE, RTT = self.tdi_isgwb_response(f0)

        H0 = 2.2*10**(-18) ## in SI units

        Omegaf = (10**self.inj['ln_omega0'])*(freqs/(1e-3))**self.inj['alpha']

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*freqs**3))*(H0/np.pi)**2

        # Generate time series data for the doppler channels
        wht_data = np.random.normal(size=int(self.params['fs']*self.params['dur']))
        f_wht  = np.fft.rfftfreq(wht_data.size, 1.0/self.params['fs'])

        # Spectrum of the SGWB signal as seen in LISA data, ie convoluted with the
        # detector response tensor, and interpolated to the psd frequencies.
        SAA_gw = np.interp(f_wht,freqs, Sgw*RAA)
        SEE_gw = np.interp(f_wht,freqs,Sgw*REE)
        STT_gw = np.interp(f_wht,freqs,Sgw*RTT)

        # PSD of band-limited white gaussian noise
        N02 = 1.0/(f_wht[-1] - f_wht[0])

        ## Do an rrft
        wht_rfft = np.fft.rfft(wht_data)

        # Generate colored ffts
        hA_fft = wht_rfft*np.sqrt(SAA_gw)/np.sqrt(N02)
        hE_fft = wht_rfft*np.sqrt(SEE_gw)/np.sqrt(N02)
        hT_fft = wht_rfft*np.sqrt(STT_gw)/np.sqrt(N02)

        ## Generate time series data
        h1_gw = np.fft.irfft(hA_fft, n=wht_data.size)
        h2_gw = np.fft.irfft(hE_fft, n=wht_data.size)
        h3_gw = np.fft.irfft(hT_fft, n=wht_data.size)

        return h1_gw, h2_gw, h3_gw

    def read_data(self):
        
        '''
        Read in time domain data from an ascii txt file. Since this was used primarily for 
        the MLDC, it assumes that the data is given in X,Y and Z channels and converts to 
        A, E and T. 
        '''
        
        hoft = np.loadtxt(self.params['datafile'])
        times, hX, hY, hZ = hoft[:, 0], hoft[:, 1], hoft[:, 2], hoft[:, 3]

        delt = times[1] - times[0]

        ## Downsample
        if self.params['fs'] < 1.0/delt:

            hX = sg.decimate(hX, int(1.0/(self.params['fs']*delt)))
            hY = sg.decimate(hY, int(1.0/(self.params['fs']*delt)))
            hZ = sg.decimate(hZ, int(1.0/(self.params['fs']*delt)))
            self.params['fs'] = (1.0/delt)/int(1.0/(self.params['fs']*delt))
            times = self.params['fs']*np.arange(0, hX.size, 1)
        else:
            self.params['fs'] = 1.0/delt

        hA = (1.0/3.0)*(2*hX - hY - hZ)
        hE = (1.0/np.sqrt(3.0))*(hZ - hY)
        hT = (1.0/3.0)*(hX + hY + hZ)

        return hA, hE, hT


    def tser2fser(self, h1, h2, h3):
        
        '''
        Convert time domain data to fourier domain and return ffts. The convention is that the 
        the ffts are divided by the sampling frequency and corrected for windowing. A hann window 
        is applied by default when moving to the fourier domain. The ffts are also normalized so that
        thier square gives the PSD.
        '''

        print ("Calculating fourier spectra... ")

        # Number of segmants
        nsegs = int(np.floor(self.params['dur']/self.params['seglen']))

        Nperseg=int(self.params['fs']*self.params['seglen'])

        # Apply band pass filter
        order = 8
        zz, pp, kk = sg.butter(order, [0.5*self.params['fmin']/(self.params['fs']/2), 0.4*self.params['fs']/(self.params['fs']/2)], btype='bandpass', output='zpk')
        sos = sg.zpk2sos(zz, pp, kk)

        h1 = sg.sosfiltfilt(sos, h1)
        h2 = sg.sosfiltfilt(sos, h2)
        h3 = sg.sosfiltfilt(sos, h3)

        # Map of spectrum
        r1 = np.zeros((1 + int(Nperseg/2), nsegs), dtype='complex')
        r2 = np.zeros((1 + int(Nperseg/2), nsegs), dtype='complex')
        r3 = np.zeros((1 + int(Nperseg/2), nsegs), dtype='complex')

        # Hann Window
        hwin = np.hanning(Nperseg)

        # We will NOT use any segments
        for ii in range(0, nsegs):

            idxmin = int(ii*Nperseg)
            idxmax = idxmin + Nperseg

            r1[:, ii] =   np.fft.rfft(hwin*h1[idxmin:idxmax])
            r2[:, ii] =   np.fft.rfft(hwin*h2[idxmin:idxmax])
            r3[:, ii] =   np.fft.rfft(hwin*h3[idxmin:idxmax])


        # "Cut" to desired frequencies
        fftfreqs = np.fft.rfftfreq(Nperseg, 1.0/self.params['fs'])

        idx = np.logical_and(fftfreqs >=  self.params['fmin'] , fftfreqs <=  self.params['fmax'])

        # Output arrays
        fdata = fftfreqs[idx]
        
        # Get desired frequencies only
        # We want to normalize ffts so thier square give the psd
        # 0.375 is to adjust for hann windowing, sqrt(2) for single sided
        r1 = 2.0/np.sqrt(0.375)*r1[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))
        r2 = 2.0/np.sqrt(0.375)*r2[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))
        r3 = 2.0/np.sqrt(0.375)*r3[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))

        return r1, r2, r3, fdata