from __future__ import division
import numpy as np
import scipy.signal as sg
from src.freqDomain import freqDomain
from scipy.interpolate import interp1d as intrp
import os


class LISAdata(freqDomain):

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

        '''
        Read an input frequency domain data file. Returns the fourier transform of the data from the three channels and an array of reference frequencyes

        Returns
        ---------

        rA, rE, rT, fdata   :   float 

        '''

        if os.path.isfile(self.params['input_spectrum']) and not self.params['doPreProc']:
            print("loading freq domain data from input file")

            data = np.load(self.params['out_dir'] + '/' +self.params['input_spectrum'])
            r1    = data['r1']
            r2    = data['r2']
            r3    = data['r3']
            fdata = data['fdata']

            return r1, r2, r3, fdata


    def gaussianData(self, Sh,freqs, fs=1, dur=1e5):
   
        '''
        Script for generation time series noise drawn from a gaussian process of a given spectral density.  Adapted from gaussian_noise.m from stamp

        Parameters
        -----------
        
        Sh : (float)
            A frequency array with the desired power spectral density
        freqs : (float)
            An array with corresponding frequencies to Sh

        fs : (float)
            SampleRate in Hz
        
        dur : (int)
            Duration in seconds

     
        Returns
        ---------
    
        ht : float
        Array with time series data of duration, dur with the prescribed spectrum Sh

       
        '''
    
        # Number of data points in the time series
        N = int(fs*dur)

        # prepare for FFT
        if  np.mod(N,2)== 0 :
            numFreqs = N/2 - 1;
        else:
            numFreqs = (N-1)/2;

        # We will make an array of the desired frequencies
        delF = 1/dur
        fmin = 1/dur
        fmax = np.around(dur*fs/2)/dur
        delF = 1/dur

        # The output frequency series
        fout = np.linspace(fmin, fmax, numFreqs)

        # Interpolate to the desired frequencies
        norms = np.interp(fout, freqs, Sh)

        # Amplitude for for ifft
        norms = np.sqrt(norms*fs*N)/2.0

        # Normally distributed in frequency space
        re1 = norms*np.random.normal(size=fout.size)
        im1 = norms*np.random.normal(size=fout.size)

        htilda = re1 + 1j*im1

        if np.mod(N, 2) == 0:
            htilda = np.concatenate((np.zeros(1), htilda,np.zeros(1), np.flipud(np.conjugate(htilda))))
        else:
            htilda = np.concatenate((np.zeros(1),htilda, np.conjugate(np.flipud(htilda))))

        # Take inverse fft to get time series data
        ht = np.real(np.fft.ifft(htilda, N))

        return ht

    def gen_michelson_noise(self):
        
        '''
        Generate interferometric michelson (time-domain) noise, using freqDomain.fundamental_noise_spectrum

        Returns
        ---------
        h1, h2, h3 : float
            Time series data for the three michelson channels
        '''

        # --------------------- Generate Fake Noise -----------------------------
        print("Simulating michelson data ...")

       # speed of light
        cspeed = 3e8 #m/s
        delf  = 1.0/self.params['dur']
        frange = np.arange(self.params['fmin'], self.params['fmax'], delf) # in Hz
        fstar = 3e8/(2*np.pi*self.armlength)
        f0 = frange/(2*fstar)

        Sp, Sa = self.fundamental_noise_spectrum(frange, Np=10**self.inj['log_Np'], Na=10**self.inj['log_Na'])

        # To implement TDI we need time shifts of multiples of L.
        tlag  = self.armlength/cspeed

        ## If we have a smaller fs than 2 samples a second, we will use 2 Hz as the sampling freq
        ## If the sampling frequency is too low, that doesn't play well with the time-shifts
        if self.params['fs'] < 4:
            print('Desired sample rate is too low for time shifts. Temporarily increasing ...')
            fs_eff = 2
        else:
            fs_eff = self.params['fs']

        # Generate data
        np12 = self.gaussianData(Sp, frange, fs_eff, 1.1*self.params['dur'])
        np21 = self.gaussianData(Sp, frange, fs_eff, 1.1*self.params['dur'])
        np13 = self.gaussianData(Sp, frange, fs_eff, 1.1*self.params['dur'])
        np31 = self.gaussianData(Sp, frange, fs_eff, 1.1*self.params['dur'])
        np23 = self.gaussianData(Sp, frange, fs_eff, 1.1*self.params['dur'])
        np32 = self.gaussianData(Sp, frange, fs_eff, 1.1*self.params['dur'])
        
        na12 = self.gaussianData(Sa, frange, fs_eff, 1.1*self.params['dur'])
        na21 = self.gaussianData(Sa, frange, fs_eff, 1.1*self.params['dur'])
        na13 = self.gaussianData(Sa, frange, fs_eff, 1.1*self.params['dur'])
        na31 = self.gaussianData(Sa, frange, fs_eff, 1.1*self.params['dur'])
        na23 = self.gaussianData(Sa, frange, fs_eff, 1.1*self.params['dur'])
        na32 = self.gaussianData(Sa, frange, fs_eff, 1.1*self.params['dur'])
 
        # time array and time shift array
        tarr = np.linspace(0, 1.1*self.params['dur'] , num=np12.size, endpoint=False)
        delt = tarr[2] - tarr[1]

        tlag_idx = int(round(tlag*fs_eff))

        if 1.0/delt != fs_eff:
            raise ValueError('Time series generated not consistant with the sampling frequency')
            

        ## One way dopper channels for each arms
        h12  = np12[tlag_idx:] - na12[tlag_idx:] + na21[0:-tlag_idx] 
        h21  = np21[tlag_idx:] + na21[tlag_idx:] - na12[0:-tlag_idx] 

        h23  = np23[tlag_idx:] - na23[tlag_idx:] + na32[0:-tlag_idx] 
        h32  = np32[tlag_idx:] + na32[tlag_idx:] - na23[0:-tlag_idx] 

        h31  = np31[tlag_idx:] - na31[tlag_idx:] + na13[0:-tlag_idx] 
        h13  = np13[tlag_idx:] + na13[tlag_idx:] - na31[0:-tlag_idx]
        

        # The Michelson channels, formed from the doppler channels
        h1 = h12[0:-tlag_idx] + h21[tlag_idx:] - h13[0:-tlag_idx] - h31[tlag_idx:]
        h2 = h23[0:-tlag_idx] + h32[tlag_idx:] - h21[0:-tlag_idx] - h12[tlag_idx:]
        h3 = h31[0:-tlag_idx] + h13[tlag_idx:] - h32[0:-tlag_idx] - h23[tlag_idx:] 
        
        '''
        Older way of doing time shifts is commented out here. Interp doesn't work since it
        creates correlated samples, but I leave it here for reference. - Sharan
        
        h1 = np.interp(tshift, tarr, h12, left=h12[0]) + h21 -\
        np.interp(tshift, tarr, h13, left=h13[0]) - h31
        
        h2 = np.interp(tshift, tarr, h23, left=h23[0]) + h32 -\
        np.interp(tshift, tarr, h21, left=h21[0]) - h12

        h3 = np.interp(tshift, tarr, h31, left=h31[0]) + h13 -\
        np.interp(tshift, tarr, h32, left=h32[0]) - h23
        '''

        return tarr[tlag_idx:], h1, h2, h3



    def gen_xyz_noise(self):
        
        '''
        Generate interferometric A, E and T channel TDI (time-domain) noise, using freqDomain.fundamental_noise_spectrum

        Returns
        ---------
    
        h1_noi, h2_noi, h3_noi : float
            Time series data for the three TDI channels

        '''
        cspeed = 3e8 #m/s

        # michelson channels
        tarr, hm1, hm2, hm3 = self.gen_michelson_noise()

        fs_eff = 1.0/(tarr[1] - tarr[0])
        delt = 1.0/fs_eff
        # Introduce time series
        tshift = 2*self.armlength/cspeed
        tshift_idx = int(round(tshift*fs_eff))

        hX = hm1[tshift_idx:] - hm1[0:-tshift_idx]
        hY = hm2[tshift_idx:] - hm2[0:-tshift_idx]
        hZ = hm3[tshift_idx:] - hm3[0:-tshift_idx]

        return tarr[tshift_idx:], hX, hY, hZ



    def gen_aet_noise(self):
        
        '''
        Generate interferometric A, E and T channel TDI (time-domain) noise, using freqDomain.fundamental_noise_spectrum

        Returns
        ---------    
        h1_noi, h2_noi, h3_noi : float
            Time series data for the three TDI channels

        '''

        cspeed = 3e8 #m/s

        # michelson channels
        tarr, hX, hY, hZ = self.gen_xyz_noise()

        h1_noi = (1.0/3.0)*(2*hX - hY - hZ)
        h2_noi = (1.0/np.sqrt(3.0))*(hZ - hY)
        h3_noi = (1.0/3.0)*(hX + hY + hZ)

        return tarr, h1_noi, h2_noi, h3_noi

    def gen_aet_isgwb(self):
        
        '''
        Generate time isotropic SGWB mock LISA data. The output are time domain TDI data. Michelson 
        channels are manipulated to generate the A, E and T channels. We implement TDI as described in

        http://iopscience.iop.org/article/10.1088/0264-9381/18/17/308


        Returns
        ---------
    
        h1_gw, h2_gw, h3_gw : float
            Time series isotropic stochastic noise for the three TDI channels

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

        Omegaf = (10**self.inj['ln_omega0'])*(freqs/(self.params['fref']))**self.inj['alpha']

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
        Read mldc domain data from an ascii txt file. Since this was used primarily for 
        the MLDC, it assumes that the data is given in X,Y and Z channels.
        Returns
        ---------
    
        h1, h2, h3 : float
            Time series data for the three TDI channels


        '''
        
        hoft = np.loadtxt(self.params['datafile'])
        

        fs_default = 1.0/(hoft[1, 0] - hoft[0, 0])

        ## Read in the duration seconds of data + one segment of buffer
        end_idx = int((self.params['dur'] + self.params['seglen'])*fs_default)

        ## the mldc data is X,Y,Z tdi
        times, h1, h2, h3 = hoft[0:end_idx, 0], hoft[0:end_idx, 1], hoft[0:end_idx, 2], hoft[0:end_idx, 3]

        delt = times[1] - times[0]

        ## Downsample
        if self.params['fs'] < 1.0/delt:

            h1 = sg.decimate(h1, int(1.0/(self.params['fs']*delt)))
            h2 = sg.decimate(h2, int(1.0/(self.params['fs']*delt)))
            h3 = sg.decimate(h3, int(1.0/(self.params['fs']*delt)))
            
            self.params['fs'] = (1.0/delt)/int(1.0/(self.params['fs']*delt))
            times = self.params['fs']*np.arange(0, h1.size, 1)
        else:
            self.params['fs'] = 1.0/delt

        
        return h1, h2, h3


    def tser2fser(self, h1, h2, h3):
        
        '''
        Convert time domain data to fourier domain and return ffts. The convention is that the 
        the ffts are divided by the sampling frequency and corrected for windowing. A hann window 
        is applied by default when moving to the fourier domain. The ffts are also normalized so that
        thier square gives the PSD.

        Parameters
        -----------
        h1,h2, h3 : float
            time series data for the three input channels

        Returns
        ---------
    
        r1, r2, r3 : float
            frequency series data for the three input channels

        fdata : float
            Reference frequency series

        '''

        print ("Calculating fourier spectra... ")

        # Number of segmants
        nsegs = 2*int(np.floor(self.params['dur']/self.params['seglen'])) -1

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
        win_fact = np.mean(hwin**2)

        # We will use 50% overlapping segments
        for ii in range(0, nsegs):

            idxmin = int(0.5*ii*Nperseg)
            idxmax = idxmin + Nperseg

            if hwin.size != h1[idxmin:idxmax].size:
                import pdb; pdb.set_trace()

            
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
        # win_fact is to adjust for hann windowing, sqrt(2) for single sided
        r1 = np.sqrt(2/win_fact)*r1[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))
        r2 = np.sqrt(2/win_fact)*r2[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))
        r3 = np.sqrt(2/win_fact)*r3[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))
        
        np.savez(self.params['out_dir'] + '/' +self.params['input_spectrum'], r1=r1, r2=r2, r3=r3, fdata=fdata)

        return r1, r2, r3, fdata
        