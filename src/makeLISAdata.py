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

        Sp, Sa = self.fundamental_noise_spectrum(frange)

        # To implement TDI we need time shifts of multiples of L.
        tlag  = self.armlength/cspeed

        ## If we have a smaller fs than tlag/2, we will use pick a factor of 2 greater than that as the sampling freq
        if self.params['fs'] < 2.0/tlag:
            fs_eff = 2**(np.floor(np.log2(2.0/tlag)))
    

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

        tlag_idx = int(round(tlag/delt))

        if 1.0/delt != fs_eff:
            raise ValueError('Time series generated not consistant with the sampling frequency')
            

        # shifted time series
        tshift = tarr - tlag

        ## The three dopper channels for each arms
        h12  = np12[tlag_idx:] - na12[tlag_idx:] + na21[0:-tlag_idx] #np.interp(tshift, tarr, na21, left=na21[0])
        h21  = np21[tlag_idx:] + na21[tlag_idx:] - na12[0:-tlag_idx] #np.interp(tshift, tarr, na12, left=na12[0])

        h23  = np23[tlag_idx:] - na23[tlag_idx:] + na32[0:-tlag_idx] #np.interp(tshift, tarr, na32, left=na32[0])
        h32  = np32[tlag_idx:] + na32[tlag_idx:] - na23[0:-tlag_idx] #np.interp(tshift, tarr, na23, left=np23[0])

        h31  = np31[tlag_idx:] - na31[tlag_idx:] + na13[0:-tlag_idx] #np.interp(tshift, tarr, na13, left=na13[0])
        h13  = np13[tlag_idx:] + na13[tlag_idx:] - na31[0:-tlag_idx] #np.interp(tshift, tarr, na31, left=na31[0])
        
 

        # The michelson channels are formed from the doppler channels
        #h1 = np.interp(tshift, tarr, h12, left=h12[0]) + h21 #-\
        #np.interp(tshift, tarr, h13, left=h13[0]) - h31
        h1 = h12[0:-tlag_idx] + h21[tlag_idx:]

        #h2 = np.interp(tshift, tarr, h23, left=h23[0]) + h32 -\
        #np.interp(tshift, tarr, h21, left=h21[0]) - h12

        #h3 = np.interp(tshift, tarr, h31, left=h31[0]) + h13 -\
        #np.interp(tshift, tarr, h32, left=h32[0]) - h23

        from scipy.signal import welch
        import matplotlib.pyplot as plt
        f12, S12 = welch(h12, fs=fs_eff, nperseg=int(5e4*fs_eff))
        plt.loglog(f12, S12, label='data', alpha=0.5)
        plt.plot(frange, 2*Sp + 4*Sa + 4*Sa*np.cos(4*f0)*np.cos(4*f0),'--' ,label='true, expected', alpha=0.7)
        plt.plot(frange, 2*Sp + 8*Sa ,':',label='guess')
        plt.plot(frange, 2*Sp, ':',label=' position terms only')
        plt.legend()
        plt.savefig('test.png', dpi=200)
        plt.close()
        import pdb; pdb.set_trace()


        return h1, h2, h3

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
        hm1, hm2, hm3 = self.gen_michelson_noise()

        ## In time domain we need to do some interpolation to get to the appropriate TDI channels
        tarr = np.linspace(0, self.params['dur'] , num= hm1.size, endpoint=False)

        # shifted time series
        tshift = tarr - 2*self.armlength/cspeed

        hX = hm1 - np.interp(tshift, tarr, hm1, left=hm1[0])
        hY = hm2 - np.interp(tshift, tarr, hm2, left=hm2[0])
        hZ = hm3 - np.interp(tshift, tarr, hm3, left=hm3[0])

        return hX, hY, hZ



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
        # 0.375 is to adjust for hann windowing, sqrt(2) for single sided
        r1 = np.sqrt(2/0.375)*r1[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))
        r2 = np.sqrt(2/0.375)*r2[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))
        r3 = np.sqrt(2/0.375)*r3[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))
        
        np.savez(self.params['out_dir'] + '/' +self.params['input_spectrum'], r1=r1, r2=r2, r3=r3, fdata=fdata)

        return r1, r2, r3, fdata
        