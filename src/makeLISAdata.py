from __future__ import division
import numpy as np
import scipy.signal as sg
from src.movingfreqDomain import movingfreqDomain
from src.freqDomain import freqDomain
from scipy.interpolate import interp1d as intrp
import os
from scipy.signal.windows import nuttall

class LISAdata(freqDomain, movingfreqDomain):

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
        fmin = 0
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

    def freqdomain_gaussianData(self, Sh,freqs, fs=1, dur=1e5):
   
        '''
        Script to generate freq Domain gaussian data of a given spectral density. 

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
        frequency domain gaussian. 
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


        return htilda, fout


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

        # Generate data
        np12 = self.gaussianData(Sp, frange, self.params['fs'], 1.1*self.params['dur'])
        np21 = self.gaussianData(Sp, frange, self.params['fs'], 1.1*self.params['dur'])
        np13 = self.gaussianData(Sp, frange, self.params['fs'], 1.1*self.params['dur'])
        np31 = self.gaussianData(Sp, frange, self.params['fs'], 1.1*self.params['dur'])
        np23 = self.gaussianData(Sp, frange, self.params['fs'], 1.1*self.params['dur'])
        np32 = self.gaussianData(Sp, frange, self.params['fs'], 1.1*self.params['dur'])
        
        na12 = self.gaussianData(Sa, frange, self.params['fs'], 1.1*self.params['dur'])
        na21 = self.gaussianData(Sa, frange, self.params['fs'], 1.1*self.params['dur'])
        na13 = self.gaussianData(Sa, frange, self.params['fs'], 1.1*self.params['dur'])
        na31 = self.gaussianData(Sa, frange, self.params['fs'], 1.1*self.params['dur'])
        na23 = self.gaussianData(Sa, frange, self.params['fs'], 1.1*self.params['dur'])
        na32 = self.gaussianData(Sa, frange, self.params['fs'], 1.1*self.params['dur'])
 
        # time array and time shift array
        tarr =  np.arange(0, 1.1*self.params['dur'], 1.0/self.params['fs'])
        tarr = tarr[0:np12.size]
        delt = tarr[2] - tarr[1]

        # We start with assuming a padding of 20 seconds on the beginning for the 
        # Michelson channels
        ## Using up ten seconds here. 
        ten_idx = int(self.params['fs']*10)

        # To implement TDI we need time shifts of multiples of L.
        tlag  = self.armlength/cspeed

        ## One way dopper channels for each arms. Using up seconds of the pad here for doing tlag
        f21 = intrp(tarr, na21, kind='cubic', fill_value='extrapolate')
        f12 = intrp(tarr, na12, kind='cubic', fill_value='extrapolate')
        f32 = intrp(tarr, na32, kind='cubic', fill_value='extrapolate')
        f23 = intrp(tarr, na23, kind='cubic', fill_value='extrapolate')
        f13 = intrp(tarr, na13, kind='cubic', fill_value='extrapolate')
        f31 = intrp(tarr, na31, kind='cubic', fill_value='extrapolate')

        h12  = np12[ten_idx:] - na12[ten_idx:] + f21(tarr[ten_idx:]-tlag)
        h21  = np21[ten_idx:] + na21[ten_idx:] - f12(tarr[ten_idx:]-tlag) 

        h23  = np23[ten_idx:] - na23[ten_idx:] + f32(tarr[ten_idx:]-tlag)
        h32  = np32[ten_idx:] + na32[ten_idx:] - f23(tarr[ten_idx:]-tlag)

        h31  = np31[ten_idx:] - na31[ten_idx:] + f13(tarr[ten_idx:]-tlag)
        h13  = np13[ten_idx:] + na13[ten_idx:] - f31(tarr[ten_idx:]-tlag)
        
        ## reduce tarr
        tarr = tarr[ten_idx:]

        # The Michelson channels, formed from the doppler channels. Using the other 
        # ten seconds here

        f12 = intrp(tarr, h12, kind='cubic', fill_value='extrapolate')
        f13 = intrp(tarr, h13, kind='cubic', fill_value='extrapolate')
        f23 = intrp(tarr, h23, kind='cubic', fill_value='extrapolate')
        f21 = intrp(tarr, h21, kind='cubic', fill_value='extrapolate')
        f31 = intrp(tarr, h31, kind='cubic', fill_value='extrapolate')
        f32 = intrp(tarr, h32, kind='cubic', fill_value='extrapolate')


        h1 = f12(tarr[ten_idx:]-tlag) + h21[ten_idx:] - \
                f13(tarr[ten_idx:]-tlag)  - h31[ten_idx:]

        h2 = f23(tarr[ten_idx:]-tlag) + h32[ten_idx:] - \
                f21(tarr[ten_idx:]-tlag)  - h12[ten_idx:]

        h3 = f31(tarr[ten_idx:]-tlag)  + h13[ten_idx:] - \
                f32(tarr[ten_idx:]-tlag)  - h23[ten_idx:] 

        
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

        return tarr[ten_idx:], h1, h2, h3



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

        ## Using up ten seconds here. 
        ten_idx = int(self.params['fs']*10)

        # Introduce time series
        tshift = 2*self.armlength/cspeed
        
        f1 = intrp(tarr, hm1, kind='cubic', fill_value='extrapolate')
        f2 = intrp(tarr, hm2, kind='cubic', fill_value='extrapolate')
        f3 = intrp(tarr, hm3, kind='cubic', fill_value='extrapolate')


        hX = hm1[ten_idx:] - f1(tarr[ten_idx:] - tshift)
        hY = hm2[ten_idx:] - f2(tarr[ten_idx:] - tshift)
        hZ = hm3[ten_idx:] - f3(tarr[ten_idx:] - tshift)

        return tarr[ten_idx:], hX, hY, hZ



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

    def gen_mich_isgwb(self, fs=0.25, dur=1e5):
        
        '''
        Generate time isotropic SGWB mock LISA data. The output are time domain data in Michelson
        channels at the three vertices oft the constellation. 

        Returns
        ---------
    
        h1_gw, h2_gw, h3_gw : float
            Time series isotropic stochastic noise for the three Michelson noise

        '''

        # --------------------- Generate Fake Data + Noise -----------------------------
        print("Simulating isgwb data for analysis ...")

        ## If we have a smaller fs than 4 samples a second, we will use 4 Hz as the sampling freq
        ## If the sampling frequency is too low, that doesn't play well with the time-shifts


        dur  = 1.1*self.params['dur']
        seglen =  self.params['seglen']

        # speed of light
        cspeed = 3e8 #m/s

        delf  = 1.0/seglen
        N, Nmid = int(self.params['fs']*seglen), int(0.5*self.params['fs']*seglen)

        tmids = np.arange(0.5*seglen, dur, 0.5*seglen )

        ## Get freqs
        freqs = np.fft.rfftfreq(int(seglen*self.params['fs']), 1.0/self.params['fs'] )
        
        freqs[0] = 1e-15
        #Charactersitic frequency
        fstar = cspeed/(2*np.pi*self.armlength)

        # define f0 = f/2f*
        f0 = freqs/(2*fstar)

        R1 = np.zeros((f0.size, 2), dtype='complex')
        R3 = np.zeros((f0.size, 2), dtype='complex')
        R2 = np.zeros((f0.size, 2), dtype='complex') 

        fidx = np.logical_and(freqs >= self.params['fmin'], freqs <= self.params['fmax'])


        import time; t0 = time.time()
        ## There are the responses for the three arms
        R1, R2, R3 = self.isgwb_mich_strain_response(f0)
        print str(time.time() - t0) + " sec"
                
        #np.exp(2*np.pi*1j*freqs[:, None]*tmids[None, :])

        H0 = 2.2*10**(-18) ## in SI units
        Omegaf = (10**self.inj['ln_omega0'])*(freqs/(self.params['fref']))**self.inj['alpha']

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*freqs**3))*(H0/np.pi)**2
        norms = np.sqrt(self.params['fs']*Sgw*N)/2.0

        h1, h2, h3 = np.array([]), np.array([]), np.array([])

        sin_Nmid, sin_N = np.sin(np.pi*np.arange(0, Nmid)/N), np.sin(np.pi*np.arange(Nmid, N)/N)

        for ii in range(tmids.size):

            dir_phases = np.random.standard_normal(size=R1.shape) + 1j*np.random.standard_normal(size=R1.shape)
            htilda1 = (norms[:, None, None]*R1[:,:,:,0]*dir_phases[:,:,:,0] + norms[:, None, None]*R1[:,:,:,1]*dir_phases[:,:,:,1]).sum(axis=(1, 2))
            htilda2 = (norms[:, None, None]*R2[:,:,:,0]*dir_phases[:,:,:,0] + norms[:, None, None]*R2[:,:,:,1]*dir_phases[:,:,:,1]).sum(axis=(1, 2))
            htilda3 = (norms[:, None, None]*R3[:,:,:,0]*dir_phases[:,:,:,0] + norms[:, None, None]*R3[:,:,:,1]*dir_phases[:,:,:,1]).sum(axis=(1, 2))

            # Generate time series data for the channels
            if np.mod(N, 2) == 0:
                htil1 =  np.concatenate((np.zeros(1), htilda1, np.zeros(1), np.flipud(np.conjugate(htilda1))))
                htil2 =  np.concatenate((np.zeros(1), htilda2, np.zeros(1), np.flipud(np.conjugate(htilda2))))
                htil3 =  np.concatenate((np.zeros(1), htilda3, np.zeros(1), np.flipud(np.conjugate(htilda3))))
            else:
                htil1 =  np.concatenate((np.zeros(1), htilda1, np.conjugate(np.flipud(htilda1))))
                htil2 =  np.concatenate((np.zeros(1), htilda2, np.conjugate(np.flipud(htilda2))))
                htil3 =  np.concatenate((np.zeros(1), htilda3, np.conjugate(np.flipud(htilda3))))

            # Take inverse fft to get time series data
            ht1 = np.real(np.fft.ifft(htil1, N))
            ht2 = np.real(np.fft.ifft(htil2, N))
            ht3 = np.real(np.fft.ifft(htil3, N))

            ht1[0:Nmid],ht2[0:Nmid],ht3[0:Nmid] = sin_Nmid*ht1[0:Nmid], sin_Nmid*ht2[0:Nmid], sin_Nmid*ht3[0:Nmid]
            ht1[Nmid:],ht2[Nmid:],ht3[Nmid:] =  ht1[Nmid:]*sin_N, ht2[Nmid:]*sin_N, ht3[Nmid:]*sin_N

            if h1.size>0:
                h1[-Nmid:], h2[-Nmid:], h3[-Nmid:] = h1[-Nmid:] + ht1[0:Nmid], h2[-Nmid:] + ht2[0:Nmid], h3[-Nmid:] + ht3[0:Nmid] 


            h1, h2, h3 = np.append(h1, ht1[Nmid:]), np.append(h2, ht2[Nmid:]), np.append(h3, ht1[Nmid:])



        import pdb; pdb.set_trace()

        times = np.linspace(0, dur, int(self.params['fs']*dur), endpoint=False)
        return h1, h2, h3, times

    def gen_xyz_isgwb(self):    
        
        '''
        Generate time isotropic SGWB mock LISA data. The output are time domain TDI data. Michelson 
        channels are manipulated to generate the X, Y and. We implement TDI as described in

        http://iopscience.iop.org/article/10.1088/0264-9381/18/17/308


        Returns
        ---------
    
        h1_gw, h2_gw, h3_gw : float
            Time series isotropic stochastic noise for the three TDI channels

        '''


        h1, h2, h3, times = self.gen_mich_isgwb()

        fs_eff = 1.0/(times[1] - times[0])

        # Introduce time series
        tshift = 2*self.armlength/3.0e8
        tshift_idx = int(round(tshift*fs_eff))

        hX = h1[tshift_idx:] - h1[0:-tshift_idx]
        hY = h2[tshift_idx:] - h2[0:-tshift_idx]
        hZ = h3[tshift_idx:] - h3[0:-tshift_idx]

        ## Cut to require size
        N = int(fs_eff*(self.params['dur'] + 10))
        hX, hY, hZ = hX[0:N], hZ[0:N], hY[0:N]



        return hX, hY, hZ, times

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
        hX, hY, hZ, times = self.gen_xyz_isgwb()

        hA = (1.0/3)*(2*hX - hY + hZ)
        hE = (1.0/np.sqrt(3.0))*(hZ - hY)
        hT = (1.0/3)*(hX + hY + hZ)

        return hA, hE, hT, times

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


        ## Check if the requested sampel rate is consistant
        if self.params['fs'] != 1.0/delt:
            self.params['fs'] = 1.0/delt
        
        return h1, h2, h3, times


    def tser2fser(self, h1, h2, h3, timearray):
        
        '''
        Convert time domain data to fourier domain and return ffts. The convention is that the 
        the ffts are divided by the sampling frequency and corrected for windowing. A hann window 
        is applied by default when moving to the fourier domain. The ffts are also normalized so that
        thier square gives the PSD.

        Parameters
        -----------
        h1, h2, h3 : float
            time series data for the three input channels
            
        timearray : float
            times corresponding to data in h1, h2, h3

        Returns
        ---------
    
        r1, r2, r3 : float
            frequency series data for the three input channels

        fdata : float
            Reference frequency series
            
        tsegstart : float
            Segmented time array giving segment start points
            
        tsegmid : float
            Segmented time array giving segment midpoints


        '''

        print ("Calculating fourier spectra... ")

        # Number of segmants
        nsegs = int(np.floor(self.params['dur']/self.params['seglen'])) -1

        Nperseg=int(self.params['fs']*self.params['seglen'])

        # Apply band pass filter
        '''
        order = 8
        zz, pp, kk = sg.butter(order, [0.5*self.params['fmin']/(self.params['fs']/2), 0.4*self.params['fs']/(self.params['fs']/2)], btype='bandpass', output='zpk')
        sos = sg.zpk2sos(zz, pp, kk)

        h1 = sg.sosfiltfilt(sos, h1)
        h2 = sg.sosfiltfilt(sos, h2)
        h3 = sg.sosfiltfilt(sos, h3)
        '''

        fftfreqs = np.fft.rfftfreq(Nperseg, 1.0/self.params['fs'])


        # Map of spectrum
        r1 = np.zeros((fftfreqs.size, nsegs), dtype='complex')
        r2 = np.zeros((fftfreqs.size, nsegs), dtype='complex')
        r3 = np.zeros((fftfreqs.size, nsegs), dtype='complex')

        
        # Hann Window
        hwin = np.hanning(Nperseg)
        #hwin = nuttall(Nperseg)
        win_fact = np.mean(hwin**2)


        zpad = np.zeros(Nperseg)
        # No overlapping segments

        ## Initiate time segment arrays
        tsegstart = np.zeros(nsegs)
        tsegmid = np.zeros(nsegs)

        # We will use 50% overlapping segments
        for ii in range(0, nsegs):

            idxmin = int(0.5*ii*Nperseg)
            idxmax = idxmin + Nperseg
            idxmid = idxmin + int(Nperseg/2)
            if hwin.size != h1[idxmin:idxmax].size:
                import pdb; pdb.set_trace()

            
            r1[:, ii] =   np.fft.rfft(hwin*h1[idxmin:idxmax], axis=0)
            r2[:, ii] =   np.fft.rfft(hwin*h2[idxmin:idxmax], axis=0)
            r3[:, ii] =   np.fft.rfft(hwin*h3[idxmin:idxmax], axis=0)


            ## There's probably a more pythonic way of doing this, but it'll work for now.
            tsegstart[ii] = timearray[idxmin]
            tsegmid[ii] = timearray[idxmid]

        # "Cut" to desired frequencies
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


        return r1, r2, r3, fdata, tsegstart, tsegmid

        