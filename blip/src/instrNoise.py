from __future__ import division
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import interp1d as intrp

class instrNoise():

    '''
    Class with methods to calcualte instrumental noises
    '''

    def fundamental_noise_spectrum(self, freqs, Np=4e-41, Na=1.44e-48):

        '''
        Creates a frequency array of fundamentla noise estimates for lisa. Currently we consisder only contain only
        position and acceleration noise sources. The default values are specifications pulled from 2017 Lisa proposal
        noise estimations.

        Parameters
        -----------

        freqs   : float
            A numpy array of frequencies

        Np (optional) : float
            Position noise value

        Na (optional) : float
            Acceleration noise level


        Returns
        ---------

        Sp, Sa   :   float
            Frequencies array for position and acceleration noises for each satellite
        '''

        Sp = Np*(1 + (2e-3/freqs)**4)
        Sa = Na*(1 + 16e-8/freqs**2)*(1 + (freqs/8e-3)**4)*(1.0/(2*jnp.pi*freqs)**4)

        return Sp, Sa

    def aet_noise_spectrum(self, freqs,f0, Np=4e-41, Na=1.44e-48):

        '''
        Calculates A, E, and T channel noise spectra for a stationary lisa. Following the defintions in
        Adams & Cornish, http://iopscience.iop.org/article/10.1088/0264-9381/18/17/308


        Parameters
        -----------

        freqs   : float
            A numpy array of frequencies

        Np (optional) : float
            Position noise value

        Na (optional) : float
            Acceleration noise level


        Returns
        ---------

        SAA, SEE, STT   :   float
            Frequencies arrays with the noise PSD for the A, E and T TDI channels


        '''

        # Get Sp and Sa
        C_xyz = self.xyz_noise_spectrum(freqs, f0, Np, Na)

        ## Upnack xyz matrix to make assembling the aet matrix easier
        CXX, CYY, CZZ = C_xyz[0, 0], C_xyz[1, 1], C_xyz[2, 2]
        CXY, CXZ, CYZ = C_xyz[0, 1], C_xyz[0, 2], C_xyz[1, 2]


        ## construct AET matrix elements
        CAA = (1/9) * (4*CXX + CYY + CZZ - 2*CXY - 2*jnp.conj(CXY) - 2*CXZ - 2*jnp.conj(CXZ) + \
                        CYZ  + jnp.conj(CYZ))

        CEE = (1/3) * (CZZ + CYY - CYZ - jnp.conj(CYZ))

        CTT = (1/9) * (CXX + CYY + CZZ + CXY + jnp.conj(CXY) + CXZ + jnp.conj(CXZ) + CYZ + jnp.conj(CYZ))

        CAE = (1/(3*jnp.sqrt(3))) * (CYY - CZZ - CYZ + jnp.conj(CYZ) + 2*CXZ - 2*CXY)

        CAT = (1/9) * (2*CXX - CYY - CZZ + 2*CXY - jnp.conj(CXY) + 2*CXZ - jnp.conj(CXZ) - CYZ - jnp.conj(CYZ))

        CET = (1/(3*jnp.sqrt(3))) * (CZZ - CYY - CYZ + jnp.conj(CYZ) + jnp.conj(CXZ) - jnp.conj(CXY))

        C_aet = jnp.array([ [CAA, CAE, CAT] , \
                                    [jnp.conj(CAE), CEE, CET], \
                                    [jnp.conj(CAT), jnp.conj(CET), CTT] ])


        return C_aet

    def xyz_noise_spectrum(self, freqs,f0, Np=4e-41, Na=1.44e-48):

        '''
        Calculates X,Y,Z channel noise spectra for a stationary lisa. Following the defintions in
        Adams & Cornish, http://iopscience.iop.org/article/10.1088/0264-9381/18/17/308


        Parameters
        -----------

        freqs   : float
            A numpy array of frequencies

        Np (optional) : float
            Position noise value

        Na (optional) : float
            Acceleration noise level


        Returns
        ---------

        SAA, SEE, STT   :   float
            Frequencies arrays with the noise PSD for the A, E and T TDI channels


        '''
        C_mich = self.mich_noise_spectrum(freqs, f0, Np, Na)

        ## Noise spectra of the X, Y and Z channels
        C_xyz =  4 * jnp.sin(2*f0)**2 * C_mich

        return C_xyz

    def mich_noise_spectrum(self, freqs,f0, Np=4e-41, Na=1.44e-48):

        '''
        Calculates michelson channel noise spectra for a stationary lisa. Following the defintions in
        Adams & Cornish, http://iopscience.iop.org/article/10.1088/0264-9381/18/17/308. We assume that
        there is no phase noise.


        Parameters
        -----------

        freqs   : float
            A numpy array of frequencies

        Np (optional) : float
            Position noise value

        Na (optional) : float
            Acceleration noise level


        Returns
        ---------

        SAA, SEE, STT   :   float
            Frequencies arrays with the noise PSD for the A, E and T TDI channels


        '''

        # Get Sp and Sa
        Sp, Sa = self.fundamental_noise_spectrum(freqs, Np, Na)


        ## Noise spectra of the michelson channels
        S_auto  = 4.0 * (2.0 * Sa * (1.0 + (jnp.cos(2*f0))**2)  + Sp)
        S_cross =  (-2 * Sp - 8 * Sa) * jnp.cos(2*f0)

        C_mich = jnp.array([[S_auto, S_cross, S_cross], [S_cross, S_auto, S_cross], [S_cross, S_cross, S_auto]])

        return C_mich


    def gen_michelson_noise(self):

        '''
        Generate interferometric michelson (time-domain) noise, using freqDomain.fundamental_noise_spectrum

        Returns
        ---------
        h1, h2, h3 : float
            Time series data for the three michelson channels
        '''

        # --------------------- Generate Fake Noise -----------------------------
        print("Simulating instrumental noise ...")

       # speed of light
        cspeed = 3e8 #m/s
        delf  = 1.0/self.params['dur']
        frange = np.arange(self.params['fmin'], self.params['fmax'], delf) # in Hz

        Sp, Sa = self.fundamental_noise_spectrum(frange, Np=10**self.injvals['log_Np'], Na=10**self.injvals['log_Na'])

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

        '''

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

        # michelson channels
        tarr, hX, hY, hZ = self.gen_xyz_noise()

        h1_noi = (1.0/3.0)*(2*hX - hY - hZ)
        h2_noi = (1.0/np.sqrt(3.0))*(hZ - hY)
        h3_noi = (1.0/3.0)*(hX + hY + hZ)

        return tarr, h1_noi, h2_noi, h3_noi


    def gen_noise_cov(self):

        '''
        Generate interferometric (time-domain) noise, using a frequency domain covariance
        spectrum matrix rather than time delays in time domain.
        ---------

        h1_noi, h2_noi, h3_noi : float
            Time series data for the three TDI channels
        '''

        fstar = 3e8/(2*np.pi*self.armlength)
        N = int(self.params['fs']*self.params['dur'])

        frange = np.fft.rfftfreq(N, 1.0/self.params['fs'])[1:]
        frange = frange[frange <= self.params['fmax']]
        frange = frange[frange >= self.params['fmin']]

        f0 = frange/(2*fstar)


        C_xyz = self.xyz_noise_spectrum(frange, f0, Np=10**self.inj['log_Np'], Na=10**self.inj['log_Na'])

        ## Cholesky decomposition to get the "sigma" matrix
        L_cholesky = np.sqrt(self.params['fs'] * N/4.0) *  np.linalg.cholesky(np.moveaxis(C_xyz, -1, 0))

        ## generate standard normal complex data frist
        z_norm = np.random.normal(size=(3, frange.size)) + 1j * np.random.normal(size=(3, frange.size))

        ## initialize a new scaled array. The data in z_norm will be rescaled into z_scale
        z_scale = np.zeros(z_norm.shape, dtype='complex')

        for ii in range(frange.size):
            z_scale[:, ii] = np.matmul(L_cholesky[ii, :, :], z_norm[:, ii])


        ## The three channels : concatenate with norm at f = 0 to be zero
        htilda1  = np.concatenate([ [0], z_scale[0, :]])
        htilda2  = np.concatenate([ [0], z_scale[1, :]])
        htilda3  = np.concatenate([ [0], z_scale[2, :]])


        # Take inverse fft to get time series data
        h1 = np.fft.irfft(htilda1, N)
        h2 = np.fft.irfft(htilda2, N)
        h3 = np.fft.irfft(htilda3, N)

        tarr =  np.arange(0, self.params['dur'], 1.0/self.params['fs'])


        return tarr, h1, h2, h3
    
    
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
            numFreqs = int(N/2 - 1)
        else:
            numFreqs = int((N-1)/2)

        # We will make an array of the desired frequencies
        fmin = 0
        fmax = np.around(dur*fs/2)/dur

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
        fmin = 1/dur
        fmax = np.around(dur*fs/2)/dur

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