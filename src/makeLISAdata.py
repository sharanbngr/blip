import numpy as np
import scipy.signal as sg
from src.instrNoise import instrNoise
from src.geometry import geometry
from src.sph_geometry import sph_geometry
from src.populations import populations
from scipy.interpolate import interp1d as intrp
import matplotlib.pyplot as plt
import healpy as hp
from astropy import units as u
from astropy import coordinates as cc
from astropy.coordinates import SkyCoord
from math import pi
import os
import legwork as lw

class LISAdata(geometry, sph_geometry, instrNoise, populations):

    '''
    Class for lisa data. Includes methods for generation of gaussian instrumental noise, and generation
    of isotropic stochastic background. Any eventually signal models should be added as methods here. This
    has the Antennapatterns class as a super class.
    '''

    def __init__(self, params, inj):
        self.params = params
        self.inj = inj
        self.armlength = 2.5e9 ## armlength in meters
        geometry.__init__(self)
        sph_geometry.__init__(self)


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
            numFreqs = int(N/2 - 1)
        else:
            numFreqs = int((N-1)/2)

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
        print("Simulating instrumental noise ...")

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

        cspeed = 3e8 #m/s

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

        cspeed = 3e8 #m/s
        delf  = 1.0/self.params['dur']
        fstar = 3e8/(2*np.pi*self.armlength)
        N = int(self.params['fs']*self.params['dur'])

        frange = np.fft.rfftfreq(N, 1.0/self.params['fs'])[1:]
        frange = frange[frange <= self.params['fmax']]
        frange = frange[frange >= self.params['fmin']]

        f0 = frange/(2*fstar)

        #Sp, Sa = self.fundamental_noise_spectrum(frange, Np=10**self.inj['log_Np'], Na=10**self.inj['log_Na'])

        C_xyz = self.xyz_noise_spectrum(frange, f0, Np=10**self.inj['log_Np'], Na=10**self.inj['log_Na'])

        ## Cholesky decomposition to get the "sigma" matrix
        L_cholesky = np.sqrt(self.params['fs'] * N/4.0) *  np.linalg.cholesky(np.moveaxis(C_xyz, -1, 0))

        #for ii in range(C_xyz.shape[-1]):
        #    np.linalg.cholesky(C_xyz[:, :, ii])
        #    print(str(ii) + '/' + str(C_xyz.shape[-1]))

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

    def add_sgwb_data(self, tbreak = 0.0):

        cspeed = 3e8 #m/s

        ## define the splice segment duration
        tsplice = 1e4
        delf  = 1.0/tsplice

        ## the segments to be splices are half-overlapping
        nsplice = 2*int(self.params['dur']/tsplice) + 1

        ## arrays of segmnent start and mid times
        #tmids = (tsplice/2.0) * np.arange(nsplice) + (tsplice/2.0)

        ## arrays of segmnent start and mid times
        tmids = self.params['tstart'] + tbreak +  (tsplice/2.0) * np.arange(nsplice) + (tsplice/2.0)

        ## Number of time-domain points in a splice segment
        N = int(self.params['fs']*tsplice)
        halfN = int(0.5*N)

        ## leave out f = 0
        frange = np.fft.rfftfreq(N, 1.0/self.params['fs'])[1:]

        ## the charecteristic frequency of LISA, and the scaled frequency array
        fstar = 3e8/(2*np.pi*self.armlength)
        f0 = frange/(2*fstar)

        ## Response matrix : shape (3 x 3 x freq x time) if isotropic
        response_mat = self.add_astro_signal(f0, tmids)

        ## Cholesky decomposition to get the "sigma" matrix
        H0 = 2.2*10**(-18) ## in SI units
        
        if self.inj['injtype'] == 'dwd_fg' or self.inj['injtype'] == 'dwd_sdg':
            if self.inj['fg_spectrum'] == 'truncated':
                ## frequency cutoff based on Fig 1. of Breivik et al (2020)
                fcutoff = 10**self.inj['log_fcut']
                fcut = (frange < fcutoff)*frange
                Omegaf = (10**self.inj['ln_omega0'])*(fcut/(self.params['fref']))**self.inj['alpha']
            elif self.inj['fg_spectrum'] == 'powerlaw':
                Omegaf = (10**self.inj['ln_omega0'])*(frange/(self.params['fref']))**self.inj['alpha']
            elif self.inj['fg_spectrum'] == 'broken_powerlaw':
                alpha_2 = self.inj['alpha1'] - 0.667
                Omegaf = ((10**self.inj['log_A1'])*(frange/self.params['fref'])**self.inj['alpha1'])/(\
                         1 + (10**self.inj['log_A2'])*(frange/self.params['fref'])**alpha_2)
#                fcutoff = 10**self.inj['log_fcut']
#                lowfilt = (frange < fcutoff)
#                highfilt = np.invert(lowfilt)
#                Omega_cut = (10**self.inj['ln_omega0'])*(fcutoff/(self.params['fref']))**self.inj['alpha'] 
#                Omegaf = lowfilt*(10**self.inj['ln_omega0'])*(frange/(self.params['fref']))**self.inj['alpha'] + \
#                         highfilt*Omega_cut*(frange/fcutoff)**self.inj['alpha2']
                
#                 plt.figure()
#                 det_PSD = lw.psd.lisa_psd(frange*u.Hz,t_obs=self.params['dur']*u.s,confusion_noise=None)
#                 plt.plot(frange,det_PSD,color='black',label='Detector PSD')
#                 plt.plot(frange,Omegaf*(3/(4*frange**3))*(H0/np.pi)**2 ,color='slategray',alpha=0.5,label='Foreground')
#                 plt.legend(loc='upper right')
#                 plt.xscale('log')
#                 plt.yscale('log')
# #                plt.ylim(1e-43,1e-31)
#                 # plt.xlim(1e-4,1e-2)
#                 plt.xlabel('Frquency [Hz]')
#                 plt.ylabel('GW Power Spectral Density [Hz$^{-1}$]')
#                 plt.savefig(self.params['out_dir'] + '/fg_bpl_test.png', dpi=150)
#                 plt.close()
            elif self.inj['fg_spectrum'] == 'population':
                print("Constructing foreground spectrum from DWD population...")
                ## factor of two b/c (h_A,h_A*)~h^2~1/2 * S_A
                ## additional factor of 2 b/c S_GW = 2 * S_A
                Sgw = self.pop2spec(self.inj['popfile'],frange,self.params['dur']*u.s,names=self.inj['columns'],sep=self.inj['delimiter'])*4 
                
#                 plt.figure()
#                 det_PSD = lw.psd.lisa_psd(frange*u.Hz,t_obs=self.params['dur']*u.s,confusion_noise=None,approximate_R=True)
#                 response_lw = lw.psd.approximate_response_function(frange,fstar=1e-3)
#                 plt.plot(frange,det_PSD,color='black',label='Detector PSD')
#                 plt.plot(frange,response_lw*Sgw,color='slategray',alpha=0.5,label='Foreground')
#                 plt.legend(loc='upper right')
#                 plt.xscale('log')
#                 plt.yscale('log')
# #                plt.ylim(1e-43,1e-31)
#                 # plt.xlim(1e-4,1e-2)
#                 plt.xlabel('Frquency [Hz]')
#                 plt.ylabel('GW Power Spectral Density [Hz$^{-1}$]')
#                 plt.savefig(self.params['out_dir'] + '/fg_test.png', dpi=150)
#                 plt.close()
                
            else:
                raise TypeError('Unknown foreground model chosen. Supported models: powerlaw, broken_powerlaw, truncated, population.')
            # Spectrum of the SGWB from Omegaf (population version goes directly to the spectrum from binary strains and frequencies)
            if self.inj['fg_spectrum'] != 'population':
                Sgw = Omegaf*(3/(4*frange**3))*(H0/np.pi)**2    
        else:
            Omegaf = (10**self.inj['ln_omega0'])*(frange/(self.params['fref']))**self.inj['alpha']
            # Spectrum of the SGWB
            Sgw = Omegaf*(3/(4*frange**3))*(H0/np.pi)**2
        
#         ## added or statement for dwd_sdg -SMR
#         if self.inj['injtype'] == 'dwd_fg' or 'dwd_sdg':
#             if self.inj['fg_spectrum'] == 'truncated':
#                 ## frequency cutoff based on Fig 1. of Breivik et al (2020)
#                 fcutoff = self.inj['fcutoff']
#                 fcut = (frange < fcutoff)*frange
#                 Omegaf = (10**self.inj['ln_omega0'])*(fcut/(self.params['fref']))**self.inj['alpha']
#             if self.inj['fg_spectrum'] == 'powerlaw':
#                 Omegaf = (10**self.inj['ln_omega0'])*(frange/(self.params['fref']))**self.inj['alpha']
#         else:
#             Omegaf = (10**self.inj['ln_omega0'])*(frange/(self.params['fref']))**self.inj['alpha']

#         # Spectrum of the SGWB
#         Sgw = Omegaf*(3/(4*frange**3))*(H0/np.pi)**2

        ## the spectrum of the frequecy domain gaussian for ifft
        norms = np.sqrt(self.params['fs']*Sgw*N)/2

        ## index array for one segment
        t_arr = np.arange(N)

        ## the window for splicing
        splice_win = np.sin(np.pi * t_arr/N)

        # deals with projection parameter to use in the hp.mollview functions below
        if self.params['projection'] is None:
            coord = 'E'
        elif self.params['projection']=='G' or self.params['projection']=='C':
            coord = ['E',self.params['projection']]
        elif self.params['projection']=='E':
            coord = self.params['projection']
        else:  
            raise TypeError('Invalid specification of projection, projection can be E, G, or C')

        ## Loop over splice segments
        for ii in range(nsplice):

            if self.inj['injtype'] == 'isgwb':
                ## move frequency to be the zeroth-axis, then cholesky decomp
                L_cholesky = norms[:, None, None] *  np.linalg.cholesky(np.moveaxis(response_mat[:, :, :, ii], -1, 0))

            elif self.inj['injtype'] == 'sph_sgwb':

                if ii == 0:

                    ## need to set up a few things before doing the spherical harmonic inj

                    ## extract alms
                    self.alms_inj = self.blm_2_alm(self.inj['blms'])

                    ## normalize
                    self.alms_inj = self.alms_inj/(self.alms_inj[0] * np.sqrt(4*np.pi))

                    ## extrct only the non-negative components
                    alms_non_neg = self.alms_inj[0:hp.Alm.getsize(self.almax)]

                    Omega_1mHz = 10**(self.inj['ln_omega0']) * (1e-3/25)**(self.inj['alpha'])

                    ## response matrix summed over Ylms
                    summ_response_mat = np.einsum('ijklm,m', response_mat, alms_inj)

                    # converts alm_inj into a healpix max to be plotted and saved
                    # Plot with twice the analysis nside for better resolution
                    skymap_inj = hp.alm2map(alms_non_neg, 2*self.params['nside'])

                    Omegamap_inj = Omega_1mHz * skymap_inj

                    hp.graticule()
                    hp.mollview(Omegamap_inj, coord=coord, title='Injected angular distribution map $\Omega (f = 1 mHz)$')

                    ## generating skymap, switches to specified projection if not ecliptic
                    # if coord=='E':
                    #     hp.mollview(Omegamap_inj, coord=coord, title='Injected angular distribution map $\Omega (f = 1 mHz)$')
                    # else:
                    #     hp.mollview(Omegamap_inj, coord=['E',coord], title='Injected angular distribution map $/Omega (f = 1 mHz)$')

                    plt.savefig(self.params['out_dir'] + '/inj_skymap.png', dpi=150)
                    print('saving injected skymap at ' +  self.params['out_dir'] + '/inj_skymap.png')
                    plt.close()
                    
                

                ## move frequency to be the zeroth-axis, then cholesky decomp
                L_cholesky = norms[:, None, None] *  np.linalg.cholesky(np.moveaxis(summ_response_mat[:, :, :, ii], -1, 0))

            elif self.inj['injtype'] == 'dwd_fg':
                
                ## for the toy model foreground
                if self.inj['fg_type'] == 'breivik2020':

                    if ii == 0:
    
                        ## need to set up a few things before doing the spherical harmonic inj
                        
                        ## generate skymap
                        DWD_FG_map, log_DWD_FG_map = self.generate_galactic_foreground(self.inj['rh'], self.inj['zh'])
                        ## convert to blms
                        DWD_FG_sph = self.sph_galactic_foreground(DWD_FG_map)
                        ## extract alms
                        self.alms_inj = self.blm_2_alm(DWD_FG_sph)
    
                        ## normalize
                        self.alms_inj = self.alms_inj/(self.alms_inj[0] * np.sqrt(4*np.pi))
    
                        ## extrct only the non-negative components
                        alms_non_neg = self.alms_inj[0:hp.Alm.getsize(self.almax)]
    
                        if 'fg_spectrum' in self.inj.keys():
                            if self.inj['fg_spectrum']=='broken_powerlaw':
                                alpha_1 = self.inj['alpha1']
                                log_A1 = self.inj['log_A1']
                                alpha_2 = self.inj['alpha1'] - 0.667
                                log_A2 = self.inj['log_A2']
                                Omega_1mHz= ((10**log_A1)*(1e-3/self.params['fref'])**alpha_1)/(1 + (10**log_A2)*(1e-3/self.params['fref'])**alpha_2)
                            elif self.inj['fg_spectrum']=='population':
                                ## need to grab the population Sgw at ~1 mHz, preferably without calling pop2spec again
                                Omega_1mHz = Sgw[np.argmin(np.abs(frange - 1e-3))]/((3/(4*(1e-3)**3))*(H0/np.pi)**2)
                            else:
                                if self.inj['fg_spectrum'] != 'powerlaw':
                                    print("Unknown spectral model. Defaulting to power law...")
                                Omega_1mHz = 10**(self.inj['ln_omega0']) * (1e-3/self.params['fref'])**(self.inj['alpha'])
                        else:
                            print("Warning: defaulting to power law spectral model. This may result in unintended behavior.")
                            Omega_1mHz = 10**(self.inj['ln_omega0']) * (1e-3/self.params['fref'])**(self.inj['alpha'])
                        ## response matrix summed over Ylms
                        summ_response_mat = np.einsum('ijklm,m', response_mat, self.alms_inj)
    
                        # converts alm_inj into a healpix map to be plotted and saved
                        # Plot with twice the analysis nside for better resolution
                        skymap_inj = hp.alm2map(alms_non_neg, 2*self.params['nside'])
    
                        Omegamap_inj = Omega_1mHz * skymap_inj
    
                        
                        hp.mollview(Omegamap_inj, coord=coord, title='Injected angular distribution map $\Omega (f = 1 mHz)$')
                        hp.graticule()
                        
                        plt.savefig(self.params['out_dir'] + '/inj_skymap.png', dpi=150)
                        print('saving injected skymap at ' +  self.params['out_dir'] + '/inj_skymap.png')
                        plt.close()
                        
                        
                        hp.mollview(DWD_FG_map, coord=coord, title='Simulated DWD Foreground skymap')
                        hp.graticule()
                        plt.savefig(self.params['out_dir'] + '/pre_inj_skymap.png', dpi=150)
                        print('saving simulated skymap at ' +  self.params['out_dir'] + '/pre_inj_skymap.png')
                        plt.close()
                        hp.mollview(hp.alm2map(DWD_FG_sph, 2*self.params['nside']), coord=coord, title='Simulated DWD Foreground alm map')
                        hp.graticule()
                        plt.savefig(self.params['out_dir'] + '/pre_inj_almmap.png', dpi=150)
                        print('saving simulated skymap at ' +  self.params['out_dir'] + '/pre_inj_almmap.png')
                        plt.close()
    
                    ## move frequency to be the zeroth-axis, then cholesky decomp
                    L_cholesky = norms[:, None, None] *  np.linalg.cholesky(np.moveaxis(summ_response_mat[:, :, :, ii], -1, 0))
                    
                elif self.inj['fg_type'] == 'population':
                    if ii == 0:
    
                        ## need to set up a few things before doing the spherical harmonic inj
                        
                        ## generate skymap
                        print("Constructing foreground skymap from DWD population...")
                        DWD_FG_map, log_DWD_FG_map = self.pop2map(self.inj['popfile'],2*self.params['nside'],self.params['dur']*u.s,
                                                                  self.params['fmin'],self.params['fmax'],names=self.inj['columns'],sep=self.inj['delimiter'])
                        ## convert to blms
                        DWD_FG_sph = self.sph_galactic_foreground(DWD_FG_map)
                        ## extract alms
                        self.alms_inj = self.blm_2_alm(DWD_FG_sph)
    
                        ## normalize
                        self.alms_inj = self.alms_inj/(self.alms_inj[0] * np.sqrt(4*np.pi))
    
                        ## extrct only the non-negative components
                        alms_non_neg = self.alms_inj[0:hp.Alm.getsize(self.almax)]
    
#                        Omega_1mHz = 10**(self.inj['ln_omega0']) * (1e-3/25)**(self.inj['alpha'])
    
                        ## response matrix summed over Ylms
                        summ_response_mat = np.einsum('ijklm,m', response_mat, self.alms_inj)
    
                        # converts alm_inj into a healpix map to be plotted and saved
                        # Plot with twice the analysis nside for better resolution
                        # get Omega at 1mHz
                        if 'fg_spectrum' in self.inj.keys():
                            if self.inj['fg_spectrum']=='broken_powerlaw':
                                alpha_1 = self.inj['alpha1']
                                log_A1 = self.inj['log_A1']
                                alpha_2 = self.inj['alpha1'] - 0.667
                                log_A2 = self.inj['log_A2']
                                Omega_1mHz= ((10**log_A1)*(1e-3/self.params['fref'])**alpha_1)/(1 + (10**log_A2)*(1e-3/self.params['fref'])**alpha_2)
                            elif self.inj['fg_spectrum']=='population':
                                ## need to grab the population Sgw at ~1 mHz, preferably without calling pop2spec again
                                Omega_1mHz = Sgw[np.argmin(np.abs(frange - 1e-3))]/((3/(4*(1e-3)**3))*(H0/np.pi)**2)
                            else:
                                if self.params['fg_spectrum'] != 'powerlaw':
                                    print("Unknown spectral model. Defaulting to power law...")
                                Omega_1mHz = 10**(self.inj['ln_omega0']) * (1e-3/self.params['fref'])**(self.inj['alpha'])
                        else:
                            print("Warning: defaulting to power law spectral model. This may result in unintended behavior.")
                            Omega_1mHz = 10**(self.inj['ln_omega0']) * (1e-3/self.params['fref'])**(self.inj['alpha'])
                        
                        skymap_inj = hp.alm2map(alms_non_neg, 2*self.params['nside'])
                        Omegamap_inj = Omega_1mHz * skymap_inj
                        hp.mollview(Omegamap_inj, coord=coord, title='Injected angular distribution map $\Omega (f = 1 mHz)$')
                        hp.graticule()
                        plt.savefig(self.params['out_dir'] + '/inj_skymap.png', dpi=150)
                        print('saving injected skymap at ' +  self.params['out_dir'] + '/inj_skymap.png')
                        plt.close()
                        
                        hp.mollview(DWD_FG_map, coord=coord, title='Population-derived DWD Foreground skymap')
                        hp.graticule()
                        plt.savefig(self.params['out_dir'] + '/pre_inj_skymap.png', dpi=150)
                        print('saving simulated skymap at ' +  self.params['out_dir'] + '/pre_inj_skymap.png')
                        plt.close()
                        hp.mollview(hp.alm2map(DWD_FG_sph, 2*self.params['nside']), coord=coord, title='Population-derived DWD Foreground alm map')
                        hp.graticule()
                        plt.savefig(self.params['out_dir'] + '/pre_inj_almmap.png', dpi=150)
                        print('saving simulated skymap at ' +  self.params['out_dir'] + '/pre_inj_almmap.png')
                        plt.close()
    
                    ## move frequency to be the zeroth-axis, then cholesky decomp
                    L_cholesky = norms[:, None, None] *  np.linalg.cholesky(np.moveaxis(summ_response_mat[:, :, :, ii], -1, 0))
                    
                else:
                    raise TypeError("Unknown foreground injection type ('fg_type'). Can be 'breivik2020' or 'population'.")
#             elif self.inj['injtype'] == 'dwd_fg':

#                 if ii == 0:

#                     ## need to set up a few things before doing the spherical harmonic inj
                    
#                     ## generate skymap
#                     DWD_FG_map, log_DWD_FG_map = self.generate_galactic_foreground(self.inj['rh'], self.inj['zh'])
#                     ## convert to blms
#                     DWD_FG_sph = self.sph_galactic_foreground(DWD_FG_map)
#                     ## extract alms
#                     self.alms_inj = self.blm_2_alm(DWD_FG_sph)

#                     ## normalize
#                     self.alms_inj = self.alms_inj/(self.alms_inj[0] * np.sqrt(4*np.pi))

#                     ## extrct only the non-negative components
#                     alms_non_neg = self.alms_inj[0:hp.Alm.getsize(self.almax)]

#                     Omega_1mHz = 10**(self.inj['ln_omega0']) * (1e-3/25)**(self.inj['alpha'])

#                     ## response matrix summed over Ylms
#                     summ_response_mat = np.einsum('ijklm,m', response_mat, self.alms_inj)

#                     # converts alm_inj into a healpix map to be plotted and saved
#                     # Plot with twice the analysis nside for better resolution
#                     skymap_inj = hp.alm2map(alms_non_neg, 2*self.params['nside'])

#                     Omegamap_inj = Omega_1mHz * skymap_inj

#                     hp.graticule()
#                     hp.mollview(Omegamap_inj, coord=coord, title='Injected angular distribution map $\Omega (f = 1 mHz)$')
                    
#                     plt.savefig(self.params['out_dir'] + '/inj_skymap.png', dpi=150)
#                     print('saving injected skymap at ' +  self.params['out_dir'] + '/inj_skymap.png')
#                     plt.close()
                    
#                     hp.graticule()
#                     hp.mollview(DWD_FG_map, coord=coord, title='Simulated DWD Foreground skymap')
#                     plt.savefig(self.params['out_dir'] + '/pre_inj_skymap.png', dpi=150)
#                     print('saving simulated skymap at ' +  self.params['out_dir'] + '/pre_inj_skymap.png')
#                     plt.close()
#                     hp.graticule()
#                     hp.mollview(hp.alm2map(DWD_FG_sph, 2*self.params['nside']), coord=coord, title='Simulated DWD Foreground alm map')
#                     plt.savefig(self.params['out_dir'] + '/pre_inj_almmap.png', dpi=150)
#                     print('saving simulated skymap at ' +  self.params['out_dir'] + '/pre_inj_almmap.png')
#                     plt.close()

#                 ## move frequency to be the zeroth-axis, then cholesky decomp
#                 L_cholesky = norms[:, None, None] *  np.linalg.cholesky(np.moveaxis(summ_response_mat[:, :, :, ii], -1, 0))
            

            
            ## adding elif statement for dwd_sdg. Copied dwd_fg from above -SMR
            elif self.inj['injtype'] == 'dwd_sdg':

                if ii == 0:

                    ## need to set up a few things before doing the spherical harmonic inj
                    
                    ## generate skymap
                    DWD_FG_map, log_DWD_FG_map = self.generate_sdg(self.inj['sdg_RA'], self.inj['sdg_DEC'], self.inj['sdg_DIST'], self.inj['sdg_RAD'], self.inj['sdg_NUM'])
                    ## convert to blms
                    DWD_FG_sph = self.sph_galactic_foreground(DWD_FG_map)
                    ## extract alms
                    self.alms_inj = self.blm_2_alm(DWD_FG_sph)

                    ## normalize
                    self.alms_inj = self.alms_inj/(self.alms_inj[0] * np.sqrt(4*np.pi))

                    ## extrct only the non-negative components
                    alms_non_neg = self.alms_inj[0:hp.Alm.getsize(self.almax)]

                    Omega_1mHz = 10**(self.inj['ln_omega0']) * (1e-3/25)**(self.inj['alpha'])

                    ## response matrix summed over Ylms
                    summ_response_mat = np.einsum('ijklm,m', response_mat, self.alms_inj)

                    # converts alm_inj into a healpix map to be plotted and saved
                    # Plot with twice the analysis nside for better resolution
                    skymap_inj = hp.alm2map(alms_non_neg, 2*self.params['nside'])

                    Omegamap_inj = Omega_1mHz * skymap_inj

                    hp.mollview(Omegamap_inj, coord=coord, title='Injected angular distribution map $\Omega (f = 1 mHz)$')
                    hp.graticule()
                    
                    plt.savefig(self.params['out_dir'] + '/inj_skymap.png', dpi=150)
                    print('saving injected skymap at ' +  self.params['out_dir'] + '/inj_skymap.png')
                    plt.close()
                    
                    hp.mollview(DWD_FG_map, coord=coord, title='Simulated DWD Foreground skymap')
                    hp.graticule()
                    plt.savefig(self.params['out_dir'] + '/pre_inj_skymap.png', dpi=150)
                    print('saving simulated skymap at ' +  self.params['out_dir'] + '/pre_inj_skymap.png')
                    plt.close()
                    hp.mollview(hp.alm2map(DWD_FG_sph, 2*self.params['nside']), coord=coord, title='Simulated DWD Foreground alm map')
                    hp.graticule()
                    plt.savefig(self.params['out_dir'] + '/pre_inj_almmap.png', dpi=150)
                    print('saving simulated skymap at ' +  self.params['out_dir'] + '/pre_inj_almmap.png')
                    plt.close()

                ## move frequency to be the zeroth-axis, then cholesky decomp
                L_cholesky = norms[:, None, None] *  np.linalg.cholesky(np.moveaxis(summ_response_mat[:, :, :, ii], -1, 0))
                

            ## generate standard normal complex data frist
            z_norm = np.random.normal(size=(frange.size, 3)) + 1j * np.random.normal(size=(frange.size, 3))

            ## The data in z_norm is rescaled into z_scale using L_cholesky
            z_scale = np.einsum('ijk, ikl -> ijl', L_cholesky, z_norm[:, :, None])[:, :, 0]

            ## The three channels : concatenate with norm at f = 0 to be zero
            htilda1  = np.concatenate([ [0], z_scale[:, 0]])
            htilda2  = np.concatenate([ [0], z_scale[:, 1]])
            htilda3  = np.concatenate([ [0], z_scale[:, 2]])


            if ii == 0:
                # Take inverse fft to get time series data
                h1 = splice_win * np.fft.irfft(htilda1, N)
                h2 = splice_win * np.fft.irfft(htilda2, N)
                h3 = splice_win * np.fft.irfft(htilda3, N)

            else:

                ## First append half-splice worth of zeros
                h1 = np.append(h1, np.zeros(halfN))
                h2 = np.append(h2, np.zeros(halfN))
                h3 = np.append(h3, np.zeros(halfN))

                ## Then add the new splice segment
                h1[-N:] = h1[-N:] + splice_win * np.fft.irfft(htilda1, N)
                h2[-N:] = h2[-N:] + splice_win * np.fft.irfft(htilda2, N)
                h3[-N:] = h3[-N:] + splice_win * np.fft.irfft(htilda3, N)


        ## remove the first half and the last half splice.
        h1, h2, h3 = h1[halfN:-halfN], h2[halfN:-halfN], h3[halfN:-halfN]

        tarr = self.params['tstart'] + tbreak +  np.arange(0, self.params['dur'], 1.0/self.params['fs'])

        return h1, h2, h3, tarr



    def add_sgwb_data_tshift(self, fs=0.25, dur=1e5):

        '''
        Wrapper function for generating stochastic data. The output are time domain data
        in whatever TDI levels are chosen,  at the three vertices oft the constellation.

        Returns
        ---------

        h1_gw, h2_gw, h3_gw : float
            Time series stochastic data

        '''

        # --------------------- Generate Fake Data + Noise -----------------------------
        print(" Adding sgwb signal ...")



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


        fidx = np.logical_and(freqs >= self.params['fmin'], freqs <= self.params['fmax'])

        H0 = 2.2*10**(-18) ## in SI units
        Omegaf = (10**self.inj['ln_omega0'])*(freqs/(self.params['fref']))**self.inj['alpha']


        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*freqs**3))*(H0/np.pi)**2
        norms = np.sqrt(self.params['fs']*Sgw*N)/2
        norms[0] = 0
        h1, h2, h3 = np.array([]), np.array([]), np.array([])

        sin_N, cos_N = np.sin(np.pi*np.arange(0, Nmid)/N), np.sin(np.pi*np.arange(Nmid, N)/N)

        for ii in range(tmids.size):

            R1, R2, R3 = self.add_astro_signal(f0)

            htilda1 = norms*(R1[:,0] + R1[:,1])
            htilda2 = norms*(R2[:,0] + R2[:,1])
            htilda3 = norms*(R3[:,0] + R3[:,1])


            # Take inverse fft to get time series data
            ht1 = np.real(np.fft.irfft(htilda1, N))
            ht2 = np.real(np.fft.irfft(htilda2, N))
            ht3 = np.real(np.fft.irfft(htilda3, N))

            if ii == 0:
                h1, h2, h3 = np.append(h1, ht1), np.append(h2, ht2), np.append(h3, ht1)
            else:

                h1[-Nmid:] = h1[-Nmid:]*cos_N + ht1[0:Nmid]*sin_N
                h2[-Nmid:] = h2[-Nmid:]*cos_N + ht2[0:Nmid]*sin_N
                h3[-Nmid:] = h3[-Nmid:]*cos_N + ht3[0:Nmid]*sin_N

                h1, h2, h3 = np.append(h1, ht1[Nmid:]), np.append(h2, ht2[Nmid:]), np.append(h3, ht1[Nmid:])

        times = (1.0/self.params['fs'])*np.arange(0, h1.size)

        return h1, h2, h3, times

    def generate_galactic_foreground(self, rh=2.9, zh=0.3):
        '''
        Generate a galactic white dwarf binary foreground modeled after Breivik et al. (2020), consisting of a bulge + disk.
        rh is the radial scale height in kpc, zh is the vertical scale height in kpc. 
        Thin disk has rh=2.9kpc, zh=0.3kpc; Thick disk has rh=3.31kpc, zh=0.9kpc. Defaults to thin disk. 
        The distribution is azimuthally symmetric in the galactocentric frame.
        Returns
        ---------
        DWD_FG_map : float
            Healpy GW power skymap of the DWD galactic foreground.
        log_DWD_FG_map : float
            Healpy log GW power skymap. For plotting purposes.
        
        '''
        ## set grid density
        grid_fill = 200
        ## create grid *in cartesian coordinates*
        ## size of density grid gives enough padding around the galactic plane without becoming needlessly large
        ## distances in kpc
        gal_rad = 20
        xs = np.linspace(-gal_rad,gal_rad,grid_fill)
        ys = np.linspace(-gal_rad,gal_rad,grid_fill)
        zs = np.linspace(-5,5,grid_fill)
        x, y, z = np.meshgrid(xs,ys,zs)
        r = np.sqrt(x**2 + y**2)
        ## Calculate density distribution
        rho_c = 1 # some fiducial central density (?? not sure what to use for this)
        r_cut = 2.1 #kpc
        r0 = 0.075 #kpc
        alpha = 1.8
        q = 0.5
        disk_density = rho_c*np.exp(-r/rh)*np.exp(-np.abs(z)/zh) 
        bulge_density = rho_c*(np.exp(-(r/r_cut)**2)/(1+np.sqrt(r**2 + (z/q)**2)/r0)**alpha)
        DWD_density = disk_density + bulge_density
        ## Use astropy.coordinates to transform from galactocentric frame to galactic (solar system barycenter) frame.
#        gc = cc.Galactocentric(x=x*u.kpc,y=y*u.kpc,z=z*u.kpc)
        gc = cc.SkyCoord(x=x*u.kpc,y=y*u.kpc,z=z*u.kpc, frame='galactocentric')
        SSBc = gc.transform_to(cc.Galactic)
        ## Calculate GW power
        #DWD_strains = DWD_density*(np.array(SSBc.distance))**-1
        DWD_powers = DWD_density*(np.array(SSBc.distance))**-2
        ## Filter nearby grid points (cut out 2kpc sphere)
        ## This is a temporary soln. Later, we will want to do something more subtle, sampling a DWD pop from
        ## the density distribution and filtering out resolveable SNR>80 binaries
        DWD_unresolved_powers = DWD_powers*(np.array(SSBc.distance) > 2)
        ## Transform to healpix basis
        ## resolution is 2x analysis resolution
        pixels = hp.ang2pix(2*self.params['nside'],np.array(SSBc.l),np.array(SSBc.b),lonlat=True)
        ## Create skymap
#        DWD_FG_mapG = np.zeros(hp.nside2npix(2*self.params['nside']))
        ## Bin
        DWD_FG_mapG = np.bincount(pixels.flatten(),weights=DWD_unresolved_powers.flatten(),minlength=hp.nside2npix(2*self.params['nside']))
#        for i in range(DWD_FG_mapG.size):
#            DWD_FG_mapG[i] = np.sum((pixels==i)*DWD_unresolved_powers)
        ## create logarithmic skymap for plotting purposes
        log_DWD_FG_mapG = np.log10(DWD_FG_mapG + 10**-15 * (DWD_FG_mapG==0))
        ## Transform into the ecliptic
        rGE = hp.rotator.Rotator(coord=['G','E'])
        DWD_FG_map = rGE.rotate_map_pixel(DWD_FG_mapG)
        log_DWD_FG_map = rGE.rotate_map_pixel(log_DWD_FG_mapG)
        
        return DWD_FG_map, log_DWD_FG_map
        
    
    def sph_galactic_foreground(self, DWD_FG_map):
        '''
        Transform the foreground produced in generate_galactic_foreground() into
        b_lm spherical harmonic basis
        
        Returns
        ---------
        DWD_FG_sph : float
            Spherical harmonic healpy expansion of the galactic foreground
        '''
        ## Take square root of powers
        sqrt_map = np.sqrt(DWD_FG_map)
        ## Generate blms of power (alms of sqrt(power))
        DWD_FG_sph = hp.sphtfunc.map2alm(sqrt_map, lmax=self.blmax)

        # Normalize        
        DWD_FG_sph = DWD_FG_sph/DWD_FG_sph[0]

        return DWD_FG_sph
        

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

        #data = np.concatenate((timearray[:, None], h1[:, None], h2[:, None], h3[:, None]),axis=1 )
        #np.savetxt('owndata_2e7_xyz.txt', data)

        # Number of segmants
        nsegs = int(np.floor(self.params['dur']/self.params['seglen'])) -1

        Nperseg=int(self.params['fs']*self.params['seglen'])

        '''
        # Apply a cascading low pass filter
        b, a = sg.butter(2, 0.4*self.params['fs']/(self.params['fs']/2),\
                btype='lowpass', output='ba')
        #sos = sg.zpk2sos(zz, pp, kk)

        for ii in range(8):
            print('low pass filtering ...')
            h1 = sg.filtfilt(b, a, h1)
            h2 = sg.filtfilt(b, a, h2)
            h3 = sg.filtfilt(b, a, h3)
        '''

        fftfreqs = np.fft.rfftfreq(Nperseg, 1.0/self.params['fs'])


        # Map of spectrum
        r1 = np.zeros((fftfreqs.size, nsegs), dtype='complex')
        r2 = np.zeros((fftfreqs.size, nsegs), dtype='complex')
        r3 = np.zeros((fftfreqs.size, nsegs), dtype='complex')


        # Hann Window
        hwin = np.hanning(Nperseg)
        win_fact = np.mean(hwin**2)


        zpad = np.zeros(Nperseg)

        ## Initiate time segment arrays
        tsegstart = np.zeros(nsegs)
        tsegmid = np.zeros(nsegs)

        # We will use 50% overlapping segments
        for ii in range(0, nsegs):

            idxmin = int(ii*Nperseg)
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


## adding new code below -SMR
## copied generate_galactic_foreground code from above
    def generate_sdg(self, RA=80.21496, DEC=-69.37772, DIST=50, RAD=2.1462, NUM=2169264):
        '''
        Should redo this text:
        Generate a galactic white dwarf binary foreground modeled after Breivik et al. (2020), consisting of a bulge + disk.
        rh is the radial scale height in kpc, zh is the vertical scale height in kpc. 
        Thin disk has rh=2.9kpc, zh=0.3kpc; Thick disk has rh=3.31kpc, zh=0.9kpc. Defaults to thin disk. 
        The distribution is azimuthally symmetric in the galactocentric frame.
        Returns
        ---------
        DWD_FG_map : float
            Healpy GW power skymap of the DWD galactic foreground.
        log_DWD_FG_map : float
            Healpy log GW power skymap. For plotting purposes.
        
        '''
        ## ===== ipynb compute_density function ========================================
        ## all below is only for galaxy model creation
            ## set grid density
        grid_fill = 200

        # sdg radius: (default is the LMC)
        sdg_r = RAD*u.kpc
        
        # default coordinates give the position of the center of the LMC in ICRS coordinates:
        sdg_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, distance=DIST*u.kpc)

        # transform to galactocentric coordinates:
        sdg_galcen = sdg_icrs.transform_to(cc.Galactocentric)
        
        # convert to cartesian coordinates with the origin at the galactic center
        x_sdg = sdg_galcen.cartesian.x
        y_sdg = sdg_galcen.cartesian.y
        z_sdg = sdg_galcen.cartesian.z
        
        ## create grid *in cartesian coordinates*
        ## distances in kpc
        xs = np.linspace(x_sdg-sdg_r,x_sdg+sdg_r,grid_fill)
        ys = np.linspace(y_sdg-sdg_r,y_sdg+sdg_r,grid_fill)
        zs = np.linspace(z_sdg-sdg_r,z_sdg+sdg_r,grid_fill)
        x, y, z = np.meshgrid(xs,ys,zs)
        
        DWD_density = NUM / (0.524*200**3)
        # 0.524 is the filling factor of a sphere in a cube
        # this gives us the number density for points only within the sphere of the sdg, instead of the entire cube
    
        ## creating a sphere_filter 3D array, with 1s in a sphere and 0s otherwise
        # r = distance from any point to the center of the sdg
        r = np.sqrt((x-x_sdg)**2+(y-y_sdg)**2+(z-z_sdg)**2)
        
        # set any points within the sdg radius to 1, any points outside to 0
        sphere_filter = np.zeros((grid_fill,grid_fill,grid_fill))
        for i in range(grid_fill):
            for j in range(grid_fill):
                for k in range(grid_fill):
                    sphere_filter[i,j,k] = 1 if (r[i,j,k]<sdg_r) else 0
        ## ** this is probably a computationally expensive way to do this, but it works

        ## =============================================================================
        
        ## ===== ipynb next block ======================================================
        ## Use astropy.coordinates to transform from galactocentric frame to galactic (solar system barycenter) frame.
        gc = cc.Galactocentric(x=x,y=y,z=z)
        SSBc = gc.transform_to(cc.Galactic)
        ## =============================================================================
       
        ## Calculate GW power
        #DWD_strains = DWD_density*(np.array(SSBc.distance))**-1
        ## density will be total power divided by the points that we're simulating
        ## assuming all grid points will contribute an equal amount of power
        DWD_powers = sphere_filter*DWD_density*(np.array(SSBc.distance))**-2
        ## Filter nearby grid points (cut out 2kpc sphere)
        ## This is a temporary soln. Later, we will want to do something more subtle, sampling a DWD pop from
        ## the density distribution and filtering out resolveable SNR>80 binaries
        DWD_unresolved_powers = sphere_filter*DWD_powers*(np.array(SSBc.distance) > 2)
        ## will need to generate DWD_unresolved_powers for sdg
        

        ## Transform to healpix basis
        ## resolution is 2x analysis resolution
        ## setting resolution, taking coordinates from before and transforming to longlat
        ## replace np.array ... with sdg coordinates
        pixels = hp.ang2pix(2*self.params['nside'],np.array(SSBc.l),np.array(SSBc.b),lonlat=True)
        

        ## Create skymap
        DWD_FG_mapG = np.zeros(hp.nside2npix(2*self.params['nside']))
        ## Bin
        DWD_FG_mapG = np.bincount(pixels.flatten(),weights=DWD_unresolved_powers.flatten(),minlength=hp.nside2npix(2*self.params['nside']))
        ## old, slow way:
        # for i in range(DWD_FG_mapG.size):
        #     DWD_FG_mapG[i] = np.sum((pixels==i)*DWD_unresolved_powers)
        
        ## create logarithmic skymap for plotting purposes
        log_DWD_FG_mapG = np.log10(DWD_FG_mapG + 10**-15 * (DWD_FG_mapG==0))
        

        ## below isn't in the jupyter notebook?
        ## Transform into the ecliptic
        rGE = hp.rotator.Rotator(coord=['G','E'])
        DWD_FG_map = rGE.rotate_map_pixel(DWD_FG_mapG)
        log_DWD_FG_map = rGE.rotate_map_pixel(log_DWD_FG_mapG)
        
        ## returning healpix skymaps
        return DWD_FG_map, log_DWD_FG_map


    def old_generate_lmc(self, rh=0.1, zh=0.1):
        '''
        Generate a galactic white dwarf binary foreground modeled after Breivik et al. (2020), consisting of a bulge + disk.
        rh is the radial scale height in kpc, zh is the vertical scale height in kpc. 
        Thin disk has rh=2.9kpc, zh=0.3kpc; Thick disk has rh=3.31kpc, zh=0.9kpc. Defaults to thin disk. 
        The distribution is azimuthally symmetric in the galactocentric frame.
        Returns
        ---------
        DWD_FG_map : float
            Healpy GW power skymap of the DWD galactic foreground.
        log_DWD_FG_map : float
            Healpy log GW power skymap. For plotting purposes.
        
        '''
        ## ===== ipynb compute_density function ========================================
        ## all below is only for galaxy model creation
        ## set grid density
        grid_fill = 200
        ## create grid *in cartesian coordinates*
        ## size of density grid gives enough padding around the galactic plane without becoming needlessly large
        ## distances in kpc
        gal_rad = 20
        xs = np.linspace(-gal_rad,gal_rad,grid_fill)
        ys = np.linspace(-gal_rad,gal_rad,grid_fill)
        zs = np.linspace(-5,5,grid_fill)
        x, y, z = np.meshgrid(xs,ys,zs)
        r = np.sqrt(x**2 + y**2)
        
        ## all of below is unnecessary
        ## Calculate density distribution
        rho_c = 1 # some fiducial central density (?? not sure what to use for this)
        r_cut = 2.1 #kpc
        r0 = 0.075 #kpc
        alpha = 1.8
        q = 0.5
        disk_density = rho_c*np.exp(-r/rh)*np.exp(-np.abs(z)/zh) 
        bulge_density = rho_c*(np.exp(-(r/r_cut)**2)/(1+np.sqrt(r**2 + (z/q)**2)/r0)**alpha)
        DWD_density = disk_density + bulge_density
        ## =============================================================================
        
        ## ===== ipynb next block ======================================================
        ## Use astropy.coordinates to transform from galactocentric frame to galactic (solar system barycenter) frame.
        gc = cc.Galactocentric(x=x*u.kpc,y=y*u.kpc,z=z*u.kpc)
        SSBc = gc.transform_to(cc.Galactic)
        ## =============================================================================
       
        ## Calculate GW power
        #DWD_strains = DWD_density*(np.array(SSBc.distance))**-1
        ## density will be total power divided by the points that we're simulating
        ## assuming all grid points will contribute an equal amount of power
        DWD_powers = DWD_density*(np.array(SSBc.distance))**-2
        ## Filter nearby grid points (cut out 2kpc sphere)
        ## This is a temporary soln. Later, we will want to do something more subtle, sampling a DWD pop from
        ## the density distribution and filtering out resolveable SNR>80 binaries
        DWD_unresolved_powers = DWD_powers*(np.array(SSBc.distance) > 2)
        ## will need to generate DWD_unresolved_powers for lmc
        

        ## Transform to healpix basis
        ## resolution is 2x analysis resolution
        ## setting resolution, taking coordinates from before and transforming to longlat
        ## replace np.array ... with lmc coordinates
        pixels = hp.ang2pix(2*self.params['nside'],np.array(SSBc.l),np.array(SSBc.b),lonlat=True)
        

        ## Create skymap
        DWD_FG_mapG = np.zeros(hp.nside2npix(2*self.params['nside']))
        ## Bin
        for i in range(DWD_FG_mapG.size):
            DWD_FG_mapG[i] = np.sum((pixels==i)*DWD_unresolved_powers)
        ## create logarithmic skymap for plotting purposes
        log_DWD_FG_mapG = np.log10(DWD_FG_mapG + 10**-15 * (DWD_FG_mapG==0))
        

        ## below isn't in the jupyter notebook?
        ## Transform into the ecliptic
        rGE = hp.rotator.Rotator(coord=['G','E'])
        DWD_FG_map = rGE.rotate_map_pixel(DWD_FG_mapG)
        log_DWD_FG_map = rGE.rotate_map_pixel(log_DWD_FG_mapG)
        
        ## returning healpix skymaps
        return DWD_FG_map, log_DWD_FG_map
