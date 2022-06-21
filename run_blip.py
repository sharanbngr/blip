import pickle
import numpy as np
import sys, configparser, subprocess
from src.makeLISAdata import LISAdata
from src.likelihoods import likelihoods
from tools.plotmaker import plotmaker
import matplotlib.pyplot as plt
from astropy import units as u
from multiprocessing import Pool
import time
import healpy as hp
# from eogtest import open_img
from src.dynesty_engine import dynesty_engine
#from src.emcee_engine import emcee_engine

class LISA(LISAdata, likelihoods):

    '''
    Generic class for getting data and setting up the prior space
    and likelihood. This is tuned for ISGWB analysis at the moment
    but it should not be difficult to modify for other use cases.
    '''

    def __init__(self,  params, inj):

        # set up the LISAdata class
        LISAdata.__init__(self, params, inj)

        # Make noise spectra
        self.which_noise_spectrum()

        if self.inj['doInj']:
            self.which_astro_signal()

        # Generate or get mldc data
        if self.params['mldc']:
            self.read_mldc_data()
        else:
            self.makedata()

        # Set up the Bayes class
        likelihoods.__init__(self)

        # Figure out which response function to use for recoveries
        self.which_response()

        # Make some simple diagnostic plots to contrast spectra
        self.diag_spectra()

        if self.params['modeltype'] != 'noise_only':
            self.snr_calc()

    def makedata(self):

        '''
        Just a wrapper function to use the methods the LISAdata class
        to generate data. Return Frequency domain data.
        '''

        # Generate TDI noise
        times, self.h1, self.h2, self.h3 = self.gen_noise_spectrum()
        delt = times[1] - times[0]

        # Cut to required size
        N = int((self.params['dur'])/delt)
        self.h1, self.h2, self.h3 = self.h1[0:N], self.h2[0:N], self.h3[0:N]

        # Generate TDI isotropic signal
        if self.inj['doInj']:

#            if self.inj['injtype']=='point_source':      
#                h1_gw, h2_gw, h3_gw, times = self.add_ps_data()
#            else:
            h1_gw, h2_gw, h3_gw, times = self.add_sgwb_data()

            h1_gw, h2_gw, h3_gw = h1_gw[0:N], h2_gw[0:N], h3_gw[0:N]

            # Add gravitational-wave time series to noise time-series
            self.h1 = self.h1 + h1_gw
            self.h2 = self.h2 + h2_gw
            self.h3 = self.h3 + h3_gw

        self.timearray = times[0:N]
        if delt != (times[1] - times[0]):
            raise ValueError('The noise and signal arrays are at different sampling frequencies!')

        # Desample if we increased the sample rate for time-shifts.
        if self.params['fs'] != 1.0/delt:
            self.params['fs'] = 1.0/delt

        # Generate lisa freq domain data from time domain data
        self.r1, self.r2, self.r3, self.fdata, self.tsegstart, self.tsegmid = self.tser2fser(self.h1, self.h2, self.h3, self.timearray)

        # Charactersitic frequency. Define f0
        cspeed = 3e8
        fstar = cspeed/(2*np.pi*self.armlength)
        self.f0 = self.fdata/(2*fstar)
        
    def read_mldc_data(self):
        '''
        Just a wrapper function to use the methods the LISAdata class to
        read data. Return frequency domain data. Since this was used
        primarily for the MLDC, this assumes that the data is doppler
        tracking and converts to strain data.
        '''

        h1, h2, h3, self.timearray = self.read_data()

        # Calculate other tdi combinations if necessary.
        if self.params['tdi_lev'] == 'aet':
            h1 = (1.0/3.0)*(2*h1 - h2 - h3)
            h2 = (1.0/np.sqrt(3.0))*(h3 - h2)
            h3 = (1.0/3.0)*(h1 + h2 + h3)

        # Generate lisa freq domain data from time domain data
        self.r1, self.r2, self.r3, self.fdata, self.tsegstart, self.tsegmid = self.tser2fser(h1, h2, h3, self.timearray)

        # Charactersitic frequency. Define f0
        cspeed = 3e8
        fstar = cspeed/(2*np.pi*self.armlength)
        self.f0 = self.fdata/(2*fstar)

        # Convert doppler data to strain if readfile datatype is doppler.
        if self.params['datatype'] == 'doppler':

            # This is needed to convert from doppler data to strain data.
            self.r1, self.r2, self.r3 = self.r1/(4*self.f0.reshape(self.f0.size, 1)), self.r2/(4*self.f0.reshape(self.f0.size, 1)), self.r3/(4*self.f0.reshape(self.f0.size, 1))

        elif self.params['datatype'] == 'strain':
            pass


    def which_noise_spectrum(self):

        # Figure out which instrumental noise spectra to use
        if self.params['tdi_lev']=='aet':
            self.instr_noise_spectrum = self.aet_noise_spectrum
            self.gen_noise_spectrum = self.gen_aet_noise
        elif self.params['tdi_lev']=='xyz':
            self.instr_noise_spectrum = self.xyz_noise_spectrum
            self.gen_noise_spectrum = self.gen_xyz_noise
        elif self.params['tdi_lev']=='michelson':
            self.instr_noise_spectrum = self.mich_noise_spectrum
            self.gen_noise_spectrum = self.gen_michelson_noise

    def which_response(self):

        # Calculate reponse function to use for analysis
        if (self.params['modeltype'] == 'isgwb' or self.params['modeltype'] == 'isgwb_only') and self.params['tdi_lev']=='aet':
            self.response_mat = self.isgwb_aet_response(self.f0, self.tsegmid)

        elif (self.params['modeltype'] == 'isgwb' or self.params['modeltype'] == 'isgwb_only') and self.params['tdi_lev']=='xyz':
            self.response_mat = self.isgwb_xyz_response(self.f0, self.tsegmid)

        elif (self.params['modeltype'] == 'isgwb' or self.params['modeltype'] == 'isgwb_only') and self.params['tdi_lev']=='michelson':
            self.response_mat = self.isgwb_mich_response(self.f0, self.tsegmid)

        elif self.params['modeltype']=='sph_sgwb' and self.params['tdi_lev']=='michelson':
            self.response_mat = self.asgwb_mich_response(self.f0, self.tsegmid)
        elif self.params['modeltype']=='sph_sgwb' and self.params['tdi_lev']=='xyz':
            self.response_mat = self.asgwb_xyz_response(self.f0, self.tsegmid)
        elif self.params['modeltype']=='sph_sgwb' and self.params['tdi_lev']=='aet':
            self.response_mat = self.asgwb_aet_response(self.f0, self.tsegmid)
            
        elif self.params['modeltype']=='dwd_fg' and self.params['tdi_lev']=='michelson':
            self.response_mat = self.asgwb_mich_response(self.f0, self.tsegmid)
        elif self.params['modeltype']=='dwd_fg' and self.params['tdi_lev']=='xyz':
            self.response_mat = self.asgwb_xyz_response(self.f0, self.tsegmid)
        elif self.params['modeltype']=='dwd_fg' and self.params['tdi_lev']=='aet':
            self.response_mat = self.asgwb_aet_response(self.f0, self.tsegmid)
        ## import pdb; pdb.set_trace()
        ## adding copy of above for dwd_sdg -SMR
        elif self.params['modeltype']=='dwd_sdg' and self.params['tdi_lev']=='michelson':
            self.response_mat = self.asgwb_mich_response(self.f0, self.tsegmid)
        elif self.params['modeltype']=='dwd_sdg' and self.params['tdi_lev']=='xyz':
            self.response_mat = self.asgwb_xyz_response(self.f0, self.tsegmid)
        elif self.params['modeltype']=='dwd_sdg' and self.params['tdi_lev']=='aet':
            self.response_mat = self.asgwb_aet_response(self.f0, self.tsegmid)

        elif self.params['modeltype'] == 'noise_only':
            print('Noise only model chosen ...')
        else:
            raise ValueError('Unknown recovery model selected')

    def which_astro_signal(self):

        # Figure out which antenna patterns to use
        if self.inj['injtype'] == 'isgwb' and self.params['tdi_lev']=='aet':
            self.add_astro_signal = self.isgwb_aet_response
        elif self.inj['injtype'] == 'isgwb' and self.params['tdi_lev']=='xyz':
            self.add_astro_signal = self.isgwb_xyz_response
        elif self.inj['injtype'] == 'isgwb' and self.params['tdi_lev']=='michelson':
            self.add_astro_signal = self.isgwb_mich_response

        elif self.inj['injtype']=='sph_sgwb' and self.params['tdi_lev']=='michelson':
            self.add_astro_signal = self.asgwb_mich_response
        elif self.inj['injtype']=='sph_sgwb' and self.params['tdi_lev']=='aet':
            self.add_astro_signal = self.asgwb_aet_response
        elif self.inj['injtype']=='sph_sgwb' and self.params['tdi_lev']=='xyz':
            self.add_astro_signal = self.asgwb_xyz_response
        elif self.inj['injtype']=='point_source' and self.params['tdi_lev']=='michelson':
#            self.add_astro_signal = self.ps_mich_response
            self.add_astro_signal = self.asgwb_mich_response
        elif self.inj['injtype']=='point_source' and self.params['tdi_lev']=='aet':
#            self.add_astro_signal = self.ps_aet_response
            self.add_astro_signal = self.asgwb_aet_response
        elif self.inj['injtype']=='point_source' and self.params['tdi_lev']=='xyz':
#            self.add_astro_signal = self.ps_xyz_response
            self.add_astro_signal = self.asgwb_xyz_response
        elif self.inj['injtype']=='dwd_fg' and self.params['tdi_lev']=='michelson':
            self.add_astro_signal = self.asgwb_mich_response
        elif self.inj['injtype']=='dwd_fg' and self.params['tdi_lev']=='aet':
            self.add_astro_signal = self.asgwb_aet_response
        elif self.inj['injtype']=='dwd_fg' and self.params['tdi_lev']=='xyz':
            self.add_astro_signal = self.asgwb_xyz_response
        ## copy of above for dwd_sdg -SMR
        elif self.inj['injtype']=='dwd_sdg' and self.params['tdi_lev']=='michelson':
            self.add_astro_signal = self.asgwb_mich_response
        elif self.inj['injtype']=='dwd_sdg' and self.params['tdi_lev']=='aet':
            self.add_astro_signal = self.asgwb_aet_response
        elif self.inj['injtype']=='dwd_sdg' and self.params['tdi_lev']=='xyz':
            self.add_astro_signal = self.asgwb_xyz_response
        else:
           raise ValueError('Unknown recovery model selected')


    def diag_spectra(self):

        '''
        A function to do simple diagnostics. Plot the expected spectra and data.
        '''

        import scipy.signal as sg

        # ------------ Calculate PSD ------------------

        # Number of segmants
        Nperseg=int(self.params['fs']*self.params['seglen'])

        # PSD from the FFTs
        data_PSD1, data_PSD2, data_PSD3  = np.mean(np.abs(self.r1)**2, axis=1), np.mean(np.abs(self.r2)**2, axis=1), np.mean(np.abs(self.r3)**2, axis=1)

        # "Cut" to desired frequencies
        idx = np.logical_and(self.fdata >=  self.params['fmin'] , self.fdata <=  self.params['fmax'])
        psdfreqs = self.fdata[idx]

        #Charactersitic frequency
        fstar = 3e8/(2*np.pi*self.armlength)

        # define f0 = f/2f*
        f0 = self.fdata/(2*fstar)

        # Get desired frequencies for the PSD
        # We want to normalize PSDs to account for the windowing
        # Also convert from doppler-shift spectra to strain spectra
        data_PSD1,data_PSD2, data_PSD3 = data_PSD1[idx], data_PSD2[idx], data_PSD3[idx]

        # The last two elements are the position and the acceleration noise levels.
        Np, Na = 10**self.inj['log_Np'], 10**self.inj['log_Na']

        # Modelled Noise PSD
        C_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)

        # Extract noise auto-power
        S1, S2, S3 = C_noise[0, 0, :], C_noise[1, 1, :], C_noise[2, 2, :]

        if self.params['modeltype'] != 'noise_only':
            ## first check if the injection is a point source
#            if self.inj['injtype'] == 'point_source':
#                # extra auto-power GW responses
#                summ_response_mat = np.sum(self.response_mat, axis=-1)
#                R1 = np.real(summ_response_mat[0, 0, :, :])
#                R2 = np.real(summ_response_mat[1, 1, :, :])
#                R3 = np.real(summ_response_mat[2, 2, :, :])
            if self.params['modeltype'] == 'sph_sgwb' or self.params['modeltype'] == 'dwd_fg' or self.params['modeltype'] == 'dwd_sdg':
#                if self.inj['injtype'] == 'point_source':
#                    npix = hp.nside2npix(10)
#                    Sgwmap = np.zeros(npix)
#                    
#                    # identify the pixel with the point source
#                    ps_id = hp.ang2pix(10, self.inj['theta'], self.inj['phi'])
#
#                    Omega_1mHz = 10**(self.inj['ln_omega0']) * (1e-3/25)**(self.inj['alpha'])
#                    
#                    H0 = 2.2*10**(-18)
#                    Sgwmap[ps_id] = np.sqrt(Omega_1mHz*(3/(4*(1e-3)**3))*(H0/np.pi)**2)
#                    blms_inj = hp.sphtfunc.map2alm(Sgwmap,lmax=self.blmax)
#                    blms_inj = blms_inj/(blms_inj[0]* np.sqrt(4*np.pi))
#                    alms_inj = self.blm_2_alm(blms_inj)
#                    self.alms_inj = alms_inj/(alms_inj[0] * np.sqrt(4*np.pi))
##                else:
#                    alms_inj = self.blm_2_alm(self.inj['blms'])

                # normalize
#                alms_inj = alms_inj/(alms_inj[0] * np.sqrt(4*np.pi))
#                else:
                summ_response_mat = np.sum(self.response_mat*self.alms_inj[None, None, None, None, :], axis=-1)
                # extra auto-power GW responses
                R1 = np.real(summ_response_mat[0, 0, :, :])
                R2 = np.real(summ_response_mat[1, 1, :, :])
                R3 = np.real(summ_response_mat[2, 2, :, :])

            else:
                # extra auto-power GW responses
                R1 = np.real(self.response_mat[0, 0, :, :])
                R2 = np.real(self.response_mat[1, 1, :, :])
                R3 = np.real(self.response_mat[2, 2, :, :])

            # SGWB signal levels of the mldc data
            Omega0, alpha = 10**self.inj['ln_omega0'], self.inj['alpha']

            # Hubble constant
            H0 = 2.2*10**(-18)
            # Calculate astrophysical power law noise

            if self.params['modeltype'] == 'dwd_fg' or self.params['modeltype'] == 'dwd_sdg':
                if self.inj['fg_spectrum'] == 'powerlaw':
                    Omegaf = Omega0*(self.fdata/self.params['fref'])**alpha

#             # added or statement for dwd_sdg -SMR
#             if (self.params['modeltype'] == 'dwd_fg' or self.params['modeltype'] == 'dwd_sdg') and self.inj['fg_spectrum'] == 'truncated':
#                 ## frequency cutoff based on Fig 1. of Breivik et al (2020)
#                 fcutoff = self.inj['fcutoff']
#                 fcut = (self.fdata < fcutoff)*self.fdata
#                 Omegaf = Omega0*(fcut/25)**alpha
#                 # Power spectra of the SGWB
#                 Sgw = (3.0*(H0**2)*Omegaf)/(4*np.pi*np.pi*self.fdata**3)
                
#                 # Spectrum of the SGWB signal convoluted with the detector response tensor.
#                 S1_gw, S2_gw, S3_gw = Sgw[:, None]*R1, Sgw[:, None]*R2, Sgw[:, None]*R3

    
                    # Power spectra of the SGWB
                    Sgw = (3.0*(H0**2)*Omegaf)/(4*np.pi*np.pi*self.fdata**3)
                    
                    # Spectrum of the SGWB signal convoluted with the detector response tensor.
                    S1_gw, S2_gw, S3_gw = Sgw[:, None]*R1, Sgw[:, None]*R2, Sgw[:, None]*R3
        
                    # The total noise spectra is the sum of the instrumental + astrophysical
                    S1, S2, S3 = S1[:, None] + S1_gw, S2[:, None] + S2_gw, S3[:, None] + S3_gw
        
                    plt.close()
                    plt.loglog(self.fdata, np.mean(S1_gw,axis=1), label='Simulated GW spectrum', lw=0.75)
                elif self.inj['fg_spectrum'] == 'broken_powerlaw':
#                    Omegaf = Omega0*(self.fdata/self.params['fref'])**alpha
#                    
#                    fcutoff = 10**self.inj['log_fcut']
#                    lowfilt = (self.fdata < fcutoff)
#                    highfilt = np.invert(lowfilt)
#                    Omega_cut = (10**self.inj['ln_omega0'])*(fcutoff/(self.params['fref']))**self.inj['alpha'] 
#                    Omegaf = lowfilt*(10**self.inj['ln_omega0'])*(self.fdata/(self.params['fref']))**self.inj['alpha'] + \
#                             highfilt*Omega_cut*(self.fdata/fcutoff)**self.inj['alpha2']
                    alpha_2 = self.inj['alpha1'] - 0.667
                    Omegaf = ((10**self.inj['log_A1'])*(self.fdata/self.params['fref'])**self.inj['alpha1'])/(\
                         1 + (10**self.inj['log_A2'])*(self.fdata/self.params['fref'])**alpha_2)
                    # Power spectra of the SGWB
                    Sgw = (3.0*(H0**2)*Omegaf)/(4*np.pi*np.pi*self.fdata**3)
                    
                    # Spectrum of the SGWB signal convolved with the detector response tensor.
                    S1_gw, S2_gw, S3_gw = Sgw[:, None]*R1, Sgw[:, None]*R2, Sgw[:, None]*R3
        
                    # The total noise spectra is the sum of the instrumental + astrophysical
                    S1, S2, S3 = S1[:, None] + S1_gw, S2[:, None] + S2_gw, S3[:, None] + S3_gw
        
                    plt.close()
                    plt.loglog(self.fdata, np.mean(S1_gw,axis=1), label='Simulated GW spectrum', lw=0.75)
                elif self.inj['fg_spectrum'] == 'truncated':
                    ## frequency cutoff based on Fig 1. of Breivik et al (2020)
                    fcutoff = 10**self.inj['log_fcut']
                    fcut = (self.fdata < fcutoff)*self.fdata
                    Omegaf = Omega0*(fcut/self.params['fref'])**alpha
                    # Power spectra of the SGWB
                    Sgw = (3.0*(H0**2)*Omegaf)/(4*np.pi*np.pi*self.fdata**3)
                    
                    # Spectrum of the SGWB signal convoluted with the detector response tensor.
                    S1_gw, S2_gw, S3_gw = Sgw[:, None]*R1, Sgw[:, None]*R2, Sgw[:, None]*R3
        
                    # The total noise spectra is the sum of the instrumental + astrophysical
                    S1, S2, S3 = S1[:, None] + S1_gw, S2[:, None] + S2_gw, S3[:, None] + S3_gw
                    
                    plt.close()
                    plt.loglog(np.append(self.fdata[self.fdata < fcutoff],fcutoff), np.append(np.mean(S1_gw,axis=1)[self.fdata < fcutoff],0), label='Simulated GW spectrum', lw=0.75)
                elif self.inj['fg_spectrum'] == 'population':
                    # Power spectra of the specified DWD population
                    Sgw = self.pop2spec(self.inj['popfile'],self.fdata,self.params['dur']*u.s,names=self.inj['columns'])*4 ##h^2 = 1/2S_A = 1/2 * 1/2S_GW
                    # Spectrum of the SGWB signal convoluted with the detector response tensor.
                    S1_gw, S2_gw, S3_gw = Sgw[:, None]*R1, Sgw[:, None]*R2, Sgw[:, None]*R3
        
                    # The total noise spectra is the sum of the instrumental + astrophysical
                    S1, S2, S3 = S1[:, None] + S1_gw, S2[:, None] + S2_gw, S3[:, None] + S3_gw
        
                    plt.close()
                    plt.loglog(self.fdata, np.mean(S1_gw,axis=1), label='Simulated GW spectrum', lw=0.75)
                    plt.loglog(self.fdata, np.mean(S1,axis=1), label='Simulated Total spectrum', lw=0.75)
            else:       
                Omegaf = Omega0*(self.fdata/self.params['fref'])**alpha

                # Power spectra of the SGWB
                Sgw = (3.0*(H0**2)*Omegaf)/(4*np.pi*np.pi*self.fdata**3)
                
                # Spectrum of the SGWB signal convoluted with the detector response tensor.
                S1_gw, S2_gw, S3_gw = Sgw[:, None]*R1, Sgw[:, None]*R2, Sgw[:, None]*R3
    
                # The total noise spectra is the sum of the instrumental + astrophysical
                S1, S2, S3 = S1[:, None] + S1_gw, S2[:, None] + S2_gw, S3[:, None] + S3_gw
    
                plt.close()
                plt.loglog(self.fdata, np.mean(S1_gw,axis=1), label='Simulated GW spectrum', lw=0.75)

        # noise budget plot
        plt.loglog(psdfreqs, data_PSD3,label='PSD, data series', alpha=0.6, lw=0.75)
        plt.loglog(self.fdata, C_noise[2, 2, :], label='Simulated instrumental noise spectrum', lw=0.75 )
#        plt.ylim([1e-43, 1e-39])
        plt.legend()
        plt.xlabel('$f$ in Hz')
        plt.ylabel('PSD 1/Hz ')
        plt.xlim(0.5*self.params['fmin'], 2*self.params['fmax'])
        plt.savefig(self.params['out_dir'] + '/psd_budget.png', dpi=200)
        print('Diagnostic spectra plot made in ' + self.params['out_dir'] + '/psd_budget.png')
        plt.close()


        plt.loglog(self.fdata, np.mean(S3,axis=1), label='required')
        plt.loglog(psdfreqs, data_PSD3,label='PSD, data', alpha=0.6)
        plt.xlabel('$f$ in Hz')
        plt.ylabel('PSD 1/Hz ')
        plt.legend()
        plt.grid(linestyle=':',linewidth=0.5 )
#        plt.ylim([1e-44, 5e-40])
        plt.xlim(0.5*self.params['fmin'], 2*self.params['fmax'])

        plt.savefig(self.params['out_dir'] + '/diag_psd.png', dpi=200)
        print('Diagnostic spectra plot made in ' + self.params['out_dir'] + '/diag_psd.png')
        plt.close()




        ## lets also plot psd residue.
        rel_res_mean = (data_PSD3 - np.mean(S3,axis=1))/np.mean(S3,axis=1)

        plt.semilogx(self.fdata, rel_res_mean , label='relative mean residue')
        plt.xlabel('f in Hz')
        plt.ylabel(' Rel. residue')
        plt.ylim([-1.50, 1.50])
        plt.legend()
        plt.grid()
        plt.xlim(0.5*self.params['fmin'], 2*self.params['fmax'])

        plt.savefig(self.params['out_dir'] + '/res_psd.png', dpi=200)
        print('Residue spectra plot made in ' + self.params['out_dir'] + '/res_psd.png')
        plt.close()
        
        ## lets also plot psd residue factor
#        rel_res_mean = (data_PSD3 - np.mean(S3,axis=1))/np.mean(S3,axis=1)
        res_fac_mean = np.mean(S3,axis=1)/data_PSD3
        plt.semilogx(self.fdata, res_fac_mean , label='mean residue factor')
        plt.xlabel('f in Hz')
        plt.ylabel('Residue Factor')
#        plt.ylim([-1.50, 1.50])
        plt.legend()
        plt.grid()
        plt.xlim(0.5*self.params['fmin'], 2*self.params['fmax'])

        plt.savefig(self.params['out_dir'] + '/res_fac_psd.png', dpi=200)
        print('Second residue spectra plot made in ' + self.params['out_dir'] + '/res)fac_psd.png')
        plt.close()
        
        # cross-power diag plots. We will only do 12. IF TDI=XYZ this is S_XY and if TDI=AET
        # this will be S_AE

        ii, jj = 2, 0

        if self.params['modeltype'] == 'noise_only':
            Sx = C_noise[ii, jj, :]
        elif self.params['modeltype'] == 'sph_sgwb':
            if self.inj['injtype'] == 'point_source':
                Sx = C_noise[ii, jj, :, None] + Sgw[:,None]*summ_response_mat[ii, jj, :, 0]
            else:
                Sx = C_noise[ii, jj, :] + Sgw*summ_response_mat[ii, jj, :, 0]
            
        else:
            Sx = C_noise[ii, jj, :, None] + Sgw[:,None]*self.response_mat[ii, jj, :, 0]
        
        CSDx = np.mean(np.conj(self.rbar[:, :, ii]) * self.rbar[:, :, jj], axis=1)

        plt.subplot(2, 1, 1)
        plt.loglog(self.fdata, np.mean(np.abs(np.real(Sx)),axis=1), label='Re(Required ' + str(ii+1) + str(jj+1) + ')' )
        plt.loglog(psdfreqs, np.abs(np.real(CSDx)) ,label='Re(CSD' + str(ii+1) + str(jj+1) + ')', alpha=0.6)
        plt.xlabel('f in Hz')
        plt.ylabel('Power in 1/Hz')
        plt.legend()
        plt.ylim([1e-44, 5e-40])
        plt.xlim(0.5*self.params['fmin'], 2*self.params['fmax'])
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.loglog(self.fdata, np.mean(np.abs(np.imag(Sx)),axis=1), label='Im(Required ' + str(ii+1) + str(jj+1) + ')' )
        plt.loglog(psdfreqs, np.abs(np.imag(CSDx)) ,label='Im(CSD' + str(ii+1) + str(jj+1) + ')', alpha=0.6)
        plt.xlabel('f in Hz')
        plt.ylabel(' Power in 1/Hz')
        plt.legend()
        plt.xlim(0.5*self.params['fmin'], 2*self.params['fmax'])
        plt.ylim([1e-44, 5e-40])
        plt.grid()
        plt.savefig(self.params['out_dir'] + '/diag_csd_' + str(ii+1) + str(jj+1) + '.png', dpi=200)
        print('Diagnostic spectra plot made in ' + self.params['out_dir'] + '/diag_csd_' + str(ii+1) + str(jj+1) + '.png')
        plt.close()
        

    def snr_calc(self):

        '''
        Calculate the *theoretical* SNR of a signal in a single channel
        '''

        # Number of segmants

        Nperseg=int(self.params['fs']*self.params['seglen'])

        # "Cut" to desired frequencies
        idx = np.logical_and(self.fdata >=  self.params['fmin'] , self.fdata <=  self.params['fmax'])
        psdfreqs = self.fdata[idx]

        #Charactersitic frequency
        fstar = 3e8/(2*np.pi*self.armlength)

        # define f0 = f/2f*
        f0 = self.fdata/(2*fstar)

        # The last two elements are the position and the acceleration noise levels.
        Np, Na = 10**self.inj['log_Np'], 10**self.inj['log_Na']

        # Modelled Noise PSD
        C_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)

        # Extract noise auto-power. This is stationary, so it has no time dependence
        S1 = C_noise[0, 0, :]


        if self.params['modeltype'] == 'sph_sgwb' or self.params['modeltype'] == 'dwd_fg' or self.params['modeltype'] == 'dwd_sdg':
            if self.inj['injtype'] == 'point_source':
                npix = hp.nside2npix(10)
                Sgwmap = np.zeros(npix)
                
                # identify the pixel with the point source
                ps_id = hp.ang2pix(10, self.inj['theta'], self.inj['phi'])

                Omega_1mHz = 10**(self.inj['ln_omega0']) * (1e-3/25)**(self.inj['alpha'])
                
                H0 = 2.2*10**(-18)
                Sgwmap[ps_id] = np.sqrt(Omega_1mHz*(3/(4*(1e-3)**3))*(H0/np.pi)**2)
                blms_inj = hp.sphtfunc.map2alm(Sgwmap,lmax=self.blmax)
                self.alms_inj = self.blm_2_alm(blms_inj)

            summ_response_mat = np.sum(self.response_mat*self.alms_inj[None, None, None, None, :], axis=-1)
            # extra auto-power GW responses
            R1 = np.real(summ_response_mat[0, 0, :, :])
#
#        if self.params['modeltype'] == 'sph_sgwb':
#            alms_inj = self.blm_2_alm(self.inj['blms'])
#
#            # normalize
#            alms_inj = alms_inj/(alms_inj[0] * np.sqrt(4*np.pi))
#
#            summ_response_mat = np.sum(self.response_mat*alms_inj[None, None, None, None, :], axis=-1)
#
#            # extra auto-power GW responses
#            R1 = np.real(summ_response_mat[0, 0, :, :])

        else:
            # extra auto-power GW responses
            R1 = np.real(self.response_mat[0, 0, :, :])


        # SGWB signal levels of the mldc data
        Omega0, alpha = 10**self.inj['ln_omega0'], self.inj['alpha']

        # Hubble constant
        H0 = 2.2*10**(-18)

        # Calculate astrophysical power law noise
        Omegaf = Omega0*(self.fdata/25)**alpha


        # Power spectra of the SGWB
        Sgw = (3.0*(H0**2)*Omegaf)/(4*np.pi*np.pi*self.fdata**3)
        
        # Spectrum of the SGWB signal convoluted with the detector response tensor.
        S1_gw = Sgw[:, None]*R1

        delf = self.fdata[1] - self.fdata[0]

        single_channel_snr = np.sqrt( self.params['seglen'] * delf * np.sum(S1_gw/S1[:, None] ) )

        print('The single channel SNR is ' + str(single_channel_snr))

        with open( self.params['out_dir'] + '/snr.txt' , 'w' ) as file:
            file.write( 'The single channel SNR is ' + str(single_channel_snr) )
#        file.close()

        return 

def blip(paramsfile='params.ini',resume=False):
    '''
    The main workhorse of the bayesian pipeline.

    Input:
    Params File

    Output: Files containing evidence and pdfs of the parameters
    '''


    #  --------------- Read the params file --------------------------------

    # Initialize Dictionaries
    params = {}
    inj = {}

    config = configparser.ConfigParser()
    config.read(paramsfile)

    # Params Dict
    params['fmin']     = float(config.get("params", "fmin"))
    params['fmax']     = float(config.get("params", "fmax"))
    params['dur']      = float(config.get("params", "duration"))
    params['seglen']   = float(config.get("params", "seglen"))
    params['fs']       = float(config.get("params", "fs"))
    params['Shfile']   = config.get("params", "Shfile")
    params['mldc'] = int(config.get("params", "mldc"))
    params['datatype'] = str(config.get("params", "datatype"))
#    params['loadResponse'] = int(config.get("params", "loadResponse"))
#    params['loadCustom'] = int(config.get("params", "loadCustom"))
#    params['responsefile1']  = str(config.get("params", "responsefile1"))
#    params['responsefile2']  = str(config.get("params", "responsefile2"))
#    params['responsefile3']  = str(config.get("params", "responsefile3"))
    params['datafile']  = str(config.get("params", "datafile"))
    params['fref'] = float(config.get("params", "fref"))
    params['modeltype'] = str(config.get("params", "modeltype"))
    params['spectrum_model'] = str(config.get("params", "spectrum_model"))
    params['truncation_alpha'] = float(config.get("params", "truncation_alpha"))
    params['tdi_lev'] = str(config.get("params", "tdi_lev"))
    params['lisa_config'] = str(config.get("params", "lisa_config"))
    params['nside'] = int(config.get("params", "nside"))
    params['lmax'] = int(config.get("params", "lmax"))
    params['tstart'] = float(config.get("params", "tstart"))
    params['sampler'] = str(config.get("params", "sampler"))


    # Injection Dict
    inj['doInj']       = int(config.get("inj", "doInj"))
    inj['injtype']     = str(config.get("inj", "injtype"))
    inj['ln_omega0']   = np.log10(float(config.get("inj", "omega0")))
    inj['alpha']       = float(config.get("inj", "alpha"))
    inj['log_Np']      = np.log10(float(config.get("inj", "Np")))
    inj['log_Na']      = np.log10(float(config.get("inj", "Na")))
    
    
    inj['fg_type']     = str(config.get("inj", "fg_type"))
    inj['fg_spectrum'] = str(config.get("inj", "fg_spectrum"))
    if inj['fg_spectrum'] == 'truncated_powerlaw':
        inj['log_fcut']     = float(config.get("inj", "log_fcut"))
        inj['alpha2']      = float(config.get("inj", "alpha2"))
    if inj['fg_spectrum'] == 'broken_powerlaw':
        inj['alpha1']     = float(config.get("inj", "alpha1"))
        inj['log_A1']      = float(config.get("inj", "log_A1"))
        inj['log_A2']      = float(config.get("inj", "log_A2"))
    if inj['fg_type'] == 'breivik2020':
        inj['rh']          = float(config.get("inj", "rh"))
        inj['zh']          = float(config.get("inj", "zh"))
    if inj['fg_type'] == 'population' or inj['fg_spectrum'] == 'population':
        inj['popfile']     = str(config.get("inj","popfile"))
        inj['SNRcut']      = float(config.get("inj","SNRcut"))
        colnames = str(config.get("inj","columns"))
        colnames = colnames.split(',')
        inj['columns'] = colnames
        delimiter = str(config.get("inj","delimiter"))
        if delimiter == 'space':
            delimiter = ' '
        elif delimiter == 'tab':
            delimiter = '\t'
        inj['delimiter'] = delimiter

    # new sdg injection parameters:
#    inj['sdg_RA']      = float(config.get("inj", "sdg_RA"))
#    inj['sdg_DEC']     = float(config.get("inj", "sdg_DEC"))
#    inj['sdg_DIST']    = float(config.get("inj", "sdg_DIST"))
#    inj['sdg_RAD']     = float(config.get("inj", "sdg_RAD"))
#    inj['sdg_NUM']     = float(config.get("inj", "sdg_NUM"))


    if inj['injtype'] ==  'sph_sgwb':
        blm_vals = config.get("inj", "blms")
        blm_vals = blm_vals.split(',')

        num_blms = int(0.5*(params['lmax'] + 1) * (params['lmax'] + 2))
        blms = np.zeros(num_blms, dtype='complex')

        for ii in range(num_blms):
            blms[ii] = complex(blm_vals[ii])

        inj['blms'] = blms
    elif inj['injtype'] == 'point_source':
        inj['theta'] = float(config.get("inj", "theta"))
        inj['phi'] = float(config.get("inj", "phi"))

    # some run parameters
    params['out_dir']            = str(config.get("run_params", "out_dir"))
    params['doPreProc']          = int(config.get("run_params", "doPreProc"))
    params['input_spectrum']     = str(config.get("run_params", "input_spectrum"))
    params['projection'] = str(config.get("run_params", "projection"))
    params['FixSeed']            = str(config.get("run_params", "FixSeed"))
    params['seed']               = int(config.get("run_params", "seed"))
    verbose            = int(config.get("run_params", "verbose"))
    nlive              = int(config.get("run_params", "nlive"))
    nthread            = int(config.get("run_params", "Nthreads"))
    # checkpointing (dynesty only for now)
    params['checkpoint']            = int(config.get("run_params", "checkpoint"))
    params['checkpoint_interval']   = float(config.get("run_params", "checkpoint_interval"))

    # Fix random seed
    if params['FixSeed']:
        from tools.SetRandomState import SetRandomState as setrs
        seed = params['seed']
        randst = setrs(seed)
    else:
        if params['checkpoint']:
            raise TypeError("Checkpointing without a fixed seed is not supported. Set 'FixSeed' to true and specify 'seed'.")
        if resume:
            raise TypeError("Resuming from a checkpoint requires re-generation of data, so the random seed MUST be fixed.")
        randst = None


    if not resume:
        # Make directories, copy stuff
        # Make output folder
        subprocess.call(["mkdir", "-p", params['out_dir']])
    
        # Copy the params file to outdir, to keep track of the parameters of each run.
        subprocess.call(["cp", paramsfile, params['out_dir']])
        
        # Initialize lisa class
        lisa =  LISA(params, inj)
    else:
        print("Resuming a previous analysis. Regenerating data...")

    if params['sampler'] == 'dynesty':
        # Create engine
        if not resume:
            # multiprocessing
            if nthread > 1:
                pool = Pool(nthread)
            else:
                pool = None
            engine, parameters = dynesty_engine().define_engine(lisa, params, nlive, nthread, randst, pool=pool)    
        else:
            pool = None
            if nthread > 1:
                print("Warning: Nthread > 1, but multiprocessing is not supported when resuming a run. Pool set to None.")
                ## To anyone reading this and wondering why:
                ## The pickle calls used by Python's multiprocessing fail when trying to run the sampler after saving/reloading it.
                ## This is because pickling the sampler maps all its attributes to their full paths;
                ## e.g., dynesty_engine.isgwb_prior is named as src.dynesty_engine.dynesty_engine.isgwb_prior
                ## BUT the object itself is still e.g. <function dynesty_engine.isgwb_prior at 0x7f8ebcc27130>
                ## so we get an error like
                ## _pickle.PicklingError: Can't pickle <function dynesty_engine.isgwb_prior at 0x7f8ebcc27130>: \
                ##                        it's not the same object as src.dynesty_engine.dynesty_engine.isgwb_prior
                ## See e.g. https://stackoverflow.com/questions/1412787/picklingerror-cant-pickle-class-decimal-decimal-its-not-the-same-object
                ## After too much time and sanity spent trying to fix this, I have admitted defeat.
                ## Feel free to try your hand -- maybe you're the chosen one. Good luck.
                
            engine, parameters = dynesty_engine.load_engine(params,randst,pool)
        ## run sampler
        if params['checkpoint']:
            checkpoint_file = params['out_dir']+'/checkpoint.pickle'
            t1 = time.time()
            post_samples, logz, logzerr = dynesty_engine.run_engine_with_checkpointing(engine,parameters,params['checkpoint_interval'],checkpoint_file)
            t2= time.time()
            print("Elapsed time to converge: {} s".format(t2-t1))
        else:
            t1 = time.time()
            post_samples, logz, logzerr = dynesty_engine.run_engine(engine)
            t2= time.time()
            print("Elapsed time to converge: {} s".format(t2-t1))
        if nthread > 1:
            engine.pool.close()
            engine.pool.join()
        # Save posteriors to file
        np.savetxt(params['out_dir'] + "/post_samples.txt",post_samples)
        np.savetxt(params['out_dir'] + "/logz.txt", logz)
        np.savetxt(params['out_dir'] + "/logzerr.txt", logzerr)

    elif params['sampler'] == 'emcee':

        # Create engine
        engine, parameters, init_samples = emcee_engine.define_engine(lisa, params, nlive, randst)
        post_samples = emcee_engine.run_engine(engine, init_samples)

        # Save posteriors to file
        np.savetxt(params['out_dir'] + "/post_samples.txt",post_samples)

    else:
        raise TypeError('Unknown sampler model chosen. Only dynesty & emcee are supported')


    # Save parameters as a pickle
    with open(params['out_dir'] + '/config.pickle', 'wb') as outfile:
        pickle.dump(params, outfile)
        pickle.dump(inj, outfile)
        pickle.dump(parameters, outfile)

    print("\nMaking posterior Plots ...")
    plotmaker(params, parameters, inj)
#    if params['modeltype'] not in ['isgwb','isgwb_only','noise_only']:
#        print("\nMaking posterior skymap ...")
#        mapmaker(params, post_samples, parameters, coord=params['projection'])
    # open_img(params['out_dir'])

if __name__ == "__main__":

    if len(sys.argv) != 2:
        if sys.argv[2] == 'resume':
            blip(sys.argv[1],resume=True)
        else:
            raise ValueError('Provide (only) the params file as an argument')
    else:
        blip(sys.argv[1])
