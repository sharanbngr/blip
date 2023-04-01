import numpy as np
import scipy.signal as sg
from src.instrNoise import instrNoise
from src.geometry import geometry
from src.sph_geometry import sph_geometry
from src.populations import populations
import matplotlib.pyplot as plt
import healpy as hp
from astropy import units as u
from astropy import coordinates as cc
from astropy.coordinates import SkyCoord
from math import pi
import os
import legwork as lw

class LISAdata(populations):

    '''
    Class for lisa data. Includes methods for generation of gaussian instrumental noise, and generation
    of isotropic stochastic background. Any eventually signal models should be added as methods here. This
    has the Antennapatterns class as a super class.
    '''

    def __init__(self, params, inj):
        self.params = params
        self.inj = inj
        self.armlength = 2.5e9 ## armlength in meters
#        geometry.__init__(self)
#        sph_geometry.__init__(self)


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


    


    

    def add_sgwb_data(self, injmodel, tbreak = 0.0):
        
        
        

#        cspeed = 3e8 #m/s
#
#        ## define the splice segment duration
#        tsplice = 1e4
#        delf  = 1.0/tsplice
#
#        ## the segments to be splices are half-overlapping
#        nsplice = 2*int(self.params['dur']/tsplice) + 1
#
#        ## arrays of segmnent start and mid times
#        #tmids = (tsplice/2.0) * np.arange(nsplice) + (tsplice/2.0)
#
#        ## arrays of segmnent start and mid times
#        tmids = self.params['tstart'] + tbreak +  (tsplice/2.0) * np.arange(nsplice) + (tsplice/2.0)
#
#        ## Number of time-domain points in a splice segment
#        N = int(self.params['fs']*tsplice)
#        halfN = int(0.5*N)
#
#        ## leave out f = 0
#        frange = np.fft.rfftfreq(N, 1.0/self.params['fs'])[1:]
#
#        ## the charecteristic frequency of LISA, and the scaled frequency array
#        fstar = 3e8/(2*np.pi*self.armlength)
#        f0 = frange/(2*fstar)

        ## Response matrix : shape (3 x 3 x freq x time) if isotropic
        ## set up different use cases
        ## pixel basis computes response in conjunction with skymap, so skip this step
#        if self.inj['injtype'] == 'multi':
#            if multi == 'a':
#                response_mat = self.add_astro_signal_a(f0,tmids)
#            elif multi == 'i':
#                response_mat = self.add_astro_signal_i(f0,tmids)
#            else:
#                raise ValueError("Invalid specification of 'multi' type. Can be 'a' (anisotropic) or 'i' (isotropic).")
#        elif self.inj['injtype'] == 'astro' and self.inj['injbasis'] == 'pixel':
#            pass
#        else:
#            if self.inj['injtype'] == 'astro' and self.inj['injbasis'] == 'sph_lmax':
#                signal_args= (f0,tmids,self.inj_almax)
#            else:
#                signal_args = (f0,tmids)
#        
#            response_mat = self.add_astro_signal(*signal_args)

        ## Cholesky decomposition to get the "sigma" matrix
#        H0 = 2.2*10**(-18) ## in SI units
        
        
        
        
#        ## compute spectra and save Omega(1mHz) for later
#        if self.inj['injtype'] == 'multi':
#            if multi == 'a':
#                Omegaf = (10**self.inj['log_omega0_a']) * (frange/self.params['fref'])**self.inj['alpha_a'] \
#                                * 0.5 * (1+np.tanh((self.inj['f_cut_a']-frange)/self.inj['f_scale_a']))
#                Omega_1mHz = (10**self.inj['log_omega0_a']) * (1e-3/self.params['fref'])**self.inj['alpha_a'] \
#                                * 0.5 * (1+np.tanh((self.inj['f_cut_a']-1e-3)/self.inj['f_scale_a']))
#            elif multi == 'i':
#                Omegaf = (10**self.inj['log_omega0_i'])*(frange/self.params['fref'])**self.inj['alpha_i']
#                Omega_1mHz = (10**self.inj['log_omega0_i'])*(1e-3/self.params['fref'])**self.inj['alpha_i']
#            else:
#                raise ValueError("Invalid specification of multi. Can be 'a' (anisotropic) or 'i' (isotropic).")
#        elif self.inj['spectral_inj'] == 'powerlaw':
#            Omegaf = (10**self.inj['log_omega0'])*(frange/(self.params['fref']))**self.inj['alpha']
#            Omega_1mHz = 10**(self.inj['log_omega0']) * (1e-3/self.params['fref'])**(self.inj['alpha'])
#        elif self.inj['spectral_inj'] == 'broken_powerlaw':
#            alpha_2 = self.inj['alpha1'] - 0.667
#            Omegaf = ((10**self.inj['log_A1'])*(frange/self.params['fref'])**self.inj['alpha1'])/(\
#                     1 + (10**self.inj['log_A2'])*(frange/self.params['fref'])**alpha_2)
#            Omega_1mHz= ((10**self.inj['log_A1'])*(1e-3/self.params['fref'])**self.inj['alpha1'])/(\
#                        1 + (10**self.inj['log_A2'])*(1e-3/self.params['fref'])**alpha_2)
#        elif self.inj['spectral_inj'] == 'broken_powerlaw_2':
#            delta = 0.1
#            Omegaf = (10**self.inj['log_omega0'])*(frange/self.inj['f_break'])**(self.inj['alpha1']) \
#                    * (0.5*(1+(frange/self.inj['f_break'])**(1/delta)))**((self.inj['alpha1']-self.inj['alpha2'])*delta)
#            Omega_1mHz = (10**self.inj['log_omega0'])*(1e-3/self.inj['f_break'])**(self.inj['alpha1']) \
#                    * (0.5*(1+(1e-3/self.inj['f_break'])**(1/delta)))**((self.inj['alpha1']-self.inj['alpha2'])*delta)
#        elif self.inj['spectral_inj'] == 'free_broken_powerlaw':
#            Omegaf = ((10**self.inj['log_A1'])*(frange/self.params['fref'])**self.inj['alpha1'])/(\
#                     1 + (10**self.inj['log_A2'])*(frange/self.params['fref'])**self.inj['alpha2'])
#            Omega_1mHz = ((10**self.inj['log_A1'])*(1e-3/self.params['fref'])**self.inj['alpha1'])/(\
#                        1 + (10**self.inj['log_A2'])*(1e-3/self.params['fref'])**self.inj['alpha2'])
#        elif self.inj['spectral_inj'] == 'truncated_broken_powerlaw':
#            delta = 0.1
#            Omegaf = (10**self.inj['log_omega0'])*(frange/self.inj['f_break'])**(self.inj['alpha1']) \
#                    * (0.5*(1+(frange/self.inj['f_break'])**(1/delta)))**((self.inj['alpha1']-self.inj['alpha2'])*delta) \
#                    * 0.5 * (1 + np.tanh((self.inj['f_cut'] - frange)/self.inj['f_scale']))
#            Omega_1mHz = (10**self.inj['log_omega0'])*(1e-3/self.inj['f_break'])**(self.inj['alpha1']) \
#                    * (0.5*(1+(1e-3/self.inj['f_break'])**(1/delta)))**((self.inj['alpha1']-self.inj['alpha2'])*delta) \
#                    * 0.5 * (1 + np.tanh((self.inj['f_cut'] - 1e-3)/self.inj['f_scale']))
#        elif self.inj['spectral_inj'] == 'truncated_powerlaw':
#            Omegaf = (10**self.inj['log_omega0'])*(frange/(self.params['fref']))**self.inj['alpha'] \
#                    * 0.5 * (1 + np.tanh((self.inj['f_cut'] - frange)/self.inj['f_scale']))
#            Omega_1mHz = 10**(self.inj['log_omega0']) * (1e-3/self.params['fref'])**(self.inj['alpha']) \
#                    * 0.5 * (1 + np.tanh((self.inj['f_cut'] - 1e-3)/self.inj['f_scale']))
#
#        elif self.inj['spectral_inj'] == 'population':
#            print("Constructing foreground spectrum from DWD population...")
#            ## factor of two b/c (h_A,h_A*)~h^2~1/2 * S_A
#            ## additional factor of 2 b/c S_GW = 2 * S_A
#            Sgw = self.pop2spec(self.inj['popfile'],frange,self.params['dur']*u.s,names=self.inj['columns'],sep=self.inj['delimiter'])*4
#            Omega_1mHz = Sgw[np.argmin(np.abs(frange - 1e-3))]/((3/(4*(1e-3)**3))*(H0/np.pi)**2)
#        else:
#            raise ValueError("Unknown spectral injection selected. Can be powerlaw, broken_powerlaw, free_broken_powerlaw, or population.")
#        
#        # Spectrum of the SGWB from Omegaf (population version goes directly to the spectrum from binary strains and frequencies)
#        if self.inj['spectral_inj'] != 'population':
#            Sgw = Omegaf*(3/(4*frange**3))*(H0/np.pi)**2    
        N = self.Injection.Npersplice
        halfN = int(0.5*N)
        
        injmodel_args = [injmodel.truevals[parameter] for parameter in injmodel.spectral_parameters]
        
        Sgw = injmodel.compute_Sgw(self.Injection.frange,injmodel_args)
        
        ## save frozen injected spectra.
        ## This has to be handled slightly differently for the population case
        ## (there are subtleties with the binning; the true spectrum has a frequency resolution of delta(f) = 1/t_obs
        ##   but we inject segment-by-segment, and the segment frequency resolution is different. Because we add all segments together
        ##   in time-domain and then fft, the *final* product is equivalent, but for e.g. plotting, we want the true spectrum)
        if hasattr(injmodel,"ispop") and injmodel.ispop:
            injmodel.frozen_spectra = injmodel.population.Sgw_true
        else:
            injmodel.frozen_spectra = Sgw
        
        ## the spectrum of the frequecy domain gaussian for ifft
        norms = np.sqrt(self.params['fs']*Sgw*N)/2

        ## index array for one segment
        t_arr = np.arange(N)

        ## the window for splicing
        splice_win = np.sin(np.pi * t_arr/N)

        # deals with projection parameter to use in the hp.mollview functions below
#        if self.params['projection'] is None:
#            coord = 'E'
#        elif self.params['projection']=='G' or self.params['projection']=='C':
#            coord = ['E',self.params['projection']]
#        elif self.params['projection']=='E':
#            coord = self.params['projection']
#        else:  
#            raise TypeError('Invalid specification of projection, projection can be E, G, or C')

        ## Loop over splice segments
        for ii in range(self.Injection.nsplice):
            ## move frequency to be the zeroth-axis, then cholesky decomp
            L_cholesky = norms[:, None, None] *  np.linalg.cholesky(np.moveaxis(injmodel.inj_response_mat[:, :, :, ii], -1, 0))

#            if self.inj['injtype'] == 'isgwb':
#                ## move frequency to be the zeroth-axis, then cholesky decomp
#                L_cholesky = norms[:, None, None] *  np.linalg.cholesky(np.moveaxis(injmodel.response_mat[:, :, :, ii], -1, 0))
#
#            elif self.inj['injtype'] == 'sph_sgwb':
#
#                if ii == 0:
#
#                    ## need to set up a few things before doing the spherical harmonic inj
#
#                    ## extract alms
#                    self.alms_inj = self.blm_2_alm(self.inj['blms'])
#
#                    ## normalize
#                    self.alms_inj = self.alms_inj/(self.alms_inj[0] * np.sqrt(4*np.pi))
#
#                    ## extrct only the non-negative components
#                    alms_non_neg = self.alms_inj[0:hp.Alm.getsize(self.almax)]
#
#                    ## response matrix summed over Ylms
#                    summ_response_mat = np.einsum('ijklm,m', response_mat, self.alms_inj)
#
#                    # converts alm_inj into a healpix max to be plotted and saved
#                    # Plot with twice the analysis nside for better resolution
#                    skymap_inj = hp.alm2map(alms_non_neg, 2*self.params['nside'])
#
#                    Omegamap_inj = Omega_1mHz * skymap_inj
#
#                    hp.mollview(Omegamap_inj, coord=coord, title='Injected angular distribution map $\Omega (f = 1 mHz)$', unit="$\\Omega(f= 1mHz)$")
#                    hp.graticule()
#                    
#                    plt.savefig(self.params['out_dir'] + '/inj_skymap.png', dpi=150)
#                    print('saving injected skymap at ' +  self.params['out_dir'] + '/inj_skymap.png')
#                    plt.close()
#                    
#
#                ## move frequency to be the zeroth-axis, then cholesky decomp
#                L_cholesky = norms[:, None, None] *  np.linalg.cholesky(np.moveaxis(summ_response_mat[:, :, :, ii], -1, 0))
#
#            elif self.inj['injtype'] == 'astro':
#                if ii==0:
#                    ## pick between different anisotropic astrophysical injections
#                    if self.inj['spatial_inj'] == 'breivik2020':
#                        ## toy model foreground
#                        astro_map, log_astro_map = self.generate_galactic_foreground(self.inj['rh'], self.inj['zh'])
#                    elif self.inj['spatial_inj'] == 'population':
#                        ## generate skymap
#                        print("Constructing skymap from DWD population...")
#                        astro_map, log_astro_map = self.pop2map(self.inj['popfile'],self.params['nside'],self.params['dur']*u.s,
#                                                                  self.params['fmin'],self.params['fmax'],names=self.inj['columns'],sep=self.inj['delimiter'])
#                    elif self.inj['spatial_inj'] == 'sdg':
#                        astro_map, log_astro_map = self.generate_sdg(self.inj['sdg_RA'], self.inj['sdg_DEC'], self.inj['sdg_DIST'], self.inj['sdg_RAD'], self.inj['sdg_NUM'])
#                    elif self.inj['spatial_inj'] == 'point_source':
#                        astro_map, log_astro_map = self.generate_point_source(self.inj['theta'],self.inj['phi'])
#                    elif self.inj['spatial_inj'] == 'two_point':
#                        astro_map, log_astro_map = self.generate_two_point_source(self.inj['theta_1'],self.inj['phi_1'],self.inj['theta_2'],self.inj['phi_2'])
#                    elif self.inj['spatial_inj'] == 'isotropic':
#                        astro_map = np.ones(hp.nside2npix(self.params['nside']))
#                    else:
#                        raise ValueError("Unsupported spatial injection. Currentlys supported: breivik2020, sdg, population, point_source, two_point_source.")
#            
#                    if self.inj['injbasis'] == 'sph':
#                        ## convert to blms
#                        astro_sph = self.skymap_pix2sph(astro_map, blmax=self.blmax)
#                        ## save blms for truevals
#                        self.inj['astro_blms'] = astro_sph
#                        ## extract alms
#                        self.alms_inj = self.blm_2_alm(astro_sph)
#    
#                        ## normalize
#                        self.alms_inj = self.alms_inj/(self.alms_inj[0] * np.sqrt(4*np.pi))
#    
#                        ## extrct only the non-negative components
#                        alms_non_neg = self.alms_inj[0:hp.Alm.getsize(self.almax)]
#                        ## response matrix summed over Ylms
#                        summ_response_mat = np.einsum('ijklm,m', response_mat, self.alms_inj)
#    
#                        # converts alm_inj into a healpix map to be plotted and saved
#                        # Plot with twice the analysis nside for better resolution
#                        skymap_inj = hp.alm2map(alms_non_neg, self.params['nside'])
#                        
#                    elif self.inj['injbasis'] == 'sph_lmax':
#                        ## version with injection lmax decoupled from analysis lmax
#                        
#                        ## convert to blms
#                        astro_sph = self.skymap_pix2sph(astro_map, blmax=self.inj['inj_lmax'])
#                        ## save blms for truevals
#                        self.inj['astro_blms'] = astro_sph
#                        ## extract alms
#                        self.alms_inj = self.inj_blm_2_alm(astro_sph)
#    
#                        ## normalize
#                        self.alms_inj = self.alms_inj/(self.alms_inj[0] * np.sqrt(4*np.pi))
#                        
#                        ## get almax from blmax
#                        almax_inj = 2*self.inj['inj_lmax']
#                        ## extract only the non-negative components
#                        alms_non_neg = self.alms_inj[0:hp.Alm.getsize(almax_inj)]
#                        ## response matrix summed over Ylms
#                        summ_response_mat = np.einsum('ijklm,m', response_mat, self.alms_inj)
#    
#                        # converts alm_inj into a healpix map to be plotted and saved
#                        # Plot with twice the analysis nside for better resolution
#                        skymap_inj = hp.alm2map(alms_non_neg, self.params['nside'])
#                    elif self.inj['injbasis'] == 'pixel':
#                        print("Warning: pixel-basis injections are still under development, results may be erroneous.")
#                        dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])
#                        skymap_inj = astro_map/(np.sum(astro_map)*(dOmega/(8*np.pi)))
#                        summ_response_mat = self.add_astro_signal(f0,tmids,skymap_inj)                     
#                        
#                    Omegamap_inj = Omega_1mHz * skymap_inj
#                    
#                    ## save injected skymap for use elsewhere (i.e., diag_spectra)
#                    self.skymap_inj = skymap_inj
#    
#                    ## also save the final healpix map to a datafile
#                    np.savetxt(self.params['out_dir'] +"/injected_healpix_skymap.dat",skymap_inj)
#                    
#                    hp.mollview(Omegamap_inj, title='Injected angular distribution map $\Omega (f = 1 mHz)$', unit="$\\Omega(f= 1mHz)$")
#                    hp.graticule()
#                    
#                    plt.savefig(self.params['out_dir'] + '/inj_skymap.png', dpi=150)
#                    print('saving injected skymap at ' +  self.params['out_dir'] + '/inj_skymap.png')
#                    plt.close()
#                    
#                    
#                    hp.mollview(astro_map, title='Simulated astrophysical skymap')
#                    hp.graticule()
#                    plt.savefig(self.params['out_dir'] + '/pre_inj_skymap.png', dpi=150)
#                    print('saving simulated skymap at ' +  self.params['out_dir'] + '/pre_inj_skymap.png')
#                    plt.close()
#                    if self.inj['injbasis']!='pixel':
#                        hp.mollview(skymap_inj, title='Simulated astrophysical alm map')
#                        hp.graticule()
#                        plt.savefig(self.params['out_dir'] + '/pre_inj_almmap.png', dpi=150)
#                        print('saving simulated skymap at ' +  self.params['out_dir'] + '/pre_inj_almmap.png')
#                        plt.close()
#
#                ## move frequency to be the zeroth-axis, then cholesky decomp
#                L_cholesky = norms[:, None, None] *  np.linalg.cholesky(np.moveaxis(summ_response_mat[:, :, :, ii], -1, 0))
#            
#            elif self.inj['injtype'] == 'multi':
#                if ii==0 and multi=='a':
#                    ## pick between different anisotropic astrophysical injections
#                    if self.inj['spatial_inj'] == 'breivik2020':
#                        ## toy model foreground
#                        astro_map, log_astro_map = self.generate_galactic_foreground(self.inj['rh'], self.inj['zh'])
#                    else:
#                        raise ValueError("Multi-SGWB injection protoype currently only supports breivik2020 model galaxy.")
#            
#                    if self.inj['injbasis'] == 'sph':
#                        ## convert to blms
#                        astro_sph = self.skymap_pix2sph(astro_map, blmax=self.blmax)
#                        ## save blms for truevals
#                        self.inj['astro_blms'] = astro_sph
#                        ## extract alms
#                        self.alms_inj = self.blm_2_alm(astro_sph)
#    
#                        ## normalize
#                        self.alms_inj = self.alms_inj/(self.alms_inj[0] * np.sqrt(4*np.pi))
#    
#                        ## extrct only the non-negative components
#                        alms_non_neg = self.alms_inj[0:hp.Alm.getsize(self.almax)]
#                        ## response matrix summed over Ylms
#                        summ_response_mat = np.einsum('ijklm,m', response_mat, self.alms_inj)
#    
#                        # converts alm_inj into a healpix map to be plotted and saved
#                        # Plot with twice the analysis nside for better resolution
#                        skymap_inj = hp.alm2map(alms_non_neg, self.params['nside'])
#                        
#                    else:
#                        raise ValueError("Multi-SGWB injection protoype currently only supports sph-basis spatial injections.")
#                        
#                    Omegamap_inj = (Omega_1mHz) * skymap_inj
#                    
#                    ## save injected skymap for use elsewhere (i.e., diag_spectra)
#                    self.skymap_inj = skymap_inj
#    
#                    ## also save the final healpix map to a datafile
#                    np.savetxt(self.params['out_dir'] +"/injected_healpix_skymap.dat",skymap_inj)
#                    
#                    hp.mollview(Omegamap_inj, title='Injected angular distribution map $\Omega (f = 1 mHz)$', unit="$\\Omega(f= 1mHz)$")
#                    hp.graticule()
#                    
#                    plt.savefig(self.params['out_dir'] + '/inj_skymap.png', dpi=150)
#                    print('saving injected skymap at ' +  self.params['out_dir'] + '/inj_skymap.png')
#                    plt.close()
#                    
#                    
#                    hp.mollview(astro_map, title='Simulated astrophysical skymap')
#                    hp.graticule()
#                    plt.savefig(self.params['out_dir'] + '/pre_inj_skymap.png', dpi=150)
#                    print('saving simulated skymap at ' +  self.params['out_dir'] + '/pre_inj_skymap.png')
#                    plt.close()
#                    if self.inj['injbasis']!='pixel':
#                        hp.mollview(skymap_inj, title='Simulated astrophysical alm map')
#                        hp.graticule()
#                        plt.savefig(self.params['out_dir'] + '/pre_inj_almmap.png', dpi=150)
#                        print('saving simulated skymap at ' +  self.params['out_dir'] + '/pre_inj_almmap.png')
#                        plt.close()
#
#                if multi=='i':
#                    ## move frequency to be the zeroth-axis, then cholesky decomp
#                    L_cholesky = norms[:, None, None] *  np.linalg.cholesky(np.moveaxis(response_mat[:, :, :, ii], -1, 0))
#                elif multi=='a':
#                    ## move frequency to be the zeroth-axis, then cholesky decomp
#                    L_cholesky = norms[:, None, None] *  np.linalg.cholesky(np.moveaxis(summ_response_mat[:, :, :, ii], -1, 0))
#                else:
#                    raise ValueError("Invalid specification of 'multi' type. Can be 'a' (anisotropic) or 'i' (isotropic).")
                
            ## generate standard normal complex data frist
            z_norm = np.random.normal(size=(self.Injection.frange.size, 3)) + 1j * np.random.normal(size=(self.Injection.frange.size, 3))

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
        Omegaf = (10**self.inj['log_omega0'])*(freqs/(self.params['fref']))**self.inj['alpha']


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

    