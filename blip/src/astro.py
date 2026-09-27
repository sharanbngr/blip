import numpy as np
import jax.numpy as jnp
import legwork as lw
import pandas as pd
from blip.src.instrNoise import instrNoise
from blip.src.geometry import geometry
from blip.src.sph_geometry import sph_geometry
from scipy.interpolate import interp1d as intrp
import matplotlib.pyplot as plt
import healpy as hp
from astropy import units as u
from astropy import coordinates as cc
from astropy.coordinates import SkyCoord
from scipy.signal import medfilt

class Population():

    '''
    Class for handling binary populations. Includes methods for loading, parsing, calculating physical quantities,
    and preparing data for use in some of the makeLISAdata.py signal simulation methods. 
    '''

    def __init__(self, params, inj, frange, popdict, seed=None, map_only=False):
        '''
        Produces a population object with an attached skymap and spectrum.
        
        Note that we don't carry around the entire set of binaries, just the overall population-level data.
        
        This is a fast way to accurately approximate the strain PSD of the unresolved DWD population in frequency-domain, 
        but does result in some smoothing of the spectrum, so sharp features may be reduced in the simulation process. 
        Also note that each bin is phase-averaged, which will underestimate the variance in a small fraction of bins.
        
        NB -- should implement a full time-domain approach when time allows, as it will be more precise. This is a good (and efficient) approximation, though.
        
        Arguments
        -------------------
        params (dict)            : params dict
        inj (dict)               : injection dict
        frange (array of floats) : injection splice fft frequencies
        popdict (str)            : the populations dict corresponding to the desired population (allows for multiple populations)
        map_only (bool)          : Whether to only compute the skymap. Used for models that rely on a fixed population skymap.
        
        '''
        self.params = params
        self.inj = inj
        self.frange = frange
        
        self.popdict = popdict
        
        ## load the population
        if self.popdict['coldict'] is None:
            pop = self.load_population(self.popdict['popfile'],self.params['fmin'],self.params['fmax'],
                                       names=self.popdict['columns'],sep=self.popdict['delimiter'],
                                       seed=seed)
        else:
            pop = self.load_population(self.popdict['popfile'],self.params['fmin'],self.params['fmax'],
                                       names=self.popdict['columns'],sep=self.popdict['delimiter'],
                                       coldict=self.popdict['coldict'],seed=seed)
        
        ## get the skymap
        df = self.frange[1] - self.frange[0]
        self.skymap = self.pop2map(pop,self.params['nside'],df*u.Hz,self.params['fmin'],self.params['fmax'])
        ## also compute the spherical harmonic transform if the injection is using the spherical harmonic basis
        if self.inj['inj_basis']=='sph':
            self.sph_skymap = skymap_pix2sph(self.skymap,self.inj['inj_lmax'])
        
        ## spectrum
        if not map_only:
            ## PSD at injection frequency binning
            self.PSD = self.pop2spec(pop,self.frange,self.params['seglen']*u.s,
                                     SNR_cut=self.popdict['snr_cut'],return_median=False,plot=True,saveto=params['out_dir'])
            ## PSD at data frequencies
            self.fftfreqs = np.fft.rfftfreq(int(self.params['fs']*self.params['seglen']),1/self.params['fs'])[1:]
            self.PSD_true = self.pop2spec(pop,self.fftfreqs,self.params['seglen']*u.s,return_median=False,plot=False)[np.logical_and(self.fftfreqs >=  self.params['fmin'] , self.fftfreqs <=  self.params['fmax'])]
            self.frange_true = self.fftfreqs[np.logical_and(self.fftfreqs >=  self.params['fmin'] , self.fftfreqs <=  self.params['fmax'])]
            self.Sgw = self.PSD
            ## reweight to match what we are injecting at data frequencies
            self.Sgw_true = self.PSD_true #* (self.params['seglen']/self.params['tsplice'])
        
        
    def rebin_PSD(self,fs_new):
        '''
        Function to correctly interpolate the population spectrum to new frequencies without violating conservation of energy
        '''
        delta_f_old = self.delta_f
        delta_f_new = fs_new[1] - fs_new[0]
        return (delta_f_new/delta_f_old)*self.PSD_interp(fs_new)
    
    def Sgw_wrapper(self,frange,spoof_arg=None):
        '''
        This is a wrapper function to allow the population spectrum to play well with some of the generic Injection-handling code.
        
        Evaluated at the injection frequencies.
        '''
        if hasattr(frange,"__len__"):
            return self.Sgw
        else:
            return self.Sgw[np.argmin(np.abs(self.frange - 1e-3))]
    
    def Sgw_wrapper_true(self,frange,spoof_arg=None):
        '''
        This is a wrapper function to allow the population spectrum to play well with some of the generic Injection-handling code.
        Evaluated at the data frequencies.
        '''
        if hasattr(frange,"__len__"):
            return self.Sgw_true
        else:
            return self.Sgw_true[np.argmin(np.abs(self.frange_true - 1e-3))]
    
    def omegaf_wrapper(self,fs,spoof_arg=None):
        '''
        This is a wrapper function to allow the pupulation spectrum to play well with some of the generic Injection-handling code.
        '''
        H0 = 2.2*10**(-18)
        omegaf = self.Sgw_wrapper(fs)/((3/(4*(fs)**3))*(H0/np.pi)**2)
        return omegaf
    
    @staticmethod
    def load_population(popfile,fmin,fmax,coldict={'f':'f','h':'h','lat':'lat','long':'long'},unitdict={'f':u.Hz,'lat':u.rad,'long':u.rad},
                        sep=' ',seed=None,**read_csv_kwargs):
        # Would also be good to have an option for giving binary parameters and computing the strain here?
        '''
        Function to load a population file and store relevant data. For now this assumes columns with labels ['f','h',lat','long'].
        Assumes LEGWORK definition of binary strain (see https://arxiv.org/abs/2111.08717)
        
        Arguments:
            popfile (str)     : '/path/to/binary/population/data/file.csv'
            fmin (float)      : Minimum analysis frequency
            fmax (float)      : Maximum analysis frequency
            coldict (dict)    : Dictionary explaining which columns correspond to which quantities (allows for users to specify different column names)
            unitdict (dict)   : Dictionary specifying units of each column.
            sep (str)         : File delimiter. Overwritten if sep or delimiter is specified in **read_csv_kwargs. Default is ' '.
            **read_csv_kwargs : Optional keyword arguments to be passed to pd.read_csv()
        Returns:
            fs (array)     : Binary frequencies
            hs (array)     : Binary strain amplitudes
            lats (array)   : Binary latitudes in degrees
            longs (array)  : Binary longitudes in degrees
        '''
        ## handle conflicting kwargs
        if 'sep' in read_csv_kwargs.keys():
            sep = read_csv_kwargs['sep']
            del read_csv_kwargs['sep']
        elif 'delimiter' in read_csv_kwargs.keys():
            sep = read_csv_kwargs['delimiter']
            del read_csv_kwargs['delimiter']
        ## load
        dwds = pd.read_csv(popfile,sep=sep,**read_csv_kwargs)
        ## unit conversions and assignments as needed
        fs = (dwds[coldict['f']].to_numpy()*unitdict['f']).to(u.Hz).value
        hs = dwds[coldict['h']].to_numpy()
        lats = (dwds[coldict['lat']].to_numpy()*unitdict['lat']).to(u.deg).value
        longs = (dwds[coldict['long']].to_numpy()*unitdict['long']).to(u.deg).value

        ## inclination handling
        if 'inc' in coldict.keys():
            cos_incs = np.cos(dwds[coldict['inc']].to_numpy()) ## assumed radians
        elif 'cosi' in coldict.keys():
            cos_incs = dwds[coldict['cosi']].to_numpy()
        else:
            print("No inclinations provided. Randomly drawing from cosi ~ U(-1,1)")
            if seed is None:
                print("Warning: No random seed was provided to the Population object, inclinations will not be reproducible.")
            rng = np.random.default_rng(seed)
            cos_incs = 2*rng.random(size=len(fs)) - 1

        ## filter to frequency band
#        f_filter = (fs >= fmin) & (fs <= fmax)
        ## generate pop dict
        pop = {'fs':fs,'hs':hs,'lats':lats,'longs':longs,'cos_incs':cos_incs}
        return pop
        
        
    @staticmethod
    def get_binary_psd(hs,cos_incs,df):
        '''
        Function to calculate PSD of catalogue binaries. Assumed monochromatic.

        We assume a definition of amplitude such that A = h0, so

        h(t) = h+(t) + hx(t),

        h+(t) = (1+cos^2(i)) * A * cos(2*omega*t + phi0),

        and

        hx(t) = 2 * cosi * A * sin(2omega*t + phi0).

        The strain power is then

        <h(t)^2> = (1+cos^2(i))^2 * A^2 + 4 * cos^2(i) * A^2

        which for optimal inclination (i=0, face-on) yields

        <h(t)^2> = 8A^2.

        The PSD contribution from the monochromatic binary at
        frequency resolution df = 1/Tobs is then

        PSD = (1/df) * < h(t)^2 >

        which in the optimal inclination case is

        PSD = 8 * (1/df) * A^2.

        Note that by combining the + and x contributions prior to convolution with the LISA
        response functions, we implicitly assume that the overall population produces an
        unpolarized stochastic signal. This is statistically true for any stochastic signal produced
        by a population of binaries with uniformly distributed inclinations, but the assumption may
        break down in some cases.
        

        Also note that there is an alternate definition for the amplitude A,
        such that A = 2h0. If used here, this will result in an erroneous factor of 4 in the PSD.

        Arguments:
            hs (1D array of floats) : Binary strains.
            cos_incs (1D array of floats) : cosine of binary inclinations
            df (astropy Quantity, frequency units) : Binning frequency resolution.
        
        Returns:
            binary_psds (1D array of floats): Monochromatic PSDs for each binary
        '''

        h2s = (1+cos_incs**2)**2 * hs**2 + 4 * cos_incs**2 * hs**2


        binary_psds = h2s/df
        
        return binary_psds
    
    @classmethod
    def get_snr(cls,fs,hs,cos_incs,t_obs,noise_PSD='default'):
        ## need to update this to take either legwork or local noise PSD
        '''
        Function to get SNRs of catalogue binaries, given their frequencies/strains, observation time, and a detector PSD.
        Assumes monochromatic systems.
        
        Arguments:
            fs (1D array of floats) : Binary frequencies. Assumed monochromatic.
            hs (1D array of floats) : Binary strains.
            t_obs (astropy Quantity, time units) : Observation time.
            PSD (varies) : If 'default', uses the Legwork LISA PSD without confusion noise. 
                           If an array, used as the detector PSD at each frequency value in fs
        
        Returns:
            SNRs (array of floats) : Binary SNRs.
        '''
        ## assuming monochromatic systems, get SNRs and filter any DWDs with SNR>7. Again per Thiele+22:
        if noise_PSD=='default':
            noise_PSD = lw.psd.lisa_psd(fs,t_obs=t_obs,confusion_noise='robson19')
        elif noise_PSD=='no_fg':
            noise_PSD = lw.psd.lisa_psd(fs,t_obs=t_obs,confusion_noise=None)
        ## we want the SNRs for the resolved binaries at the full frequency resolution
        SNRs = cls.get_binary_psd(hs,cos_incs,1/t_obs)/(4*noise_PSD)
        return SNRs
    
    @staticmethod
    def filter_by_snr(data,SNRs,SNR_cut=7,get_type='unresolved'):
        '''
        Function to filter DWD data by SNR. Can return either unresolved (SNR < SNR_cut) or resolved (SNR > SNR_cut) binaries.
        
        Arguments:
            data (1D array of floats) : Binary population data of your choice (or list thereof), corresponding to the given SNRs
            SNRs (1D array of floats) : SNR value for each system corresponding to data
            SNR_cut (float) : Value of SNR that delineates resolved and unresolved binaries. Default is SNR = 7.
            get_type (str) : Whether to return the resolved or unresolved binaries. Default is unresolved.
        
        Returns:
            data_filt : Filtered arrays of frequencies and strains.
        '''
        if type(data) is not list:
            data = [data]
        if get_type=='unresolved':
            data_filt = [data_i[SNRs<SNR_cut] for data_i in data]
            if len(data_filt) == 1:
                data_filt = data_filt[0]
            return data_filt
        elif get_type=='resolved':
            data_filt = [data_i[SNRs>=SNR_cut] for data_i in data]
            if len(data_filt) == 1:
                data_filt = data_filt[0]
            return data_filt
        else:
            print("Invalid specification of get_type; can be 'resolved' or 'unresolved'.")
            raise
    
    @classmethod
    def gen_summed_spectrum(cls,fs,hs,cos_incs,frange,t_obs,plot=False,saveto=None,return_median=False):
        '''
        Function to calculate the foreground spectrum arising from a set of monochromatic strains and associated frequencies.

        Binaries passed to this function are assumed to all contribute to the foreground.
        
        Arguments:
            fs (1D array of floats) : Binary frequencies. Assumed monochromatic.
            hs (1D array of floats) : Binary strain amplitudes.
            cos_incs (1D array of floats) : Cosine of the binary inclinations
            t_obs (astropy Quantity, time units) : Observation time.
            frange (1D array of floats) : Frequencies at which to calculate binned PSD
        
        Returns:
            fg_PSD_binned (array of floats) : Resulting PSD of unresolved binary background/foreground for all f in frange
        '''
        
        

        
        ## get BLIP frequency bins
        bin_width = frange[1] - frange[0]
        bin_widths = bin_width
        bins = np.append(frange - bin_width/2,frange[-1]+bin_width/2)
        ## get strain squared power
        PSDs_unres = cls.get_binary_psd(hs,cos_incs,bin_width)

        ## check minimum frequency resolution
        ## set minimum bin width to delta_f = 1/T_obs
        ## for now fix to LISA 4yr duration
        min_bin_width = (1/(t_obs)).to(u.Hz)
        if np.any(bin_widths*u.Hz<min_bin_width):
            print("Warning: frequency resolution exceeds the maximum allowed by t_obs.")
        
        ## bin
        fg_hist_binned, edges = np.histogram(fs,bins=bins,weights=PSDs_unres)

        ## np.histogram computes p((1/dfbin)*h^2|f) x N_unres x dfbin
        ## PSD should be p(h^2|f) x N_unres / dfbin
        fg_PSD_binned = fg_hist_binned #/ (bin_width)
    
        ## get running median if needed
        if plot or return_median:
            runmed_binned = medfilt(fg_PSD_binned,kernel_size=3)
    
        ## make plots if desired
        ## note that in BLIP proper, this is called for every segment, and is then ifft'd. 
        ## The true FG spectrum will be the result of splicing these segments together in time domain and taking another fft. Do not expect these to be representative of your expectations of the FG.
        if plot:
            plt.figure()
            det_PSD = lw.psd.lisa_psd(frange*u.Hz,t_obs=4*u.yr,confusion_noise=None,approximate_R=True)
            response_lw = lw.psd.approximate_response_function(frange,fstar=1e-3)
            det_PSD_robson = lw.psd.lisa_psd(frange*u.Hz,t_obs=4*u.yr,confusion_noise='robson19',approximate_R=True)
            plt.plot(frange,det_PSD,color='black',ls='--',label='Detector PSD')
            plt.plot(frange,det_PSD_robson,color='black',label='Detector PSD (R19)')
            plt.plot(frange,response_lw*fg_PSD_binned,color='slategray',alpha=0.5,label='Foreground')
            plt.plot(frange,response_lw*runmed_binned,color='teal',alpha=0.5,label='FG Running Median')
            plt.plot(frange,response_lw*runmed_binned*(1/u.Hz)+det_PSD,color='mediumorchid',alpha=0.5,label='FG + Det. PSD')
            plt.legend(loc='upper right')
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim(1e-43,1e-34)
            # plt.xlim(1e-4,1e-2)
            plt.xlabel('Frquency [Hz]')
            plt.ylabel('GW Power Spectral Density [Hz$^{-1}$]')
            if saveto is None:
                saveto = '.'
            plt.savefig(saveto + '/population_injection.png', dpi=150)
            plt.close()
            ## zoom zoom
            plt.figure()
            plt.plot(frange,det_PSD,color='black',ls='--',label='Detector PSD')
            plt.plot(frange,det_PSD_robson,color='black',label='Detector PSD (R19)')
            plt.plot(frange,response_lw*fg_PSD_binned,color='slategray',alpha=0.5,label='Foreground')
            plt.plot(frange,response_lw*runmed_binned,color='teal',alpha=0.5,label='FG Running Median')
            plt.plot(frange,response_lw*runmed_binned*(1/u.Hz)+det_PSD,color='mediumorchid',alpha=0.5,label='FG + Det. PSD')
            plt.legend(loc='upper right')
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim(1e-40,1e-35)
            plt.xlim(2e-4,4e-3)
            plt.xlabel('Frquency [Hz]')
            plt.ylabel('GW Power Spectral Density [Hz$^{-1}$]')
            plt.savefig(saveto + '/population_injection_zoom.png', dpi=150)
            plt.close()
        
        if return_median:
            spectrum =  fg_PSD_binned *u.Hz*u.s
            median_spectrum = runmed_binned *u.Hz*u.s
            return spectrum, median_spectrum
        else:
            spectrum =  fg_PSD_binned *u.Hz*u.s
            return spectrum
     
    @staticmethod
    def gen_summed_map(lats,longs,PSDs,nside,return_log=False):
        '''
        Function to get a skymap from a collection of binary sky coordinates and (monochromatic) PSDs.
        Note that this function will process all binaries given to it; SNR filtering must be done beforehand.
        
        Arguments:
            lats, longs (arrays of floats) : Latitudes and longitudes of catalogue binaries. 
                                             IMPORTANT: Must be given in ecliptic coordinates and units of degrees!
            PSDs (array of floats) : Corresponding catalogue binary PSDs (assumed monochromatic)
            nside (int) : Healpix nside to use for skymap. Must be power of 2 < 2**32.
            return_log (bool) : If True, also return the log skymap, with slight zero-buffering.
        Returns:
            skymap (array of floats) : Healpix skymap of GW power on the sky
            logskymap (array of floats) : Healpix skymap of log GW power on the sky
        '''
        ## convert sky coordinates to healpy pixels
        pix_idx = hp.ang2pix(nside,longs,lats,lonlat=True)
        ## sum power from all binaries in same pixel
        skymap = np.bincount(pix_idx,weights=PSDs.value,minlength=hp.nside2npix(nside))
        if return_log:
            ## set any zero pixels to a very small number to avoid problems with taking the log
            skymap[skymap<=0] = 1e-80
            ## get log
            logskymap = np.log10(skymap)
            return skymap, logskymap
        else:
            return skymap
    
    @classmethod
    def pop2spec(cls,pop,frange,t_obs,SNR_cut=7,plot=False,return_median=False,saveto=None):
        '''
        Function to calculate the foreground spectrum arising from a population catalogue of unresolved DWD binaries.
        
        Arguments:
            popfile (str)     : '/path/to/binary/population/data/file.csv'
            frange (1D array of floats) : Frequencies at which to calculate binned PSD
            t_obs (astropy Quantity, time units) : Observation time.
            SNR_cut (float) : SNR above which a binary will be assumed to be individually resolveable and subtracted. Default SNR=7.
            return_median (bool) : If True, also return a running median of the spectrum (Useful for smoothing, plotting). Default False.
        Returns:
            fg_PSD (array of floats) : Resulting PSD of unresolved binary background/foreground for all f in frange
        '''
        fs, hs, cos_incs = pop['fs'], pop['hs'], pop['cos_incs']
        ## note, for now we are fixing t_obs=4yr for the purpose of determining which systems are unresolved!!
        snrs = cls.get_snr(fs*u.Hz,hs,cos_incs,(4*u.yr).to(u.s))
        fs_unres, hs_unres, cos_incs_unres = cls.filter_by_snr([fs,hs,cos_incs],snrs,SNR_cut=SNR_cut)
        PSD = cls.gen_summed_spectrum(fs_unres,hs_unres,cos_incs_unres,frange,t_obs,return_median=return_median,plot=plot,saveto=saveto)
        if return_median:
            return PSD[0].value, PSD[1].value
        else:
            return PSD.value
    
    @classmethod
    def pop2map(cls,pop,nside,df,fmin,fmax,SNR_cut=7):
        '''
        Function to get a skymap from a catalogue of binaries.
        
        Arguments:
            lats, longs (arrays of floats) : Latitudes and longitudes of catalogue binaries. 
                                             IMPORTANT: Must be given in ecliptic coordinates and units of degrees!
            PSDs (array of floats) : Corresponding catalogue binary PSDs (assumed monochromatic)
            t_obs (astropy Quantity, time units) : Observation time.
            nside (int) : Healpix nside to use for skymap. Must be power of 2 < 2**32.
            
        Returns:
            skymap (array of floats) : Healpix skymap of GW power on the sky
            logskymap (array of floats) : Healpix skymap of log GW power on the sky
        '''
        fs, hs, lats, longs, cos_incs = pop['fs'], pop['hs'], pop['lats'], pop['longs'], pop['cos_incs']
        ## note, for now we are fixing t_obs=4yr for the purpose of determining which systems are unresolved!!
        snrs = cls.get_snr(fs*u.Hz,hs,cos_incs,(4*u.yr).to(u.s))
        hs_unres, cos_incs_unres, lats_unres, longs_unres = cls.filter_by_snr([hs,cos_incs,lats,longs],snrs,SNR_cut=SNR_cut)
        psds = cls.get_binary_psd(hs_unres,cos_incs_unres,df)
        skymap = cls.gen_summed_map(lats_unres,longs_unres,psds,nside)
        return skymap
    
    @classmethod
    def file2spec(cls,popfile,frange,t_obs,SNR_cut=7,plot=False,return_median=False,**read_csv_kwargs):
        '''
        Wrapper function to calculate the foreground spectrum directly from a population catalogue file.
        
        Arguments:
            popfile (str)     : '/path/to/binary/population/data/file.csv'
            frange (1D array of floats) : Frequencies at which to calculate binned PSD
            t_obs (astropy Quantity, time units) : Observation time.
        Returns:
            fg_PSD (array of floats) : Resulting PSD of unresolved binary background/foreground for all f in frange
        '''
        
        pop = cls.load_population(popfile,frange.min(),frange.max(),**read_csv_kwargs)
        
        return cls.pop2spec(pop,frange,t_obs,SNR_cut=SNR_cut,plot=plot,return_median=return_median)
    
    @classmethod
    def file2map(cls,popfile,nside,df,fmin,fmax,SNR_cut=7,**read_csv_kwargs):
        '''
        Wrapper function to get a skymap directly from a population catalogue file.
        
        Arguments:
            lats, longs (arrays of floats) : Latitudes and longitudes of catalogue binaries. 
                                             IMPORTANT: Must be given in ecliptic coordinates and units of degrees!
            PSDs (array of floats) : Corresponding catalogue binary PSDs (assumed monochromatic)
            t_obs (astropy Quantity, time units) : Observation time.
            nside (int) : Healpix nside to use for skymap. Must be power of 2 < 2**32.
            
        Returns:
            skymap (array of floats) : Healpix skymap of GW power on the sky
            logskymap (array of floats) : Healpix skymap of log GW power on the sky
        '''
        pop = cls.load_population(popfile,fmin,fmax,**read_csv_kwargs)
        
        return cls.pop2map(pop,nside,df,fmin,fmax,SNR_cut=SNR_cut)
        
        
        
        
        
        
##################################################
## Analytic Astrophysical Spatial Distributions ##
##################################################
class Galaxy_Model():
    '''
    Class to support parameterized inference of the Galactic white dwarf binary spatial distribution.
    '''
    def __init__(self,nside,grid_spec='interval',grid_res=0.33,gal_rad=16,gal_height=8,max_rh=4,max_zh=2,fix_rh=None):
        '''
        Function to initialize a grid on which to generate simple parameterized density models of the galactic DWD distribution.
        
        Arguments:
            nside (int)         :   Healpy nside (pixel resolution).
            grid_spec (str)     :   Determines the nature of grid_res (below). Can be 'interval' or 'npoints'. 
                                    If 'interval', grid_res is the dx=dy=dz grid interval in kpc.
                                    If 'npoints', grid_res is the number of number of points along x and y.
                                    (Note that the number of points along z will be scaled to keep dx=dy=dz if gal_rad and gal_height are different.)
            grid_res (float)    :   Grid resolution as defined above. If grid_spec='npoints', type must be int.
            gal_rad (float)     :   Max galactic radius of the grid in kpc. Grid will be definded on -gal_rad <= x,y <= +gal_rad.
            gal_height (float)  :   Max galactic height of the grid in kpc. Grid will be definded on -gal_height <= z <= +gal_height.
            max_rh (float)      :   Maximum prior value of the Galaxy model's radial scale height (rh). Used to create a mask for response function calculations.
            max_zh (float)      :   As max_rh, but for the vertical scale height (zh).
            fix_rh (float)      :   Value of the radial scale height to fix for the model (if None, rh is treated as a parameter.)
            
        '''
        self.nside = nside
        ## for binning
        self.length = hp.nside2npix(self.nside)
        ## create grid *in cartesian coordinates*
        ## size of density grid gives enough padding around the galactic plane without becoming needlessly large
        ## set to 4x max default radial/vertical scale height, respectively (corresponds to "edge" density ~1/10 of central density)
        ## distances in kpc
        if grid_spec=='interval':
            resolution = grid_res
            print("Generating grid with dx = dy = dz = {:0.2f} kpc".format(resolution))
            xs = jnp.arange(-gal_rad,gal_rad,resolution)
            ys = jnp.arange(-gal_rad,gal_rad,resolution)
            zs = jnp.arange(-gal_height,gal_height,resolution)
        elif grid_spec=='npoints':
            if type(grid_res) is not int:
                raise TypeError("If grid_spec is 'npoints', grid_res must be an integer.")
            resolution = gal_rad*2 / grid_res
            print("Generating grid with dx = dy = dz = {:0.2f} kpc".format(resolution))
            xs = jnp.linspace(-gal_rad,gal_rad,grid_res)
            ys = jnp.linspace(-gal_rad,gal_rad,grid_res)
            zs = jnp.arange(-gal_height,gal_height,resolution)
        
        ## generate meshgrid
        x, y, z = jnp.meshgrid(xs,ys,zs)
        self.z = z
        self.r = jnp.sqrt(x**2 + y**2)
        ## Use astropy.coordinates to transform from galactocentric frame to galactic (solar system barycenter) frame.
        gc = cc.SkyCoord(x=x*u.kpc,y=y*u.kpc,z=z*u.kpc, frame='galactocentric')
        SSBc = gc.transform_to(cc.BarycentricMeanEcliptic)
        ## 1/D^2 with filtering to avoid nearby, presumeably resolved, DWDs
        self.dist_adj = (jnp.array(SSBc.distance)>2)*(jnp.array(SSBc.distance))**-2
        ## make pixel grid
        self.pixels = hp.ang2pix(self.nside,np.array(SSBc.lon),np.array(SSBc.lat),lonlat=True).flatten()
        
        ## set global (fixed) MW model parameters
        self.rho_c = 1 # some fiducial central density
        self.r_cut = 2.1 #kpc
        self.r0 = 0.075 #kpc
        self.alpha = 1.8
        self.q = 0.5
        
        ## compute the bulge density (independent of rh,zh)
        rp = jnp.sqrt(self.r**2 + (self.z/self.q)**2)
        self.bulge_density = self.rho_c*(jnp.exp(-(rp/self.r_cut)**2)/(1+rp/self.r0)**self.alpha)
        
        
        if fix_rh is not None:
            self.rh = fix_rh
            self.disk_density_radial_prefactor = self.rho_c*jnp.exp(-self.r/self.rh)
            self.max_skymap = self.mw_mapmaker_2par(fix_rh+0.1,max_zh+0.1)
        else:
            ## create skymap with maximum allowed spatial extent (plus some buffer)
            self.max_skymap = self.mw_mapmaker_2par(max_rh+0.1,max_zh+0.1)
    
    
    def mw_mapmaker_1par(self,zh):
        '''
        
        Generate a galactic white dwarf binary foreground modeled after Breivik et al. (2020), consisting of a bulge + disk. The default values are those given in Breivik+20; the model itself traces back to electromagnetic studies of the Galaxy by McMillan et al. (2011), Bissantz and Gerhard (2002), and Binney et al. (1997).
        zh is the vertical scale height in kpc. 
        The distribution is azimuthally symmetric in the galactocentric frame.
        
        Designed for speed, as it is intended for use during sampling. Relies on pre-computed galaxy grid that is produced as part of Galaxy_Model() initialization.
        
        Returns
        ---------
        skymap : float
            Healpy GW power skymap of the Milky Way white dwarf binary distribution.
        
        '''
        ## Calculate density distribution
        disk_density = self.disk_density_radial_prefactor*jnp.exp(-jnp.abs(self.z)/zh) 
        summed_density = disk_density + self.bulge_density
        ## use stored grid to convert density to power and filter nearby resolved DWDs
        unresolved_powers = summed_density*self.dist_adj
        ## Bin
        skymap = jnp.bincount(self.pixels,weights=unresolved_powers.flatten(),length=self.length)
        
        return skymap

    def mw_mapmaker_2par(self,rh,zh):
        '''
        
        Generate a galactic white dwarf binary foreground modeled after Breivik et al. (2020), consisting of a bulge + disk. The default values are those given in Breivik+20; the model itself traces back to electromagnetic studies of the Galaxy by McMillan et al. (2011), Bissantz and Gerhard (2002), and Binney et al. (1997).
        rh is the radial scale height in kpc, zh is the vertical scale height in kpc. 
        The distribution is azimuthally symmetric in the galactocentric frame.
        
        Designed for speed, as it is intended for use during sampling. Relies on pre-computed galaxy grid that is produced as part of Galaxy_Model() initialization.
        
        Returns
        ---------
        skymap : float
            Healpy GW power skymap of the Milky Way white dwarf binary distribution.
        
        '''
        ## Calculate density distribution
        disk_density = self.rho_c*jnp.exp(-self.r/rh)*jnp.exp(-jnp.abs(self.z)/zh) 
        summed_density = disk_density + self.bulge_density
        ## use stored grid to convert density to power and filter nearby resolved DWDs
        unresolved_powers = summed_density*self.dist_adj
        ## Bin
        skymap = jnp.bincount(self.pixels,weights=unresolved_powers.flatten(),length=self.length)
        
        return skymap

# FIXME there is no log_DWD_FG_map being returned
def generate_galactic_foreground(rh,zh,nside):
    '''
    Generate a galactic white dwarf binary foreground modeled after Breivik et al. (2020), consisting of a bulge + disk. The default values are those given in Breivik+20; the model itself traces back to electromagnetic studies of the Galaxy by McMillan et al. (2011), Bissantz and Gerhard (2002), and Binney et al. (1997).
    rh is the radial scale height in kpc, zh is the vertical scale height in kpc. 
    Thin disk has rh=2.9kpc, zh=0.3kpc; Thick disk has rh=3.31kpc, zh=0.9kpc. Defaults to thin disk. 
    The distribution is azimuthally symmetric in the galactocentric frame.
    
    Arguments
    ----------
    rh (float)  : MW DWD population radial scale height
    zh (float)  : MW DWD population vertical scale height
    nside (int) : healpix nside (skymap resolution)
    
    Returns
    ---------
    astro_map : float
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
    rp = np.sqrt(r**2 + (z/q)**2)
    bulge_density = rho_c*(np.exp(-(rp/r_cut)**2)/(1+rp/r0)**alpha)
    DWD_density = disk_density + bulge_density
    ## Use astropy.coordinates to transform from galactocentric frame to eclipticc (solar system barycenter) frame.
    gc = cc.SkyCoord(x=x*u.kpc,y=y*u.kpc,z=z*u.kpc, frame='galactocentric')    
    SSBc = gc.transform_to(cc.BarycentricMeanEcliptic)
    ## Calculate GW power
    DWD_powers = DWD_density*(np.array(SSBc.distance))**-2
    ## Filter nearby grid points (cut out 2kpc sphere)
    ## This is a temporary soln. Later, we will want to do something more subtle, sampling a DWD pop from
    ## the density distribution and filtering out resolveable SNR>80 binaries
    DWD_unresolved_powers = DWD_powers*(np.array(SSBc.distance) > 2)
    ## Transform to healpix basis
    ## resolution is 2x analysis resolution
#    import pdb; pdb.set_trace()
    pixels = hp.ang2pix(nside,np.array(SSBc.lon),np.array(SSBc.lat),lonlat=True)
    ## Create skymap
    ## Bin
    astro_map = np.bincount(pixels.flatten(),weights=DWD_unresolved_powers.flatten(),minlength=hp.nside2npix(nside))
    
    return astro_map

def generate_galactic_bulge(nside,rho_c=1,r_cut=2.1, r0=0.075,alpha=1.8,q=0.5):
    '''
    Generate the bulge component of a galactic white dwarf binary foreground modeled after the Breivik et al. (2020) bulge+disk spatial model. The default values are those given in Breivik+20; the model itself traces back to electromagnetic studies of the Galaxy by McMillan et al. (2011), Bissantz and Gerhard (2002), and Binney et al. (1997).
    The distribution is azimuthally symmetric in the galactocentric frame.

    Arguments
    ----------
    nside (int) : healpix nside (skymap resolution)

    rho_c (float) : A fiducial central density for the galaxy (the normalization is not important for BLIP simulations, as the skymap will be normalized such that the sky integral is one.) Default 1.
    r_cut (float) : The cutoff radius of the bulge. Default 2.1 kpc.
    r0 (float) : Scale radius of the bulge. Default 0.075 kpc.
    alpha (float) Bulge density power law index. Default 1.8.
    q (float) : Bulge axis ratio. Default 0.5.


    Returns
    ---------
    astro_map : float
        Healpy GW power skymap of the Galactic bulge contribution to the foreground.
    log_DWD_FG_map : float
        Healpy log GW power skymap. For plotting purposes.

    '''
    ## set grid density
    grid_fill = 200
    ## create grid *in cartesian coordinates*
    ## size of density grid gives enough padding around the galactic plane without becoming needlessly large
    ## distances in kpc
    gal_rad = 5
    xs = np.linspace(-gal_rad,gal_rad,grid_fill)
    ys = np.linspace(-gal_rad,gal_rad,grid_fill)
    zs = np.linspace(-5,5,grid_fill)
    ## filter to only bulge range while keeping same 3D grid resolution
    xs = xs[(xs>-5)*(xs<5)]
    ys = ys[(ys>-5)*(ys<5)]
    x, y, z = np.meshgrid(xs,ys,zs)
    r = np.sqrt(x**2 + y**2)
    ## Calculate density distribution
    # disk_density = rho_c*np.exp(-r/rh)*np.exp(-np.abs(z)/zh)
    rp = np.sqrt(r**2 + (z/q)**2)
    bulge_density = rho_c*(np.exp(-(rp/r_cut)**2)/(1+rp/r0)**alpha)
    DWD_density = bulge_density
    ## Use astropy.coordinates to transform from galactocentric frame to eclipticc (solar system barycenter) frame.
    gc = cc.SkyCoord(x=x*u.kpc,y=y*u.kpc,z=z*u.kpc, frame='galactocentric')
    SSBc = gc.transform_to(cc.BarycentricMeanEcliptic)
    ## Calculate GW power
    DWD_powers = DWD_density*(np.array(SSBc.distance))**-2
    ## Filter nearby grid points (cut out 2kpc sphere)
    ## This is a temporary soln. Later, we will want to do something more subtle, sampling a DWD pop from
    ## the density distribution and filtering out resolveable SNR>80 binaries
    DWD_unresolved_powers = DWD_powers*(np.array(SSBc.distance) > 2)
    ## Transform to healpix basis
    pixels = hp.ang2pix(nside,np.array(SSBc.lon),np.array(SSBc.lat),lonlat=True)
    ## Create skymap
    ## Bin
    astro_map = np.bincount(pixels.flatten(),weights=DWD_unresolved_powers.flatten(),minlength=hp.nside2npix(nside))

    return astro_map

def generate_galactic_disk(rh,zh,nside):
    '''
    Generate the disk component of a galactic white dwarf binary foreground modeled after the Breivik et al. (2020) bulge+disk spatial model. The default values are those given in Breivik+20; the model itself traces back to electromagnetic studies of the Galaxy by McMillan et al. (2011), Bissantz and Gerhard (2002), and Binney et al. (1997).
    The distribution is azimuthally symmetric in the galactocentric frame.

    Arguments
    ----------
    rh (float)  : MW DWD population radial disk scale height
    zh (float)  : MW DWD population vertical disk scale height
    nside (int) : healpix nside (skymap resolution)

    Returns
    ---------
    astro_map : float
        Healpy GW power skymap of the Galactic disk contribution to the foreground.
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
    rho_c = 1
    disk_density = rho_c*np.exp(-r/rh)*np.exp(-np.abs(z)/zh)
    DWD_density = disk_density
    ## Use astropy.coordinates to transform from galactocentric frame to eclipticc (solar system barycenter) frame.
    gc = cc.SkyCoord(x=x*u.kpc,y=y*u.kpc,z=z*u.kpc, frame='galactocentric')
    SSBc = gc.transform_to(cc.BarycentricMeanEcliptic)
    ## Calculate GW power
    DWD_powers = DWD_density*(np.array(SSBc.distance))**-2
    ## Filter nearby grid points (cut out 2kpc sphere)
    ## This is a temporary soln. Later, we will want to do something more subtle, sampling a DWD pop from
    ## the density distribution and filtering out resolveable SNR>80 binaries
    DWD_unresolved_powers = DWD_powers*(np.array(SSBc.distance) > 2)
    ## Transform to healpix basis
    pixels = hp.ang2pix(nside,np.array(SSBc.lon),np.array(SSBc.lat),lonlat=True)
    ## Create skymap
    ## Bin
    astro_map = np.bincount(pixels.flatten(),weights=DWD_unresolved_powers.flatten(),minlength=hp.nside2npix(nside))

    return astro_map

def generate_sdg(nside,ra=80.21496, dec=-69.37772, D=50, r=2.1462, N=2169264):
    
    '''
    Generates the stochastic DWD signal from a a generic toy model spherical dwarf galaxy (SDG). Default values are for the LMC.
    
    Arguments
    ---------
    ra, dec : float
        Right ascension and declination.
    D : float
        Distance to SDG in kpc.
    r : float
        radius of SDG in kpc
    N : int
        Number of DWD systems in the SDG
    
    Returns
    ---------
    
    skymap : float
        Healpy GW power skymap of the stochastic DWD signal.
    
    '''
    ## ===== ipynb compute_density function ========================================
    ## all below is only for galaxy model creation
    ## set grid density
    grid_fill = 200

    # sdg radius: (default is the LMC)
    sdg_r = r*u.kpc
    
    # default coordinates give the position of the center of the LMC in ICRS coordinates:
    sdg_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, distance=D*u.kpc)

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
    
    
    ## creating a uniform spherical density, and zero density beyond that
    # rs = distance from any point to the center of the sdg
    rs = np.sqrt((x-x_sdg)**2+(y-y_sdg)**2+(z-z_sdg)**2)
    
    DWD_density = (rs<=sdg_r) * N / (0.524*grid_fill**3) 
    # 0.524 is the filling factor of a sphere in a cube
    # this gives us the number density for points only within the sphere of the sdg, instead of the entire cube

    ## Use astropy.coordinates to transform from galactocentric frame to galactic (solar system barycenter) frame.
    gc = cc.SkyCoord(x=x,y=y,z=z, frame='galactocentric')
    #cc.Galactocentric(x=x,y=y,z=z)
    SSBc = gc.transform_to(cc.Galactic)

    ## Calculate GW power
    ## density will be total power divided by the points that we're simulating
    ## assuming all grid points will contribute an equal amount of power
    DWD_powers = DWD_density*(np.array(SSBc.distance))**-2
    ## Filter nearby grid points (cut out 2kpc sphere)
    ## This is a temporary soln. Later, we will want to do something more subtle, sampling a DWD pop from
    ## the density distribution and filtering out resolveable SNR>80 binaries
    DWD_unresolved_powers = DWD_powers*(np.array(SSBc.distance) > 2)
    ## will need to generate DWD_unresolved_powers for sdg
    

    ## Transform to healpix basis
    ## resolution is 2x analysis resolution
    ## setting resolution, taking coordinates from before and transforming to longlat
    ## replace np.array ... with sdg coordinates
    pixels = hp.ang2pix(nside,np.array(SSBc.l),np.array(SSBc.b),lonlat=True)
    

    ## Create skymap
    ## Bin
    astro_mapG = np.bincount(pixels.flatten(),weights=DWD_unresolved_powers.flatten(),minlength=hp.nside2npix(nside))
    ## below isn't in the jupyter notebook?
    ## Transform into the ecliptic
    rGE = hp.rotator.Rotator(coord=['G','E'])
    astro_map = rGE.rotate_map_pixel(astro_mapG)
    
    ## returning healpix skymaps
    return astro_map

def generate_point_source(ang_coord1,ang_coord2,nside,convention='healpy',pad=True):
    '''
    Generates a point source skymap. 
    
    Arguments
    ---------
    ang_coord1, ang_coord2 : float
        angular coordinates of the point source in radians. Either theta, phi or ra, dec (see convention variable)
    nside : int
        Healpy nside (skymap resolution)
    convention : str
        Angle specification convention. Can be 'healpy' (Healpy polar theta, aziumuthal phi) or 'radec' (standard astronomical RA/DEC). Default is theta/phi.
    pad : bool
        Whether to allow a small amount of power to artifically bleed into adjacent pixels to avoid numerical error issues later on. Only needed for single-pixel case.
    
    Returns
    ---------
    astro_map (array of floats) : healpy skymap
    '''
    
    if convention=='healpy':
        theta, phi = ang_coord1, ang_coord2
    elif convention=='radec':
        ra, dec = ang_coord1, ang_coord2
        theta, phi = np.pi/2 - np.deg2rad(dec), np.deg2rad(ra)
    else:
        raise ValueError("Unknown specification of angular coordinate convention. Can be 'healpy' (Healpy theta/phi) or 'radec' (RA/DEC).")
    
    astro_map = np.zeros(hp.nside2npix(nside))
    ps_id = hp.ang2pix(nside, theta, phi)
    astro_map[ps_id] = 1
    
    if pad:
        neighbours = hp.pixelfunc.get_all_neighbours(nside,ps_id)
        astro_map[neighbours] = 1e-10
        astro_map = astro_map/np.sum(astro_map)
    
    return astro_map

def generate_two_point_source(theta_1,phi_1,theta_2,phi_2,nside):
    '''
    Generates a two-point-source skymap. 
    
    Depreciation note: Keeping until the angular resolution study is finished, then will depreciate in favor of generate_point_sources() (below)/
    
    Arguments
    ---------
    theta_1, phi_1 : float
        angular coordinates of the 1st point source in radians
    theta_2, phi_2 : float
        angular coordinates of the 2nd point source in radians
    
    Returns
    ---------
    astro_map (array of floats) : healpy skymap
    '''
    
    astro_map = np.zeros(hp.nside2npix(nside))
    ps_idx = [hp.ang2pix(nside, theta_1, phi_1),
              hp.ang2pix(nside, theta_2, phi_2)]
    astro_map[ps_idx] = 0.5
    
    return astro_map

def generate_point_sources(coord_list,nside,convention='healpy'):
    '''
    Generates a skymap with a flexible number of point sources. 

    Arguments
    ---------
    coord_list : list of tuples
        List of (ang_coord1,ang_coord2) tuples, one tuple per source. Each tuple gives angular coordinates of their respective  point sourc as either (theta, phi) or (ra, dec) (see convention variable).
    nside : int
        Healpy nside (skymap resolution)
    convention : str
        Angle specification convention. Can be 'healpy' (Healpy polar theta, aziumuthal phi) or 'radec' (standard astronomical RA/DEC). Default is theta/phi.
    
    Returns
    ---------
    astro_map (array of floats) : healpy skymap
    '''
    
    astro_map = np.zeros(hp.nside2npix(nside))
    
    ## add sources
    for source_coord in coord_list:
        source_map = generate_point_source(source_coord[0],source_coord[1],nside,convention=convention,pad=False)
        astro_map += source_map
    
    ## normalise to 1
    astro_map = astro_map/np.sum(astro_map)
    
    return astro_map

def skymap_pix2sph(skymap, blmax):
    '''
    Transform a pixel-basis skymap into the b_lm spherical harmonic basis
    
    Returns
    ---------
    astro_blms : float
        Spherical harmonic healpy expansion of the galactic foreground
    '''
    ## Take square root of powers
    sqrt_map = np.sqrt(skymap)
    ## Generate blms of power (alms of sqrt(power))
    astro_blms = hp.sphtfunc.map2alm(sqrt_map, lmax=blmax)

    # Normalize such that b00 = 1    
    astro_blms = astro_blms/(astro_blms[0])

    return astro_blms
