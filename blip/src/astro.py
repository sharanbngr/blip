import numpy as np
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

    def __init__(self, params, inj, frange):
        '''
        Produces a population object with an attached skymap and spectrum.
        
        Note that we don't carry around the entire set of binaries, just the overall population-level data.
        '''
        self.params = params
        self.inj = inj
        self.frange = frange
        
        
        pop = self.load_population(self.inj['popfile'],self.params['fmin'],self.params['fmax'],names=self.inj['columns'],sep=self.inj['delimiter'])
        
        self.skymap = self.pop2map(pop,self.params['nside'],self.params['dur']*u.s,self.params['fmin'],self.params['fmax'])
        
        ## PSD at injection frequency binning
        self.PSD= self.pop2spec(pop,self.frange,self.params['dur']*u.s,return_median=False,plot=False)

        ## PSD at data frequencies
        fs_spec = np.fft.rfftfreq(int(self.params['fs']*self.params['dur']),1/self.params['fs'])[1:]
        PSD_spec = self.pop2spec(pop,fs_spec,self.params['dur']*u.s,return_median=False,plot=False)
        self.PSD_interp = intrp(fs_spec,PSD_spec)
        self.fftfreqs = np.fft.rfftfreq(int(self.params['fs']*self.params['seglen']),1/self.params['fs'])[1:]
        self.frange_true = self.fftfreqs[np.logical_and(self.fftfreqs >=  self.params['fmin'] , self.fftfreqs <=  self.params['fmax'])]
        self.PSD_true = self.PSD_interp(self.frange_true)

        ## factor of two b/c (h_A,h_A*)~h^2~1/2 * S_A
        ## additional factor of 2 b/c S_GW = 2 * S_A
        self.Sgw = self.PSD * 4
        self.Sgw_true = self.PSD_true * 4
        
        self.sph_skymap = skymap_pix2sph(self.skymap,self.inj['inj_lmax'])
        
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
                        sep=' ',**read_csv_kwargs):
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
        ## filter to frequency band
        f_filter = (fs >= fmin) & (fs <= fmax)
        ## generate pop dict
        pop = {'fs':fs[f_filter],'hs':hs[f_filter],'lats':lats[f_filter],'longs':longs[f_filter]}
        return pop
        
        
    @staticmethod
    def get_binary_psd(hs,t_obs):
        '''
        Function to calculate PSD of catalogue binaries. Assumed monochromatic.
        
        Arguments:
            hs (1D array of floats) : Binary strains.
            t_obs (astropy Quantity, time units) : Observation time.
        
        Returns:
            binary_psds (1D array of floats): Monochromatic PSDs for each binary
        '''
        binary_psds = t_obs*hs**2
        
        return binary_psds
    
    @classmethod
    def get_snr(cls,fs,hs,t_obs,noise_PSD='default'):
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
        SNRs = cls.get_binary_psd(hs,t_obs)/(4*noise_PSD)
        return SNRs
    
    @staticmethod
    def filter_by_snr(data,SNRs,SNR_cut=7,get_type='unresolved'):
        '''
        Function to filter DWD data by SNR. Can return either unresolved (SNR < SNR_cut) or resolved (SNR > SNR_cut) binaries.
        
        Arguments:
            data (1D array of floats) : Binary population data of your choice, corresponding to the given SNRs
            SNRs (1D array of floats) : SNR value for each system corresponding to data
            SNR_cut (float) : Value of SNR that delineates resolved and unresolved binaries. Default is SNR = 7.
            get_type (str) : Whether to return the resolved or unresolved binaries. Default is unresolved.
        
        Returns:
            data_filt : Filtered arrays of frequencies and strains.
        '''
        if get_type=='unresolved':
            return data[SNRs<SNR_cut]
        elif get_type=='resolved':
            return data[SNRs>=SNR_cut]
        else:
            print("Invalid specification of get_type; can be 'resolved' or 'unresolved'.")
            raise
    
    @classmethod
    def gen_summed_spectrum(cls,fs,hs,frange,t_obs,plot=False,saveto=None,return_median=False):
        '''
        Function to calculate the foreground spectrum arising from a set of monochromatic strains and associated frequencies.
        
        Arguments:
            fs (1D array of floats) : Binary frequencies. Assumed monochromatic.
            hs (1D array of floats) : Binary strains.
            t_obs (astropy Quantity, time units) : Observation time.
            frange (1D array of floats) : Frequencies at which to calculate binned PSD
        
        Returns:
            fg_PSD_binned (array of floats) : Resulting PSD of unresolved binary background/foreground for all f in frange
        '''
        
        
        ## get strain squared power
        PSDs_unres = cls.get_binary_psd(hs,4*u.yr)
        
        ## get BLIP frequency bins
        log_frange = np.log10(frange)
        log_bin_width = log_frange[1]-log_frange[0]
        bins = 10**np.append(log_frange-log_bin_width/2,log_frange[-1]+log_bin_width/2)
        bin_widths = bins[1:] - bins[:-1]
        
        ## check minimum frequency resolution
        ## set minimum bin width to delta_f = 1/T_obs
        ## for now fix to LISA 4yr duration
        min_bin_width = (1/(4*u.yr)).to(u.Hz)
        if np.any(bin_widths*u.Hz<min_bin_width):
            print("Warning: frequency resolution exceeds the maximum allowed by t_obs.")
        
        ## bin
        fg_PSD_binned, edges = np.histogram(fs,bins=bins,weights=PSDs_unres)
    
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
            plt.plot(frange,response_lw*fg_PSD_binned/bin_widths,color='slategray',alpha=0.5,label='Foreground')
            plt.plot(frange,response_lw*runmed_binned/bin_widths,color='teal',alpha=0.5,label='FG Running Median')
            plt.plot(frange,response_lw*runmed_binned/bin_widths*(1/u.Hz)+det_PSD,color='mediumorchid',alpha=0.5,label='FG + Det. PSD')
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
            plt.plot(frange,response_lw*fg_PSD_binned/bin_widths,color='slategray',alpha=0.5,label='Foreground')
            plt.plot(frange,response_lw*runmed_binned/bin_widths,color='teal',alpha=0.5,label='FG Running Median')
            plt.plot(frange,response_lw*runmed_binned/bin_widths*(1/u.Hz)+det_PSD,color='mediumorchid',alpha=0.5,label='FG + Det. PSD')
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
            spectrum =  fg_PSD_binned/bin_widths *u.Hz*u.s
            median_spectrum = runmed_binned/bin_widths *u.Hz*u.s
            return spectrum, median_spectrum
        else:
            spectrum =  fg_PSD_binned/bin_widths *u.Hz*u.s
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
        fs, hs = pop['fs'], pop['hs']
        ## note, for now we are fixing t_obs=4yr for the purpose of determining which systems are unresolved!!
        snrs = cls.get_snr(fs*u.Hz,hs,(4*u.yr).to(u.s))
        fs_unres, hs_unres = cls.filter_by_snr(fs,snrs,SNR_cut=SNR_cut), cls.filter_by_snr(hs,snrs,SNR_cut=SNR_cut)
        PSD = cls.gen_summed_spectrum(fs_unres,hs_unres,frange,t_obs,return_median=return_median,plot=plot,saveto=saveto)
        if return_median:
            return PSD[0].value, PSD[1].value
        else:
            return PSD.value
    
    @classmethod
    def pop2map(cls,pop,nside,t_obs,fmin,fmax,SNR_cut=7):
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
        fs, hs, lats, longs = pop['fs'], pop['hs'], pop['lats'], pop['longs']
        snrs = cls.get_snr(fs*u.Hz,hs,t_obs)
        lats_unres, longs_unres = cls.filter_by_snr(lats,snrs,SNR_cut=SNR_cut), cls.filter_by_snr(longs,snrs,SNR_cut=SNR_cut)
        hs_unres = cls.filter_by_snr(hs,snrs,SNR_cut=SNR_cut)
        psds = cls.get_binary_psd(hs_unres,t_obs)
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
        
        return cls.pop2spec(pop,frange,t_obs,SNR_cut=7,plot=plot,return_median=return_median)
    
    @classmethod
    def file2map(cls,popfile,nside,t_obs,fmin,fmax,SNR_cut=7,**read_csv_kwargs):
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
        
        return cls.pop2map(pop,nside,t_obs,fmin,fmax,SNR_cut=7)
        
        
        
        
        
        
##################################################
## Analytic Astrophysical Spatial Distributions ##
##################################################

        
def generate_galactic_foreground(rh,zh,nside):
    '''
    Generate a galactic white dwarf binary foreground modeled after Breivik et al. (2020), consisting of a bulge + disk.
    rh is the radial scale height in kpc, zh is the vertical scale height in kpc. 
    Thin disk has rh=2.9kpc, zh=0.3kpc; Thick disk has rh=3.31kpc, zh=0.9kpc. Defaults to thin disk. 
    The distribution is azimuthally symmetric in the galactocentric frame.
    
    Arguments
    ----------
    rh (float)  : MW DWD population radial scale height
    zh (float)  : MW DWD population vertical scale height
    nside (int) : healpix nside (skymap resolutio)
    
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
    bulge_density = rho_c*(np.exp(-(r/r_cut)**2)/(1+np.sqrt(r**2 + (z/q)**2)/r0)**alpha)
    DWD_density = disk_density + bulge_density
    ## Use astropy.coordinates to transform from galactocentric frame to galactic (solar system barycenter) frame.
    gc = cc.SkyCoord(x=x*u.kpc,y=y*u.kpc,z=z*u.kpc, frame='galactocentric')
    SSBc = gc.transform_to(cc.Galactic)
    ## Calculate GW power
    DWD_powers = DWD_density*(np.array(SSBc.distance))**-2
    ## Filter nearby grid points (cut out 2kpc sphere)
    ## This is a temporary soln. Later, we will want to do something more subtle, sampling a DWD pop from
    ## the density distribution and filtering out resolveable SNR>80 binaries
    DWD_unresolved_powers = DWD_powers*(np.array(SSBc.distance) > 2)
    ## Transform to healpix basis
    ## resolution is 2x analysis resolution
    pixels = hp.ang2pix(nside,np.array(SSBc.l),np.array(SSBc.b),lonlat=True)
    ## Create skymap
    ## Bin
    astro_mapG = np.bincount(pixels.flatten(),weights=DWD_unresolved_powers.flatten(),minlength=hp.nside2npix(nside))

    ## Transform into the ecliptic
    rGE = hp.rotator.Rotator(coord=['G','E'])
    astro_map = rGE.rotate_map_pixel(astro_mapG)
    
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
    
    DWD_density = N / (0.524*200**3)
    # 0.524 is the filling factor of a sphere in a cube
    # this gives us the number density for points only within the sphere of the sdg, instead of the entire cube

    ## creating a sphere_filter 3D array, with 1s in a sphere and 0s otherwise
    # rs = distance from any point to the center of the sdg
    rs = np.sqrt((x-x_sdg)**2+(y-y_sdg)**2+(z-z_sdg)**2)
    
    # set any points within the sdg radius to 1, any points outside to 0
    sphere_filter = np.zeros((grid_fill,grid_fill,grid_fill))
    for i in range(grid_fill):
        for j in range(grid_fill):
            for k in range(grid_fill):
                sphere_filter[i,j,k] = 1 if (rs[i,j,k]<sdg_r) else 0
    ## ** this is probably a computationally expensive way to do this, but it works

    ## =============================================================================
    
    ## ===== ipynb next block ======================================================
    ## Use astropy.coordinates to transform from galactocentric frame to galactic (solar system barycenter) frame.
    gc = cc.Galactocentric(x=x,y=y,z=z)
    SSBc = gc.transform_to(cc.Galactic)
    ## =============================================================================
   
    ## Calculate GW power
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
