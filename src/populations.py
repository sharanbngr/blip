import numpy as np
import legwork as lw
import pandas as pd
from src.instrNoise import instrNoise
from src.geometry import geometry
from src.sph_geometry import sph_geometry
from scipy.interpolate import interp1d as intrp
import matplotlib.pyplot as plt
import healpy as hp
from astropy import units as u
from astropy import coordinates as cc

class populations():

    '''
    Class for handling binary populations. Includes methods for loading, parsing, calculating physical quantities,
    and preparing data for use in some of the makeLISAdata.py signal simulation methods. 
    '''

    def __init__(self, params, inj):
        self.params = params
        self.inj = inj
    
    def load_population(self,popfile,coldict={'f':'f','h':'h','lat':'lat','long':'long'},unitdict={'f':u.Hz,'lat':u.rad,'long':u.rad},
                        sep=' ',**read_csv_kwargs):
        # Would also be good to have an option for giving binary parameters and computing the strain here?
        '''
        Function to load a population file and store relevant data. For now this assumes columns with labels ['f','h',lat','long'].
        Assumes LEGWORK definition of binary strain (see https://arxiv.org/abs/2111.08717)
        
        Arguments:
            popfile (str)     : '/path/to/binary/population/data/file.csv'
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
        
        return fs, hs, lats, longs
        
        
    
    def get_binary_psd(self,hs,t_obs):
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
    
    def get_snr(self,fs,hs,t_obs,noise_PSD='default'):
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
            noise_PSD = lw.psd.lisa_psd(fs,t_obs=t_obs,confusion_noise=None)
        SNRs = self.get_binary_psd(hs,t_obs)/(4*noise_PSD)
        return SNRs
    
    def filter_by_snr(self,data,SNRs,SNR_cut=7,get_type='unresolved'):
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

    # This function should be incorporated in future, but I need to think a little carefully about how to integrate it first.
    def gen_summed_spectrum(self,fs,hs,frange,t_obs):
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
        ## set bin to delta_f = 1/T_obs
        bin_width = (1/t_obs).to(u.Hz)
        ## get unresolved system PSDs
        PSDs_unres = self.get_binary_psd(hs,t_obs)
        ## bin and sum
        edges = np.arange(fs.min(),fs.max()+bin_width.value,bin_width.value)
        fg_PSD, bins = np.histogram(fs,bins=edges,weights=PSDs_unres)
        mids = bins[:-1]+bin_width.value/2
        ## integrate to get total power
        power_before = np.sum(fg_PSD*bin_width)
        new_bin_width = frange[1]-frange[0]
        bins = np.append(frange-new_bin_width/2,frange[-1]+new_bin_width/2)
        fg_PSD_binned, edges = np.histogram(mids,bins=bins,weights=fg_PSD)
        ## normalize to conserve power
        power_after = np.sum(fg_PSD_binned*new_bin_width*u.Hz)
        fg_PSD_binned = (power_before/power_after)*fg_PSD_binned
        ## safety check: conservation of total power just in case things go sideways for some reason
        if power_before != np.sum(fg_PSD_binned*new_bin_width*u.Hz):
            print("Warning: Power is not being conserved in the spectrum rebinning process.")
            diff = (power_before - np.sum(fg_PSD_binned*new_bin_width*u.Hz))
            frac_diff = (diff/power_before).value
            print("Difference (pre-binning - post-binning) is {} (fractional difference of {:e})".format(diff,frac_diff))
        
        return fg_PSD_binned
    
    def gen_summed_map(self,lats,longs,PSDs,nside):
        '''
        Function to get a skymap from a collection of binary sky coordinates and (monochromatic) PSDs.
        Note that this function will process all binaries given to it; SNR filtering must be done beforehand.
        
        Arguments:
            lats, longs (arrays of floats) : Latitudes and longitudes of catalogue binaries. 
                                             IMPORTANT: Must be given in ecliptic coordinates and units of degrees!
            PSDs (array of floats) : Corresponding catalogue binary PSDs (assumed monochromatic)
            nside (int) : Healpix nside to use for skymap. Must be power of 2 < 2**32.
            
        Returns:
            skymap (array of floats) : Healpix skymap of GW power on the sky
            logskymap (array of floats) : Healpix skymap of log GW power on the sky
        '''
        ## convert sky coordinates to healpy pixels
        pix_idx = hp.ang2pix(nside,longs,lats,lonlat=True)
        ## sum power from all binaries in same pixel
        skymap = np.bincount(pix_idx,weights=PSDs.value,minlength=hp.nside2npix(nside))
        ## set any zero pixels to a very small number to avoid problems with taking the log
        skymap[skymap<=0] = 1e-80
        ## get log
        logskymap = np.log10(skymap)
        return skymap, logskymap
    
    def pop2spec(self,popfile,frange,t_obs,SNR_cut=7,**read_csv_kwargs):
        '''
        Function to calculate the foreground spectrum arising from a population catalogue of unresolved DWD binaries.
        
        Arguments:
            popfile (str)     : '/path/to/binary/population/data/file.csv'
            frange (1D array of floats) : Frequencies at which to calculate binned PSD
            t_obs (astropy Quantity, time units) : Observation time.
        Returns:
            fg_PSD (array of floats) : Resulting PSD of unresolved binary background/foreground for all f in frange
        '''
        fs, hs, lats, longs = self.load_population(popfile,**read_csv_kwargs)
        snrs = self.get_snr(fs*u.Hz,hs,t_obs)
        fs_unres, hs_unres = self.filter_by_snr(fs,snrs,SNR_cut=SNR_cut), self.filter_by_snr(hs,snrs,SNR_cut=SNR_cut)
        fg_PSD = self.gen_summed_spectrum(fs_unres,hs_unres,frange,t_obs).value
        return fg_PSD
    
    def pop2map(self,popfile,nside,t_obs,SNR_cut=7,**read_csv_kwargs):
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
        fs, hs, lats, longs = self.load_population(popfile,**read_csv_kwargs)
        snrs = self.get_snr(fs*u.Hz,hs,t_obs)
        lats_unres, longs_unres = self.filter_by_snr(lats,snrs,SNR_cut=SNR_cut), self.filter_by_snr(longs,snrs,SNR_cut=SNR_cut)
        hs_unres = self.filter_by_snr(hs,snrs,SNR_cut=SNR_cut)
        psds = self.get_binary_psd(hs_unres,t_obs)
        skymap, logskymap = self.gen_summed_map(lats_unres,longs_unres,psds,nside)
        return skymap, logskymap
        
        
        