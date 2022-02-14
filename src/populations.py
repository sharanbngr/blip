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
        fs = dwds[coldict['f']].to_numpy()*unitdict['f'].to(u.Hz).value
        hs = dwds[coldict['h']].to_numpy()
        lats = dwds[coldict['lat']].to_numpy()*unitdict['lat'].to(u.deg).value
        longs = dwds[coldict['long']].to_numpy()*unitdict['long'].to(u.deg).value
        
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
    
    def filter_by_snr(self,fs,hs,SNRs,SNR_cut=7,get_type='unresolved'):
        '''
        Function to filter DWD data by SNR. Can return either unresolved (SNR < SNR_cut) or resolved (SNR > SNR_cut) binaries.
        
        Arguments:
            fs (1D array of floats) : Binary frequencies. Assumed monochromatic.
            hs (1D array of floats) : Binary strains.
            SNRs (1D array of floats) : Binary SNRs.
            SNR_cut (float) : Value of SNR that delineates resolved and unresolved binaries. Default is SNR = 7.
            get_type (str) : Whether to return the resolved or unresolved binaries. Default is unresolved.
        
        Returns:
            fs_filt, hs_filt : Filtered arrays of frequencies and strains.
        '''
        if get_type=='unresolved':
            return fs[SNRs<SNR_cut],hs[SNRs<SNR_cut]
        elif get_type=='resolved':
            return fs[SNRs>=SNR_cut],hs[SNRs>=SNR_cut]
        else:
            print("Invalid specification of get_type; can be 'resolved' or 'unresolved'.")
            raise

    # This function should be incorporated in future, but I need to think a little carefully about how to integrate it first.
#    def pop2spec(fs,hs,t_obs,fmin=5e-5,fmax=1e-1,output='foreground'):
#        '''
#        Function to calculate the foreground spectrum arising from a population catalogue of unresolved DWD binaries.
#        
#        Arguments:
#            fs (1D array of floats) : Binary frequencies. Assumed monochromatic.
#            hs (1D array of floats) : Binary strains.
#            t_obs (astropy Quantity, time units) : Observation time.
#            plot (bool) : Whether to also produce a plot of the foreground spectrum. Default True.
#            saveto (str) : If specified, the location to which the plot will be saved. Default None (won't save).
#            fmin, fmax (float) : Minimum and maximum frequencies of the calculated and plotted PSD.
#            output (str) : Can be 'foreground' or 'all'. Which sets of PSDs to return.
#                           If 'foreground', only the foreground PSD and frequencies will be returned.
#                           If 'all', total PSD and detector PSD will also be returned.
#        
#        Returns:
#            (if output=='foreground'):
#            fg_PSD_fullspec (array of floats) : Resulting PSD of unresolved binary background/foreground
#            fs_all (array of floats) : Frequencies at which PSD is evaluated.
#            (additionally, if output=='all'):
#            det_PSD (array of floats) : Detector PSD evaluated at fs_all
#            PSD_tot (array of floats) : Total PSD (detector noise + foreground)
#        '''
#        ## set bin to delta_f = 1/T_obs
#        bin_width = (1/t_obs).to(u.Hz)
#        ## get unresolved system PSDs
#        PSDs_unres = t_obs*hs**2
#        ## bin and sum
#        edges = np.arange(fs_unres.min().value,fs_unres.max().value+bin_width.value,bin_width.value)
#        fg_PSD, bins = np.histogram(fs_unres.value,bins=edges,weights=PSDs_unres)
#        mids = bins[:-1]+bin_width.value
#        ## get total PSD
#        fs_low = np.arange(fmin,mids[0]-bin_width.value,bin_width.value)
#        fs_high = np.arange(mids[-1]+bin_width.value,fmax,bin_width.value)
#        fs_all = np.append(np.append(fs_low,mids),fs_high)
#        PSD_tot = np.append(np.append(lw.psd.lisa_psd(fs_low*u.Hz,t_obs=t_obs,confusion_noise=None),
#                                      lw.psd.lisa_psd(mids*u.Hz,t_obs=t_obs,confusion_noise=None)+fg_PSD),
#                            lw.psd.lisa_psd(fs_high*u.Hz,t_obs=t_obs,confusion_noise=None))
#        ## extend foreground PSD to match with fs_all (filling in zeros where there is no foreground contribution)
#        fg_PSD_fullspec = np.append(np.append(np.zeros(len(fs_low)),fg_PSD.value),np.zeros(len(fs_high)))/u.Hz
#        ## get detector PSD
#        det_PSD = lw.psd.lisa_psd(fs_all*u.Hz,t_obs=t_obs,confusion_noise=None)
#        
#        if output=='foreground':
#            return fg_PSD_fullspec, fs_all
#        elif output=='all':
#            return fg_PSD_fullspec, fs_all, det_PSD, PSD_tot
#        else:
#            print("Invalid specification of output. Can be 'foreground' or 'all'.")
    
    def skymap_from_pop(self,lats,longs,PSDs,nside):
        '''
        Function to get a skymap from a catalogue of binaries. 
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
    
    
