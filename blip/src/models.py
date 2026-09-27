import os
import shutil
import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import healpy as hp
import logging
import time
from blip.src.submodel import submodel
from blip.src.utils import log_manager, gen_suffixes, catch_color_duplicates
from blip.src.fast_geometry import fast_geometry
from blip.src.faster_geometry import calculate_response_functions

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
jax.config.update("jax_enable_x64", True)


###################################################
###      UNIFIED MODEL PRIOR & LIKELIHOOD       ###
###################################################

@register_pytree_node_class
class Model():
    """
    Analysis model. This is a container of :class:`submodels
    <blip.src.submodel.submodel>` with a prior and a likelihood that can be sampled.

    Parameters
    ----------
    params, inj : dict
        Configuration dictionaries from :func:`parse_config <blip.config.parse_config>`.
    fs : array (nfreqs,)
        Frequency grid in Hz.
    f0 : array (nfreqs,)
        Frequency grid scaled by transfer frequency (f0 = fs/2*FSTAR).
    tsegmid : array (ntimes,)
        Time grid in seconds.
    rmat : complex array (nfreqs, ntimes, 3, 3)
        Data correlation matrix.
    """

    # TODO deduce (fs, f0, tsegmid) from (params, inj)
    def __init__(self,params,inj,fs,f0,tsegmid,rmat):
        self.fs = fs
        self.f0 = f0
        self.tsegmid = tsegmid
        self.params = params
        self.inj = inj

        base_component_names = [spec.raw_name for spec in params["model"]]
        self.submodel_names = base_component_names
        suffixes = gen_suffixes(base_component_names)
        
        self.submodels = {}
        self.Npar = 0
        self.parameters = {}
        all_parameters = []
        spectral_parameters = []
        spatial_parameters = []
        self.blm_phase_idx = []

        for submodel_spec, suffix in zip(params["model"], suffixes):
            submodel_name = submodel_spec.raw_name
            sm = submodel(params,inj,submodel_spec,fs,f0,tsegmid,suffix=suffix)
            self.submodels[submodel_name] = sm

            if hasattr(sm,"blm_phase_idx"):
                for ii in sm.blm_phase_idx:
                    self.blm_phase_idx.append(self.Npar+sm.blm_start+ii)
#            if sm.Npar==0:
#                sm.fixed_cov = ... ## add handling for 0-parameter, non-noise models here (both spatial and spectral models fixed)
            self.Npar += sm.Npar
            self.parameters[submodel_name] = sm.parameters
            spectral_parameters += sm.spectral_parameters
            spatial_parameters += sm.spatial_parameters
            all_parameters += sm.parameters
        
        self.parameters['spectral'] = spectral_parameters
        self.parameters['spatial'] = spatial_parameters
        self.parameters['all'] = all_parameters
        
        ## Having initialized all the components, now compute the LISA response functions
#        t1 = time.time()
#        fast_rx = fast_geometry(self.params)
#        fast_rx.calculate_response_functions(self.f0,self.tsegmid,[self.submodels[smn] for smn in self.submodel_names if smn !='noise'],self.params['tdi_lev'])
#        t2 = time.time()
#        print("Time elapsed for calculating the LISA response functions for all submodels via joint computation is {} s.".format(t2-t1))
#        ## deallocate to save on memory now that the response functions have been calculated and stored elsewhere
#        del fast_rx
        
        ## update colors as needed
        catch_color_duplicates(self)
        
        ## assign reference to data for use in likelihood
        self.rmat = rmat
        
        return
    
    
#    @jax.jit
    def prior(self,unit_theta):
        '''
        Unified prior function to interatively perform prior draws for each submodel in the proper order
        
        Arguments
        ----------------
        unit_theta (array) : draws from the unit cube
        
        Returns
        ----------------
        theta (list) : transformed prior draws for all submodels in sequence
        '''
        theta = []
        start_idx = 0
        
        for sm_name in self.submodel_names:
            sm = self.submodels[sm_name]
            theta += sm.prior(unit_theta[start_idx:(start_idx+sm.Npar)])
            start_idx += sm.Npar
        
        if len(theta) != len(unit_theta):
            raise ValueError("Input theta does not have same length as output theta, something has gone wrong!")
        
        return theta
    
#    @jax.jit
    def likelihood(self,theta):
        '''
        Unified likelihood function to compare the combined covariance contributions of a generic set of noise/SGWB models to the data.
        
        Arguments
        ----------------
        theta (list) : transformed prior draws for all submodels in sequence
        
        Returns
        ----------------
        loglike (float) : resulting joint log likelihood
        '''
        start_idx = 0
        for i, sm_name in enumerate(self.submodel_names):
            sm = self.submodels[sm_name]
            if sm.Npar == 0:
                theta_i = None
            else:
                theta_i = theta[start_idx:(start_idx+sm.Npar)]
                start_idx += sm.Npar
            if i==0:
                cov_mat = sm.cov(theta_i)
            else:
                cov_mat = cov_mat + sm.cov(theta_i)


        ## change axis order to make taking an inverse easier
        cov_mat = jnp.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -jnp.einsum('ijkl,ijkl', inv_cov, self.rmat) - jnp.einsum('ij->', jnp.log(jnp.pi * self.params['seglen'] * jnp.abs(det_cov)))


        loglike = jnp.real(logL)

        return loglike
    
    ## this allows for jax/numpyro to properly perform jitting of the class
    ## all attributes of the model class should be static
    ## may need to tweak this if/when we implement any kind of RJMCMC approach
    def tree_flatten(self):
        children = []  # arrays / dynamic values
        aux_data = {'params':self.params,'inj':self.inj,'fs':self.fs,'f0':self.f0,'tsegmid':self.tsegmid,'rmat':self.rmat} # static values
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    

###################################################
###       UNIFIED INJECTION INFRASTRUCTURE      ###
################################################### 

    
class Injection():
    """
    Simulation model. This is a container of :class:`submodels
    <blip.src.submodel.submodel>` used for synthesizing LISA data.

    Parameters
    ----------
    params, inj : dict
        Configuration dictionaries from :func:`parse_config <blip.config.parse_config>`.
    fs : array (nfreqs,)
        Frequency grid in Hz.
    f0 : array (nfreqs,)
        Frequency grid scaled by transfer frequency (f0 = fs/2*FSTAR).
    tsegmid : array (ntimes,)
        Time grid in seconds.
    """

    # TODO deduce (fs, f0, tsegmid) from (params, inj)
    def __init__(self,params,inj,fs,f0,tsegmid):
        self.params = params
        self.inj = inj
        
        self.frange = fs
        self.f0 = f0
        self.tsegmid = tsegmid
        
        ## separate into components
        self.component_specs = inj['injection']
        self.component_names = [spec.raw_name for spec in self.component_specs]
        N_inj = len(self.component_specs)
        
        ### commenting this out because we're switching to active specification of duplicates in the params file
        ## check for and differentiate duplicate injections
        ## this will append 1 (then 2, then 3, etc.) to any duplicate component names
        ## we will also generate appropriate variable suffixes to use in plots, etc..
#        self.component_names = catch_duplicates(base_component_names)
        
        ## it's useful to have a version of this without the detector noise
        self.sgwb_component_names = [name for name in self.component_names if name!='noise']
        suffixes = gen_suffixes(self.component_names)
                        
        ## initialize components
        self.components = {}
        self.truevals = {}
        

        ## step through and build components
        ## parallelization has been depreciated now that the response function calculations are handled elsewhere
        for i, (component_spec, suffix) in enumerate(zip(self.component_specs,suffixes)):
            print("Building injection for {} (component {} of {})...".format(component_spec.raw_name,i+1,N_inj))
            cm = submodel(params,inj,component_spec,fs,f0,tsegmid,injection=True,suffix=suffix)
            self.components[component_spec.raw_name] = cm
            self.truevals[component_spec.raw_name] = cm.truevals
    
            if cm.has_map:
                self.plot_skymaps(component_spec.raw_name)
        
        ## Having initialized all the components, now compute the LISA response functions
        if self.inj['parallel_inj'] and self.inj['response_nthread']>1:
            rx_nthreads = self.inj['response_nthread']
        else:
            rx_nthreads = 1
        t1 = time.time()
        submodels_sgwb = [self.components[cmn] for cmn in self.sgwb_component_names]
        if params["faster_geometry"]:
            calculate_response_functions(fs, self.tsegmid, submodels_sgwb, params)
        else:
            fast_rx = fast_geometry(self.params,nthreads=rx_nthreads)
            fast_rx.calculate_response_functions(self.f0,self.tsegmid,submodels_sgwb,self.params['tdi_lev'])
        t2 = time.time()
        print("Time elapsed for calculating the LISA response functions for all components via joint computation is {} s.".format(t2-t1))
        
        ## initialize default plotting lower ylim
        self.plot_ylim = None
        
        ## update colors as needed
        catch_color_duplicates(self)
    
    
    def add_component(self,name_args):
        '''
        Wrapper function for the injection component creation process, to allow for parallelization.
        
        Arguments
        ------------------
        name_args (tuple)    : (component_name,suffix) for one component
        
        Returns
        ------------------
        cm (submodel object) : Injection component
        '''
        
        component_name, suffix = name_args
        cm = submodel(self.params,self.inj,component_name,self.frange,self.f0,self.tsegmid,injection=True,suffix=suffix)
        print("Injection component build complete for component: {}".format(component_name))
        
        return cm
        
#    def compute_convolved_spectra(self,component_name,fs_new=None,channels='11',return_fs=False,imaginary=False):
#        '''
#        Wrapper to return the frozen injected detector-convolved GW spectra for the desired channels.
#        
#        Useful note - these frozen spectra are computed in diag_spectra(), as they are calculated and saved at the analysis frequencies.
#        
#        Also note that this is meant for plotting purposes only, and includes interpolation/absolute values that are not desirable in a data generation/analysis environment.
#        
#        Arguments
#        -----------
#        component_name (str) : the name (key) of the Injection component to use.
#        fs_new (array) : If desired, frequencies at which to interpolate the convolved PSD
#        channels (str) : Which channel cross/auto-correlation PSD to plot. Default is '11' auto-correlation, i.e. XX for XYZ, 11 for Michelson, AA for AET.
#        return_fs (bool) : If True, also returns the frequencies at which the PSD has been evaluated. Default False.
#        imaginary (bool) : If True, returns the magnitude of the imaginary component. Default False.
#        
#        Returns
#        -----------
#        PSD (array) : Power spectral density of the specified channels' auto/cross-correlation at the desired frequencies.
#        fs (array, optional) : The PSD frequencies, if return_fs==True.
#        
#        '''
#        
#        cm = self.components[component_name]
#        ## split the channel indicators
#        c1_idx, c2_idx = int(channels[0]) - 1, int(channels[1]) - 1
#        
#        if not imaginary:
#            PSD = np.abs(np.real(cm.frozen_convolved_spectra[c1_idx,c2_idx,:]))
#        else:
#            PSD = 1j * np.abs(np.imag(cm.frozen_convolved_spectra[c1_idx,c2_idx,:]))
#        
#        ## populations need some finessing due to frequency subtleties                
#        if hasattr(cm,"ispop") and cm.ispop:
#            fs = cm.population.frange_true
#            if (fs_new is not None) and not np.array_equal(fs_new,cm.population.frange_true):
#                with log_manager(logging.ERROR):
#                    PSD_interp = interp1d(fs,PSD,bounds_error=False,fill_value=0)
#                    PSD = PSD_interp(fs_new)
#                    fs = fs_new
#        else:
#            fs = self.frange
#            ## there is no way to compute the convolved injected spectra once the injected response functions have been flushed
#            ## we have saved them, however, and can either just use the saved frozen spectra or interpolate to a new frequency grid
#            ## WARNING: interpolation will likely result in low fidelity at f < 3e-4 Hz.
#            if fs_new is not None:
#                with log_manager(logging.ERROR):
#                    PSD_interp = interp1d(fs,np.log10(PSD))
#                    PSD = 10**PSD_interp(fs_new)
#                    fs = fs_new
#
#        if return_fs:
#            return fs, PSD
#        else:
#            return PSD
    
    def compute_convolved_spectra(self,component_name,fs_new=None,channels='11',return_fs=False,imaginary=False):
        '''
        Wrapper to return the frozen injected detector-convolved GW spectra for the desired channels.
        
        Useful note - these frozen spectra are computed in diag_spectra(), as they are calculated and saved at the analysis frequencies.
        
        Also note that this is meant for plotting purposes only, and includes interpolation/absolute values that are not desirable in a data generation/analysis environment.
        
        Arguments
        -----------
        component_name (str) : the name (key) of the Injection component to use.
        fs_new (array) : If desired, frequencies at which to interpolate the convolved PSD
        channels (str) : Which channel cross/auto-correlation PSD to plot. Default is '11' auto-correlation, i.e. XX for XYZ, 11 for Michelson, AA for AET.
        return_fs (bool) : If True, also returns the frequencies at which the PSD has been evaluated. Default False.
        imaginary (bool) : If True, returns the magnitude of the imaginary component. Default False.
        
        Returns
        -----------
        PSD (array) : Power spectral density of the specified channels' auto/cross-correlation at the desired frequencies.
        fs (array, optional) : The PSD frequencies, if return_fs==True.
        
        '''
        
        cm = self.components[component_name]
        ## split the channel indicators
        c1_idx, c2_idx = int(channels[0]) - 1, int(channels[1]) - 1
            
        ## simulated data frequencies
        if fs_new == 'data':
            fs = cm.fdata
            PSD_complex = cm.fdata_convolved_spectra[c1_idx,c2_idx,:]
        ## all other cases start from the original injected frequencies
        else:
            fs = self.frange
            PSD_complex = cm.frozen_convolved_spectra[c1_idx,c2_idx,:]
        
        ## handle complex spectra as desired
        if not imaginary:
            PSD = np.abs(np.real(PSD_complex))
        else:
            PSD = 1j * np.abs(np.imag(PSD_complex))
        
        ## estimate spectra at new frequencies -- WARNING: requires interpolation, usually produces low-fidelity results
        ## only really useful for quick checks and visualization, NOT for analysis purposes!
        if fs_new is not None and fs_new != 'data':
        ## populations need some finessing due to frequency subtleties                
            if hasattr(cm,"ispop") and cm.ispop:
                fs = cm.population.frange_true
                if (fs_new is not None) and not np.array_equal(fs_new,cm.population.frange_true):
                    with log_manager(logging.ERROR):
                        PSD_interp = interp1d(fs,PSD,bounds_error=False,fill_value=0)
                        PSD = PSD_interp(fs_new)
                        fs = fs_new
            else:
                ## there is no way to compute the convolved injected spectra once the injected response functions have been flushed
                ## we have saved them, however, and can either just use the saved frozen spectra or interpolate to a new frequency grid
                ## WARNING: interpolation will likely result in low fidelity at f < 3e-4 Hz.
                with log_manager(logging.ERROR):
                    PSD_interp = interp1d(fs,np.log10(PSD))
                    PSD = 10**PSD_interp(fs_new)
                    fs = fs_new

        if return_fs:
            return fs, PSD
        else:
            return PSD    
    
    def plot_injected_spectra(self,component_name,fs_new=None,ax=None,convolved=False,legend=False,channels='11',return_PSD=False,scale='log',flim=None,ymins=None,**plt_kwargs):
        '''
        Wrapper to plot the injected spectrum component on the specified matplotlib axes (or current axes if unspecified).
        
        Arguments
        -----------
        component_name (str) : the name (key) of the Injection component to use.
        fs_new (array) : If desired, frequencies at which to interpolate the convolved PSD
        ax (matplotlib axes) : Axis on which to plot. Default None (will plot on current axes.)
        convolved (bool) : If True, convolve the injected spectra with the detector response. Default False.
        legend (bool) : If True, generate a legend entry. Default False.
        channels (str) : Which channel cross/auto-correlation PSD to plot. Default is '11' auto-correlation, i.e. XX for XYZ, 11 for Michelson, AA for AET.
        return_PSD (bool) : If True, also returns the plotted PSD. Default False.
        scale (str) : Matplotlib scale at which to plot ('log' or 'linear'). Default 'log'.
        flim (tuple) : (fmin,fmax) plot limits. Default None (will use fmin,fmax as specified in the params file.)
        ymins (list) : External list to which, if specified, will be added the lower ylim of the injected spectra.
        **plt_kwargs (kwargs) : matplotlib.pyplot keyword arguments
        
        Returns
        -----------
        PSD plot on specified axes.
        PSD (array, optional) : Power spectral density of the specified channels' auto/cross-correlation at the desired frequencies.

        '''
        ## grab component
        cm = self.components[component_name]
        
        ## set axes
        if ax is None:
            ax = plt.gca()
        
        ## set fmin/max to specified values, or default to the ones in params
        if flim is not None:
            fmin = flim[0]
            fmax = flim[1]
        else:
            fmin = self.params['fmin']
            fmax = self.params['fmax']
        
        ## special treatment of population frequencies
#        if hasattr(self.components[component_name],"ispop") and self.components[component_name].ispop:
#            fs_base = self.components[component_name].population.frange_true
#        else:
#        fs_base = self.frange
        
        ## get frozen injected spectra at original injection frequencies and convolve with detector response if desired
        if convolved:
            if component_name == 'noise':
                raise ValueError("Cannot convolve noise spectra with the detector GW response - this is not physical. (Set convolved=False in the function call!)")
            fs, PSD = self.compute_convolved_spectra(component_name,channels=channels,return_fs=True,fs_new=fs_new)
        else:
            ## handle wanting to plot at new frequencies (typically the data frequencies)
            ## original injection frequencies
            if fs_new is None:
                if hasattr(cm,"ispop") and cm.ispop:
                    PSD = cm.population.Sgw_true
                    fs = cm.population.frange_true
                else:
                    fs = self.frange
                    PSD = cm.frozen_spectra
            ## data frequencies (self.fdata in the code)
            elif (type(fs_new) is str) and (fs_new == 'data'):
                fs = cm.fdata
                PSD = cm.fdata_spectra
            ## estimate spectra at new frequencies -- WARNING: requires interpolation, usually produces low-fidelity results
            ## only really useful for quick checks and visualization, NOT for analysis purposes!
            else:
                if component_name == 'noise':
                    fstar = 3e8/(2*np.pi*cm.armlength)
                    f0_new = fs_new/(2*fstar)
                    PSD = cm.instr_noise_spectrum(fs_new,f0_new,Np=10**cm.injvals['log_Np'],Na=10**cm.injvals['log_Na'])
                ## special treatment for the population case
                elif hasattr(cm,"ispop") and cm.ispop:
                    PSD = cm.population.Sgw_true
                    fs = cm.population.frange_true
                    if not np.array_equal(fs_new,cm.population.frange_true):
                        ## the interpolator gets grumpy sometimes, but it's not an actual issue hence the logging wrapper
                        with log_manager(logging.ERROR):
                            PSD_interp = interp1d(fs,PSD,bounds_error=False,fill_value=0)
                            PSD = PSD_interp(fs_new)
                            fs = fs_new
                else:
                    Sgw_args = [cm.truevals[parameter] for parameter in cm.spectral_parameters]
                    PSD = cm.compute_Sgw(fs_new,Sgw_args)
                
                fs = fs_new
                
        ## noise will return the 3x3 covariance matrix, need to grab the desired channel cross-/auto-power
        ## generically capture anything that looks like a covariance matrix for future-proofing
        if (len(PSD.shape)==3) and (PSD.shape[0]==PSD.shape[1]==3):
            I, J = int(channels[0]) - 1, int(channels[1]) - 1
            PSD = PSD[I,J,:]
        
        filt = (fs>=fmin)*(fs<=fmax)

        if legend:
            label = cm.fancyname
            if plt_kwargs is None:
                plt_kwargs = {}
                plt_kwargs['label'] = label
            else:
                if 'label' not in plt_kwargs.keys():
                    plt_kwargs['label'] = label
        
        if scale=='log':
            ax.loglog(fs[filt],PSD[filt],**plt_kwargs)
        elif scale=='linear':
            ax.plot(fs[filt],PSD[filt],**plt_kwargs)
        else:
            raise ValueError("We only support linear and log plots, there is no secret third option!")
        
        if ymins is not None:
            ymins.append(PSD.min())
        
        if return_PSD:
            return PSD
        else:
            return
        
    def plot_skymaps(self,component_name,save_figures=True,return_mapdata=False,**plt_kwargs):
        '''
        Function to plot the injected skymaps.
        
        NOTE - will need to be generalized when I add the astro injections
        '''
        cm = self.components[component_name]
        
        # deals with projection parameter 
        if self.params['projection'] is None:
            coord = 'E'
        elif self.params['projection']=='G' or self.params['projection']=='C':
            coord = ['E',self.params['projection']]
        elif self.params['projection']=='E':
            coord = self.params['projection']
        else:  
            raise TypeError('Invalid specification of projection, projection can be E, G, or C')
        
        if return_mapdata:
            cm_data = {}
        
        
        ## dimensionless energy density at 1 mHz
        spec_args = [cm.truevals[parameter] for parameter in cm.spectral_parameters]
        Omega_1mHz = cm.omegaf(1e-3,*spec_args)
        
        if hasattr(cm,"skymap"):
            Omegamap_pix = Omega_1mHz * cm.skymap/(np.sum(cm.skymap)*hp.nside2pixarea(self.params['nside'])/(4*np.pi))
            
            ## tell healpy to shush
            with log_manager(logging.ERROR):
                hp.mollview(Omegamap_pix, coord=coord, title=r'Injected pixel map $\Omega (f = 1 mHz)$', unit=r"$\Omega(f= 1mHz)$", cmap=self.params['colormap'])
                hp.graticule()
            
            if save_figures:
                np.savetxt(self.params['out_dir']+'/inj_pixelmap_data.txt',Omegamap_pix)
                plt.savefig(self.params['out_dir'] + '/inj_pixelmap'+component_name+'.png', dpi=150)
                print('Saving injection pixel map at ' +  self.params['out_dir'] + '/inj_pixelmap'+component_name+'.png')
                plt.close()
            
            if return_mapdata:
                cm_data['Omega_pixelmap'] = Omegamap_pix
                cm_data['normed_pixelmap'] = cm.skymap/(np.sum(cm.skymap)*hp.nside2pixarea(self.params['nside'])/(4*np.pi))
                
        if hasattr(cm,"sph_skymap"):
            ## sph map
            Omegamap_inj = Omega_1mHz * cm.sph_skymap
            ## tell healpy to shush
            with log_manager(logging.ERROR):
                hp.mollview(Omegamap_inj, coord=coord, title=r'Injected angular distribution map $\Omega (f = 1 mHz)$', unit=r"$\Omega(f= 1mHz)$", cmap=self.params['colormap'])
                hp.graticule()
            
            if save_figures:
                plt.savefig(self.params['out_dir'] + '/inj_skymap'+component_name+'.png', dpi=150)
                print('Saving injected sph skymap at ' +  self.params['out_dir'] + '/inj_skymap'+component_name+'.png')
                plt.close()
            
            if return_mapdata:
                cm_data['Omega_sphmap'] = Omegamap_inj
                cm_data['normed_sphmap'] = cm.sph_skymap
            
        if return_mapdata:
            return cm_data
        else:
            return
        
    def extract_and_save_skymap_data(self,map_data_path=None):
        ## load or create plot_data dict
        if map_data_path is None:
            map_data_path = self.params['out_dir']+'/plot_data.pickle'
        if os.path.exists(map_data_path):
            with open(map_data_path, 'rb') as datafile:
                plot_data = pickle.load(datafile)
            if 'map_data' not in plot_data.keys():
                plot_data['map_data'] = {}
            
        else:
            plot_data = {'map_data':{}}
        
        plot_data['map_data']['inj_maps'] = {}
        
        for cmn in self.component_names:
            if self.components[cmn].has_map:    
                plot_data['map_data']['inj_maps'][cmn] = self.plot_skymaps(cmn,save_figures=False,return_mapdata=True)
        
        ## save map data
        if os.path.exists(map_data_path):
            ## move to temp file
            temp_file = map_data_path + ".temp"
            with open(temp_file, "wb") as datafile:
                pickle.dump(plot_data,datafile)
            shutil.move(temp_file, map_data_path)
        else:
            with open(map_data_path, 'wb') as datafile:
                plot_data = pickle.dump(plot_data,datafile)
        print("Data for injected skymaps saved to {}".format(map_data_path))

@jax.jit
def bespoke_inv(A):


    """

    compute inverse without division by det; ...xv3xc3 input, or array of matrices assumed

    Credit to Eelco Hoogendoorn at stackexchange for this piece of wizardy. This is > 3 times
    faster than numpy's det and inv methods used in a fully vectorized way as of numpy 1.19.1

    https://stackoverflow.com/questions/21828202/fast-inverse-and-transpose-matrix-in-python

    """


    AI = jnp.empty_like(A)

    for i in range(3):
#        AI[...,i,:] = jnp.cross(A[...,i-2,:], A[...,i-1,:])
        AI = AI.at[...,i,:].set(jnp.cross(A[...,i-2,:], A[...,i-1,:])) ## jax version

    det = jnp.einsum('...i,...i->...', AI, A).mean(axis=-1)

    inv_T =  AI / det[...,None,None]

    # inverse by swapping the inverse transpose
    return jnp.swapaxes(inv_T, -1,-2), det
