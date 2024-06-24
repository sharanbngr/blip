import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import healpy as hp
import logging
import os, shutil, pickle
import time
from multiprocessing import Pool
from blip.src.utils import log_manager, catch_duplicates, gen_suffixes, catch_color_duplicates
from blip.src.geometry import geometry
from blip.src.sph_geometry import sph_geometry
from blip.src.clebschGordan import clebschGordan
from blip.src.astro import Population
from blip.src.instrNoise import instrNoise
import blip.src.astro as astro

from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


class submodel(geometry,sph_geometry,clebschGordan,instrNoise):
    '''
    Modular class that can represent either an injection or an analysis model. Will have different attributes depending on use case.
    
    Includes all information required to generate an injection or a likelihood/prior.
    
    New models (injection or analysis) should be added here.
    
    '''
    def __init__(self,params,inj,submodel_name,fs,f0,tsegmid,injection=False,suffix='',parallel_response=False):
        '''
        Each submodel should be defined as "[spectral]_[spatial]", save for the noise model, which is just "noise".
        
        e.g., "powerlaw_isgwb" defines a submodel with an isotropic spatial distribution and a power law spectrum.
        
        Resulting objects has different attributes depending on if it is to be used as an Injection component or part of our unified multi-signal Model.
        
        Arguments
        ------------
        params, inj (dict)  : params and inj config dictionaries as generated in run_blip.py
        submodel_name (str) : submodel name, defined as "[spectral]_[spatial]" (or just "noise")
        fs, f0 (array)      : frequency array and its LISA-characteristic-frequency-scaled counterpart (f0=fs/(2*fstar))
        tsegmid (array)     : array of time segment midpoints
        injection (bool)    : If True, generate the submodel as an injection component, rather than a Model submodel.
        suffix (str)        : String to append to parameter names, etc., to differentiate between duplicate submodels.
        parallel_response (bool)     : If True, employ multiprocessing for the response calculations. Default False. 
        
        Returns
        ------------
        submodel (object) : submodel with all needed attributes to serve as an Injection component or Model submodel as desired.
        
        '''
        
        ## preliminaries
        self.params = params
        self.inj = inj
        self.armlength = 2.5e9 ## armlength in meters
        self.fs = fs
        self.f0 = f0
        self.tsegmid = tsegmid
        self.time_dim = tsegmid.size
        self.name = submodel_name
        self.injection = injection
        geometry.__init__(self)
        
        ## remove the duplicate identifier if needed (powerlaw_isgwb-3 -> powerlaw_isgwb)
        submodel_full_name = submodel_name
        submodel_split = submodel_full_name.split('-')
        submodel_name = submodel_split[0]
        if len(submodel_split) == 1:
            submodel_count = ''
        elif len(submodel_split) == 2:
            submodel_count = ' ({})'.format(submodel_split[1])
        else:
            raise ValueError("'{}' is not a valid submodel/component specfication.".format(submodel_full_name))
        
        if submodel_full_name in params['alias'].keys():
            self.alias = params['alias'][submodel_full_name]
        
        if injection:
            self.truevals = {}
            ## for ease of use, assign the trueval dict to a variable
            if submodel_full_name in self.inj['truevals'].keys():
                self.injvals = self.inj['truevals'][submodel_full_name]
        else:
            self.fixedvals = {}
            if submodel_full_name in self.params['fixedvals'].keys():
                self.fixedvals |= self.params['fixedvals'][submodel_full_name]
        
        ## plot kwargs dict to allow for case-by-case exceptions to our usual plotting approach
        ## e.g., the population spectra look real weird as dotted lines.
        self.plot_kwargs = {}
            
        ## handle & return noise case in bespoke fashion, as it is quite different from the signal models
        if submodel_name == 'noise':
            self.spectral_parameters = [r'$\log_{10} (Np)$'+suffix, r'$\log_{10} (Na)$'+suffix]
            self.spatial_parameters = []
            self.parameters = self.spectral_parameters
            self.Npar = 2
            ## for plotting
            self.fancyname = "Instrumental Noise"
            self.color = 'dimgrey'
            self.has_map = False
            # Figure out which instrumental noise spectra to use
            if self.params['tdi_lev']=='aet':
                self.instr_noise_spectrum = self.aet_noise_spectrum
                if injection:
                    self.gen_noise_spectrum = self.gen_aet_noise
            elif self.params['tdi_lev']=='xyz':
                self.instr_noise_spectrum = self.xyz_noise_spectrum
                if injection:
                    self.gen_noise_spectrum = self.gen_xyz_noise
            elif self.params['tdi_lev']=='michelson':
                self.instr_noise_spectrum = self.mich_noise_spectrum
                if injection:
                    self.gen_noise_spectrum = self.gen_michelson_noise
            else:
                raise ValueError("Unknown specification of 'tdi_lev'; can be 'michelson', 'xyz', or 'aet'.")
            if not injection:
                ## prior transform
                self.prior = self.instr_noise_prior
                ## covariance calculation
                self.cov = self.compute_cov_noise
            else:
                ## truevals
                self.truevals[r'$\log_{10} (Np)$'] = self.injvals['log_Np']
                self.truevals[r'$\log_{10} (Na)$'] = self.injvals['log_Na']
                ## save the frozen noise spectra
                self.frozen_spectra = self.instr_noise_spectrum(self.fs,self.f0,Np=10**self.injvals['log_Np'],Na=10**self.injvals['log_Na'])
            
            return
        
        elif submodel_name == 'fixednoise':
            self.spectral_parameters = []
            self.spatial_parameters = []
            self.parameters = []
            self.Npar = 0
            ## for plotting
            self.fancyname = "Known Instrumental Noise"
            self.color = 'dimgrey'
            self.has_map = False
            self.fixedspec = True
            # Figure out which instrumental noise spectra to use
            if self.params['tdi_lev']=='aet':
                self.instr_noise_spectrum = self.aet_noise_spectrum
                if injection:
                    self.gen_noise_spectrum = self.gen_aet_noise
            elif self.params['tdi_lev']=='xyz':
                self.instr_noise_spectrum = self.xyz_noise_spectrum
                if injection:
                    self.gen_noise_spectrum = self.gen_xyz_noise
            elif self.params['tdi_lev']=='michelson':
                self.instr_noise_spectrum = self.mich_noise_spectrum
                if injection:
                    self.gen_noise_spectrum = self.gen_michelson_noise
            else:
                raise ValueError("Unknown specification of 'tdi_lev'; can be 'michelson', 'xyz', or 'aet'.")
            if not injection:
                ## prior transform
                self.prior = self.fixed_model_wrapper_prior
                ## covariance calculation
                self.cov_fixed = self.compute_cov_noise([self.fixedvals['log_Np'],self.fixedvals['log_Na']])
                self.cov = self.compute_cov_fixed
            else:
                raise ValueError("Fixed submodels are not supported for injections. Use the corresponding unfixed submodel.")
            
            return

        else:
            self.parameters = []
            self.spectral_parameters = []
            self.spatial_parameters = []
            ## for convenience, so there's no need to specify e.g., "population_population"
            if submodel_name == 'population':
                self.spectral_model_name = self.spatial_model_name = submodel_name
            else:
                self.spectral_model_name, self.spatial_model_name = submodel_name.split('_')
            
            
        
        ###################################################
        ###            BUILD NEW MODELS HERE            ###
        ###################################################

        ## assignment of spectrum
        if self.spectral_model_name == 'powerlaw':
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha$', r'$\log_{10} (\Omega_0)$']
            self.omegaf = self.powerlaw_spectrum
            self.fancyname = "Power Law"+submodel_count
            if not injection:
                self.spectral_prior = self.powerlaw_prior
            else:
                self.truevals[r'$\alpha$'] = self.injvals['alpha']
                self.truevals[r'$\log_{10} (\Omega_0)$'] = self.injvals['log_omega0']
        elif self.spectral_model_name == 'twothirdspowerlaw':
            ## it may be worth implementing a more general fixed powerlaw model
            ## but this suffices for investigating the effects of the stellar-origin binary background
            self.spectral_parameters = self.spectral_parameters + [r'$\log_{10} (\Omega_0)$']
            self.omegaf = self.twothirdspowerlaw_spectrum
            self.fancyname = r'$\alpha=2/3$'+" Power Law"+submodel_count
            if not injection:
                self.spectral_prior = self.fixedpowerlaw_prior
            else:
                self.truevals[r'$\log_{10} (\Omega_0)$'] = self.injvals['log_omega0']
        elif self.spectral_model_name == 'brokenpowerlaw':
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha_1$',r'$\log_{10} (\Omega_0)$',r'$\alpha_2$',r'$\log_{10} (f_{break})$']
            self.omegaf = self.broken_powerlaw_spectrum
            self.fancyname = "Broken Power Law"+submodel_count
            if not injection:
                self.spectral_prior = self.broken_powerlaw_prior
            else:
                self.truevals[r'$\alpha_1$'] = self.injvals['alpha1']
                self.truevals[r'$\log_{10} (\Omega_0)$'] = self.injvals['log_omega0']
                self.truevals[r'$\alpha_2$'] = self.injvals['alpha2']
                self.truevals[r'$\log_{10} (f_{break})$'] = self.injvals['log_fbreak']
        
        elif self.spectral_model_name == 'fixedalpha1brokenpowerlaw':
            self.spectral_parameters = self.spectral_parameters + [r'$\log_{10} (\Omega_0)$',r'$\alpha_2$',r'$\log_{10} (f_{break})$',r'$\delta$']
            self.omegaf = self.broken_powerlaw_fixed_a1_spectrum
            self.fancyname = "Broken Power Law"+submodel_count
            if not injection:
                self.spectral_prior = self.broken_powerlaw_fixed_a1_prior
                if 'alpha_1' not in self.fixedvals.keys():
                    raise KeyError("Fixed-alpha_1 broken power law spectral model selected, but no low-frequeny slope parameter (alpha_1) was provided to the fixedvals dict.")
            else:
                self.truevals[r'$\log_{10} (\Omega_0)$'] = self.injvals['log_omega0']
                self.truevals[r'$\alpha_2$'] = self.injvals['alpha2']
                self.truevals[r'$\log_{10} (f_{break})$'] = self.injvals['log_fbreak']
                self.truevals[r'$\delta$'] = self.injvals['delta']
        
        elif self.spectral_model_name == 'truncatedpowerlaw':
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha$', r'$\log_{10} (\Omega_0)$', r'$\log_{10} (f_{\mathrm{cut}})$']
            self.omegaf = self.truncated_powerlaw_3par_spectrum
            self.fancyname = "Truncated Power Law"+submodel_count
            if not injection:
                if 'log_fscale' not in self.fixedvals.keys():
                    print("Warning: Truncated power law spectral model selected, but no scaling parameter (fscale) was provided to the fixedvals dict. Defaulting to fscale=4e-4 Hz.")
                    self.fixedvals['log_fscale'] = np.log10(4e-4)
                self.spectral_prior = self.truncated_powerlaw_3par_prior
            else:
                self.truevals[r'$\alpha$'] = self.injvals['alpha']
                self.truevals[r'$\log_{10} (\Omega_0)$'] = self.injvals['log_omega0']
                self.truevals[r'$\log_{10} (f_{\mathrm{cut}})$'] = self.injvals['log_fcut']
                self.truevals[r'$\log_{10} (f_{\mathrm{scale}})$'] = np.log10(4e-4)
                ## this is a bit hacky but oh well. Solves an issue that comes up if you use the 3par TPL for an injection.
                self.fixedvals = {'log_fscale':np.log10(4e-4)}
        
        elif self.spectral_model_name == 'truncatedpowerlaw4par':
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha$', r'$\log_{10} (\Omega_0)$', r'$\log_{10} (f_{\mathrm{cut}})$',r'$\log_{10} (f_{\mathrm{scale}})$']
            self.omegaf = self.truncated_powerlaw_4par_spectrum
            self.fancyname = "4-Parameter Truncated Power Law"+submodel_count
            if not injection:
                self.spectral_prior = self.truncated_powerlaw_4par_prior
            else:
                self.truevals[r'$\alpha$'] = self.injvals['alpha']
                self.truevals[r'$\log_{10} (\Omega_0)$'] = self.injvals['log_omega0']
                self.truevals[r'$\log_{10} (f_{\mathrm{cut}})$'] = self.injvals['log_fcut']
                self.truevals[r'$\log_{10} (f_{\mathrm{scale}})$'] = self.injvals['log_fscale']
                
        elif self.spectral_model_name == 'fixedtruncatedpowerlaw':
            self.fixed_spec = True
            self.spectral_parameters = self.spectral_parameters
            self.omegaf = self.fixed_truncated_powerlaw_spectrum
            self.fancyname = "Fixed Truncated Power Law"+submodel_count
            if not injection:
                self.spectral_prior = self.fixed_model_wrapper_prior
                self.fixed_args = [self.fixedvals['alpha'],self.fixedvals['log_omega0'],self.fixedvals['log_fcut'],self.fixedvals['log_fscale']]
            else:
                raise ValueError("Fixed submodels are not supported for injections. Use the corresponding unfixed submodel.")
        
        elif self.spectral_model_name == 'mwspec':
            ## this is a spectral model tailored to analyses of the MW foreground
            # it is a truncated power law with alpha = 2/3 and fscale = 4e-4
            # and astrophysically-motivated prior bounds
            self.spectral_parameters = self.spectral_parameters + [r'$\log_{10} (\Omega_0)$', r'$\log_{10} (f_{\mathrm{cut}})$',r'$\log_{10} (f_{\mathrm{scale}})$']
            self.omegaf = self.truncated_powerlaw_fixedalpha_spectrum
            self.fancyname = "MW Foreground"+submodel_count
            if not injection:
                self.fixedvals['alpha'] = 2/3
                self.spectral_prior = self.mwspec_prior
            else:
                raise ValueError("mwspec is an inference-only spectral submodel. Use the truncatedpowerlaw submodel for injections.")
        
        elif self.spectral_model_name == 'lmcspec':
            ## this is a spectral model tailored to analyses of the LMC SGWB
            # it is a broken power law with alpha_1 = 2/3
            # and astrophysically-motivated prior bounds
            self.spectral_parameters = self.spectral_parameters + [r'$\log_{10} (\Omega_0)$',r'$\alpha_2$',r'$\log_{10} (f_{break})$',r'$\delta$']
            self.omegaf = self.broken_powerlaw_fixed_a1_spectrum
            self.fancyname = "LMC Spectrum"+submodel_count
            if not injection:
                self.fixedvals['alpha_1'] = 2/3
                self.spectral_prior = self.lmcspecbpl_prior
            else:
                raise ValueError("lmcspec is an inference-only spectral submodel. Use the truncatedpowerlaw submodel for injections.")
        
        elif self.spectral_model_name == 'population':
            if not injection:
                raise ValueError("Populations are injection-only.")
            popdict = self.inj['popdict'][submodel_full_name]
            if popdict['name'] is not None:
                self.fancyname = popdict['name']
            else:
                self.fancyname = "DWD Population"+submodel_count
            self.population = Population(self.params,self.inj,self.fs,popdict)
            self.compute_Sgw = self.population.Sgw_wrapper
            self.omegaf = self.population.omegaf_wrapper
            self.ispop = True
            self.plot_kwargs |= {'ls':'-','lw':0.75,'alpha':0.6}
        
        else:
            ValueError("Unsupported spectrum type. Check your spelling or add a new spectrum model!")
        
        ## assignment of response and spatial methods
        response_kwargs = {}
        
        ## This is the isotropic spatial model, and has no additional parameters.
        if self.spatial_model_name == 'isgwb':
            if self.params['tdi_lev'] == 'michelson':
                if parallel_response:
                    self.response = self.isgwb_mich_response_parallel
                    self.response_non_parallel = self.isgwb_mich_response ## useful for data frequencies, external regen
                else:
                    self.response = self.isgwb_mich_response
            elif self.params['tdi_lev'] == 'xyz':
                if parallel_response:
                    self.response = self.isgwb_xyz_response_parallel
                    self.response_non_parallel = self.isgwb_xyz_response ## useful for data frequencies, external regen
                else:
                    self.response = self.isgwb_xyz_response
                
            elif self.params['tdi_lev'] == 'aet':
                if parallel_response:
                    self.response = self.isgwb_aet_response_parallel
                    self.response_non_parallel = self.isgwb_aet_response ## useful for data frequencies, external regen
                else:
                    self.response = self.isgwb_aet_response
            else:
                raise ValueError("Invalid specification of tdi_lev. Can be 'michelson', 'xyz', or 'aet'.")
            
            ## compute response matrix
            self.response_mat = self.response(f0,tsegmid,**response_kwargs)
            
            ## plotting stuff
            self.fancyname = "Isotropic "+self.fancyname
            self.subscript = "_{\mathrm{I}}"
            self.color='darkorange'
            self.has_map = False

            if not injection:
                ## prior transform
                self.prior = self.isotropic_prior
                self.cov = self.compute_cov_isgwb
            else:
                ## create a wrapper b/c isotropic and anisotropic injection responses are different
                self.inj_response_mat = self.response_mat
        
        ## This is the spherical harmonic spatial model. It is the workhorse of the spherical harmonic anisotropic analysis.
        ## It can also be used to perform arbitrary injections in the spherical harmonic basis via direct specification of the blms.
        elif self.spatial_model_name == 'sph':
            
            if injection:
                if self.inj['inj_basis'] == 'pixel':
                    raise ValueError("Only astrophysical injections are supported in the pixel basis. Spherical harmonic injections must use the spherical harmonic basis.")
                self.lmax = self.inj['inj_lmax']
            else:
                self.lmax = self.params['lmax']
            
            ## almax is twice the blmax
            self.almax = 2*self.lmax
            response_kwargs['set_almax'] = self.almax
            
            if self.params['tdi_lev']=='michelson':
                if parallel_response:
                    self.response = self.asgwb_mich_response_parallel
                    self.response_non_parallel = self.asgwb_mich_response ## useful for data frequencies, external regen
                else:
                    self.response = self.asgwb_mich_response  
            elif self.params['tdi_lev']=='xyz':
                if parallel_response:
                    self.response = self.asgwb_xyz_response_parallel
                    self.response_non_parallel = self.asgwb_xyz_response ## useful for data frequencies, external regen
                else:
                    self.response = self.asgwb_xyz_response
            elif self.params['tdi_lev']=='aet':
                if parallel_response:
                    self.response = self.asgwb_aet_response_parallel
                    self.response_non_parallel = self.asgwb_aet_response ## useful for data frequencies, external regen
                else:
                    self.response = self.asgwb_aet_response
            else:
                raise ValueError("Invalid specification of tdi_lev. Can be 'michelson', 'xyz', or 'aet'.")
            
            ## compute response matrix
            self.response_mat = self.response(f0,tsegmid,**response_kwargs)
            
            ## plotting stuff
            self.fancyname = "Anisotropic "+self.fancyname
            self.subscript = "_{\mathrm{A}}"
            self.color = 'teal'
            self.has_map = True
            self.basis = 'sph'
            
            # add the blms
            blm_parameters = gen_blm_parameters(self.lmax)
            
            ## save the blm start index for the prior, then add the blms to the parameter list
            self.blm_start = len(self.spectral_parameters)
            self.spatial_parameters = self.spatial_parameters + blm_parameters
            
            if not injection:
                self.fixed_map = False
                ## set sph indices
                self.blm_m0_idx = []
                self.blm_amp_idx = []
                self.blm_phase_idx = []
                cnt = 0
                for lval in range(1, self.lmax + 1):
                    for mval in range(lval + 1):
                        if mval == 0:
                            self.blm_m0_idx.append(cnt)
                            cnt = cnt + 1
                        else:
                            ## amplitude, phase
                            self.blm_amp_idx.append(cnt)
                            self.blm_phase_idx.append(cnt+1)
                            cnt = cnt + 2
                ## set prior, cov
                self.prior = self.sph_prior
                self.cov = self.compute_cov_asgwb
            else:
                ## get blm truevals
                val_list = self.blms_2_blm_params(self.injvals['blms'])
                
                for param, val in zip(blm_parameters,val_list):
                    self.truevals[param] = val
                
                ## get alms
                self.alms_inj = np.array(self.compute_skymap_alms(self.injvals['blms']).tolist())
#                import pdb; pdb.set_trace()
                ## get sph basis skymap
                self.sph_skymap =  hp.alm2map(self.alms_inj[0:hp.Alm.getsize(self.almax)],self.params['nside'])
                ## get response integrated over the Ylms
                self.summ_response_mat = self.compute_summed_response(self.alms_inj)
                ## create a wrapper b/c isotropic and anisotropic injection responses are different
                self.inj_response_mat = self.summ_response_mat
        
        ## Handle all the static (non-inferred) astrophysical spatial distributions together due to their similarities
        elif self.spatial_model_name in ['galaxy','dwarfgalaxy','lmc','pointsource','twopoints','pointsources','population','fixedgalaxy','hotpixel','pixiso','popmap']:
            
            ## the astrophysical spatial models are mostly injection-only, with some exceptions.
            if self.spatial_model_name in ['galaxy','dwarfgalaxy','lmc','pointsource','twopoints','population'] and not injection:
                raise ValueError("This model is injection-only.")
            
            self.has_map = True
            
            if (Injection and inj['inj_basis']=='pixel') or (not Injection and params['model_basis']=='pixel'):
                basis = 'pixel'
            else:
                basis = 'sph'
            self.basis = basis
            
            ## set lmax for sph case & define responses
            if basis == 'sph':
                ## almax is twice the blmax
                self.lmax = self.inj['inj_lmax']
                self.almax = 2*self.lmax
                response_kwargs['set_almax'] = self.almax
                if self.params['tdi_lev']=='michelson':
                    if parallel_response:
                        self.response = self.asgwb_mich_response_parallel
                        self.response_non_parallel = self.asgwb_mich_response ## useful for data frequencies, external regen
                    else:
                        self.response = self.asgwb_mich_response  
                elif self.params['tdi_lev']=='xyz':
                    if parallel_response:
                        self.response = self.asgwb_xyz_response_parallel
                        self.response_non_parallel = self.asgwb_xyz_response ## useful for data frequencies, external regen
                    else:
                        self.response = self.asgwb_xyz_response
                elif self.params['tdi_lev']=='aet':
                    if parallel_response:
                        self.response = self.asgwb_aet_response_parallel
                        self.response_non_parallel = self.asgwb_aet_response ## useful for data frequencies, external regen
                    else:
                        self.response = self.asgwb_aet_response
                else:
                    raise ValueError("Invalid specification of tdi_lev. Can be 'michelson', 'xyz', or 'aet'.")
            elif basis == 'pixel':
                if self.params['tdi_lev']=='michelson':
                    if parallel_response:
                        self.response = self.pixel_mich_response_parallel
                        self.response_non_parallel = self.pixel_mich_response ## useful for data frequencies, external regen
                    else:
                        self.response = self.pixel_mich_response     
                elif self.params['tdi_lev']=='xyz':
                    if parallel_response:
                        self.response = self.pixel_xyz_response_parallel
                        self.response_non_parallel = self.pixel_xyz_response ## useful for data frequencies, external regen
                    else:
                        self.response = self.pixel_xyz_response
                elif self.params['tdi_lev']=='aet':
                    if parallel_response:
                        self.response = self.pixel_aet_response_parallel
                        self.response_non_parallel = self.pixel_aet_response ## useful for data frequencies, external regen
                    else:
                        self.response = self.pixel_aet_response
                else:
                    raise ValueError("Invalid specification of tdi_lev. Can be 'michelson', 'xyz', or 'aet'.")
            
            
            ## model-specific quantities
            ## injection-only models
            if self.spatial_model_name == 'galaxy':
                ## store the high-level MW truevals for the hierarchical analysis
                self.truevals[r'$r_{\mathrm{h}}$'] = self.injvals['rh']
                self.truevals[r'$z_{\mathrm{h}}$'] = self.injvals['zh']
                ## plotting stuff
                self.fancyname = "Galactic Foreground"
                self.subscript = "_{\mathrm{G}}"
                self.color = 'mediumorchid'
                ## generate skymap
                self.skymap = astro.generate_galactic_foreground(self.injvals['rh'],self.injvals['zh'],self.params['nside'])
                ## mask to only the first four scale heights
                mask = self.skymap > (1/np.e**4)*np.max(self.skymap)
                self.skymap = self.skymap * mask
            elif self.spatial_model_name == 'lmc':
                ## plotting stuff
                self.fancyname = "LMC"
                self.subscript = "_{\mathrm{LMC}}"
                self.color = 'darkmagenta'
                ## generate skymap
                self.skymap = astro.generate_sdg(self.params['nside']) ## sdg defaults are for the LMC
            elif self.spatial_model_name == 'dwarfgalaxy':
                ## plotting stuff
                self.fancyname = "Dwarf Galaxy"+submodel_count
                self.subscript = "_{\mathrm{DG}}"
                self.color = 'maroon'
                ## generate skymap
                self.skymap = astro.generate_sdg(self.params['nside'],ra=self.injvals['sdg_RA'], dec=self.injvals['sdg_DEC'], D=self.injvals['sdg_dist'], r=self.injvals['sdg_rad'], N=self.injvals['sdg_N'])
            elif self.spatial_model_name == 'pointsource':
                ## plotting stuff
                self.fancyname = "Point Source"+submodel_count
                self.subscript = "_{\mathrm{1P}}"
                self.color = 'forestgreen'
                ## generate skymap
                ## some flexibility, can be defined in either (RA,DEC) or (theta,phi)
                if 'ra' in self.injvals.keys() and 'dec' in self.injvals.keys():
                    coord1, coord2 = self.injvals['ra'], self.injvals['dec']
                    convention = 'radec'
                elif 'theta' in self.injvals.keys() and 'phi' in self.injvals.keys():
                    coord1, coord2 = self.injvals['theta'], self.injvals['phi']
                    convention = 'healpy'
                else:
                    raise ValueError("Using pointsource spatial model but either no coordinates were provided to the truevals dict or invalid notation was used.")
                self.skymap = astro.generate_point_source(coord1,coord2,self.params['nside'],convention=convention)
            elif self.spatial_model_name == 'pointsources':
                ## plotting stuff
                self.fancyname = "Multiple Point Sources"+submodel_count
                self.subscript = "_{\mathrm{NP}}"
                self.color = 'forestgreen'
                ## generate skymap
                ## some flexibility, can be defined in either (RA,DEC) or (theta,phi)
                if 'radec_list' in self.injvals.keys():
                    coord_list = self.injvals['radec_list']
                    convention = 'radec'
                elif 'thetaphi_list' in self.injvals.keys():
                    coord_list = self.injvals['thetaphi_list']
                    convention = 'healpy'
                else:
                    raise ValueError("Using pointsources spatial model but either no coordinates were provided to the truevals dict or invalid notation was used.")
                self.skymap = astro.generate_point_sources(coord_list,self.params['nside'],convention=convention)
            elif self.spatial_model_name == 'twopoints':
                ## revisit this when I have duplicates sorted, maybe unnecessary (could just have 2x point source injection components)
                ## plotting stuff
                self.fancyname = "Two Point Sources"+submodel_count
                self.subscript = "_{\mathrm{2P}}"
                self.color = 'gold'
                ## generate skymap
                self.skymap = astro.generate_two_point_source(self.injvals['theta_1'],self.injvals['phi_1'],self.injvals['theta_2'],self.injvals['phi_2'],self.params['nside'])
            elif self.spatial_model_name == 'population':
                ## flag the fact that we have a population skymap
                self.skypop = True
                ## plotting stuff
                self.subscript = "_{\mathrm{P}}"
                self.color = 'midnightblue'
                if self.spectral_model_name != 'population':
                    ## generate population if still needed
                    popdict = self.inj['popdict'][submodel_full_name]
                    if popdict['name'] is not None:
                        self.fancyname = popdict['name']
                    else:
                        self.fancyname = "DWD Population"+submodel_count
                    self.population = Population(self.params,self.inj,self.fs,popdict)
                self.skymap = self.population.skymap
            ## inference models
            elif self.spatial_model_name == 'fixedgalaxy':
                ## get the fixed values
                if 'rh' in self.fixedvals.keys():
                    rh = self.fixedvals['rh']
                else:
                    print("Warning: Using fixedgalaxy spatial model but no 'rh' fixed value was provided. Defaulting to Breivik+2020 thin disk galaxy (rh = 2.9 kpc.)")
                    rh = 2.9
                if 'zh' in self.fixedvals.keys():
                    zh = self.fixedvals['zh']
                else:
                    print("Warning: Using fixedgalaxy spatial model but no 'zh' fixed value was provided. Defaulting to Breivik+2020 thin disk galaxy (zh = 0.3 kpc).")
                    zh = 0.3
                ## plotting stuff
                self.fancyname = "Galactic Foreground"
                self.subscript = "_{\mathrm{G}}"
                self.color = 'mediumorchid'
                ## generate skymap
                self.skymap = astro.generate_galactic_foreground(rh,zh,self.params['nside'])
                ## mask to only the first four scale heights
                mask = self.skymap > (1/np.e**4)*np.max(self.skymap)
                self.skymap = self.skymap * mask
                self.fixed_map = True
            elif self.spatial_model_name == 'hotpixel':
                ## get the fixed values
                ## some flexibility, can be defined in either (RA,DEC) or (theta,phi)
                if 'ra' in self.fixedvals.keys() and 'dec' in self.fixedvals.keys():
                    coord1, coord2 = self.fixedvals['ra'], self.fixedvals['dec']
                    convention = 'radec'
                elif 'theta' in self.fixedvals.keys() and 'phi' in self.fixedvals.keys():
                    coord1, coord2 = self.fixedvals['theta'], self.fixedvals['phi']
                    convention = 'healpy'
                else:
                    raise ValueError("Using hotpixel spatial model but either no coordinates were provided to the fixedvals dict or invalid notation was used.")
                ## plotting stuff
                self.fancyname = "Point Source"
                self.subscript = "_{\mathrm{1P}}"
                self.color = 'forestgreen'
                self.skymap = astro.generate_point_source(coord1,coord2,self.params['nside'],convention=convention,pad=True)
                self.fixed_map = True
            
            elif self.spatial_model_name == 'pixiso':
                self.fancyname = "Pixel Isotropic"
                self.subscript = "_{\mathrm{PI}}"
                self.color = 'forestgreen'
                self.skymap = np.ones(hp.nside2npix(self.params['nside']))
                self.fixed_map = True
            
            elif self.spatial_model_name == 'popmap':
                self.fancyname = "Population Skymap"
                self.subscript = "_{\mathrm{PM}}"
                self.color = 'mediumorchid'
                popkey = self.fixedvals['pop_id']
                popdict = self.inj['popdict'][popkey]
                if popdict['name'] is not None:
                    self.fancyname = popdict['name']
                else:
                    self.fancyname = "DWD Population"+submodel_count
                self.population = Population(self.params,self.inj,self.fs,popdict,map_only=True)
                self.skymap = self.population.skymap
                self.fixed_map = True
            
            else:
                raise ValueError("Astrophysical submodel type not found. Did you add a new model to the list at the top of this section?")
            
            ## compute response matrix
            if basis == 'pixel':
                response_kwargs['skymap_inj'] = self.skymap #/(np.sum(self.skymap)*hp.nside2pixarea(self.params['nside']))
            self.response_mat = self.response(f0,tsegmid,**response_kwargs)
            
            ## process skymap
            if not injection:
                if basis == 'sph':
                    self.process_astro_skymap_model(self.skymap)
                    self.prior = self.fixedsky_prior
                    self.cov = self.compute_cov_fixed_asgwb
                elif basis=='pixel':
#                    self.process_astro_skymap_pixel_model(self.skymap)
                    self.summ_response_mat = self.response_mat
                    self.prior = self.fixedsky_prior
                    self.cov = self.compute_cov_fixed_asgwb
                else:
                    raise TypeError("Basis was not defined, or was incorrectly defined.")
            else:
                if basis == 'sph':
                    self.process_astro_skymap_injection(self.skymap)
                elif basis == 'pixel':
                    self.inj_response_mat = self.response_mat
                else:
                    raise TypeError("Basis was not defined, or was incorrectly defined.")
            
            
#            ## compute response matrix
#            self.response_mat = self.response(f0,tsegmid,**response_kwargs)
#            if basis == 'pixel':
#                self.inj_response_mat = self.response_mat

        ## Parameterized astrophysical spatial distributions.
        ## Distinct from the fixedsky/injection-only models as we need spatial inference infrastructure
        ## pixel-basis only
        elif self.spatial_model_name in ['1parametermw','2parametermw']:
            
            ## enforce pixel basis
            if params["model_basis"] != "pixel":
                raise ValueError("Parameterized astrophysical spatial submodels are only supported in the pixel basis. (You have set basis={}.)".format(params["model_basis"]))
            
            ## calculate pixel area
            self.dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])
            
            ## set starting index for spatial model parameters
            self.spatial_start = len(self.spectral_parameters)
            
            ## set response functions
            if self.params['tdi_lev']=='michelson':
                if parallel_response:
                    self.response = self.unconvolved_pixel_mich_response_parallel
                    self.response_non_parallel = self.unconvolved_pixel_mich_response ## useful for data frequencies, external regen
                else:
                    self.response = self.unconvolved_pixel_mich_response
            elif self.params['tdi_lev']=='xyz':
                if parallel_response:
                    self.response = self.unconvolved_pixel_xyz_response_parallel
                    self.response_non_parallel = self.unconvolved_pixel_xyz_response ## useful for data frequencies, external regen
                else:
                    self.response = self.unconvolved_pixel_xyz_response
            elif self.params['tdi_lev']=='aet':
                if parallel_response:
                    self.response = self.unconvolved_pixel_aet_response_parallel
                    self.response_non_parallel = self.unconvolved_pixel_aet_response ## useful for data frequencies, external regen
                else:
                    self.response = self.unconvolved_pixel_aet_response
            else:
                raise ValueError("Invalid specification of tdi_lev. Can be 'michelson', 'xyz', or 'aet'.")
            
            ## 2-parameter Milky Way model
            if self.spatial_model_name == '1parametermw':
                ## model to infer the Milky Way spatial distribution, using a simplified 1-parameter model of the Galaxy
                ## only infers the vertical scale height z_h
                ## plotting stuff
                self.fancyname = "1-Parameter Milky Way"
                self.subscript = "_{\mathrm{G}}"
                self.color = 'mediumorchid'
                self.has_map = True
                self.fixed_map = False
                self.parameterized_map = True

                ## Initialize the galaxy grid
                self.galaxy = astro.Galaxy_Model(self.params['nside'],gal_rad=14,gal_height=6,max_rh=3.5,max_zh=1.5,fix_rh=self.fixedvals['rh'])
                self.max_sky_extent = self.galaxy.max_skymap
                
                ## Set the parameterized spatial model function
                self.compute_skymap = self.galaxy.mw_mapmaker_1par
                
                ## mask maps to maximum allowed spatial extent
                self.mask = self.galaxy.max_skymap > (1/np.e**4)*np.max(self.galaxy.max_skymap)
                self.masked_skymap = self.galaxy.max_skymap * self.mask
                self.mask_idx = np.flatnonzero(self.mask)
                
                ## ensure normalization
                self.masked_skymap = self.masked_skymap/(np.sum(self.masked_skymap)*self.dOmega)
                
                
                ## set response kwargs
                response_kwargs['masked_skymap'] = self.masked_skymap
                
                self.spatial_parameters = [r'$z_\mathrm{h}$']
                self.prior = self.mw1parameter_prior
                self.cov = self.compute_cov_parameterized_asgwb
            
            elif self.spatial_model_name == '2parametermw':
                ## model to infer the Milky Way spatial distribution, using a basic 2-parameter model of the Galaxy
                ## plotting stuff
                self.fancyname = "2-Parameter Milky Way"
                self.subscript = "_{\mathrm{G}}"
                self.color = 'mediumorchid'
                self.has_map = True
                self.fixed_map = False
                self.parameterized_map = True

                ## Initialize the galaxy grid
                self.galaxy = astro.Galaxy_Model(self.params['nside'],gal_rad=14,gal_height=6,max_rh=3.5,max_zh=1.5)
                self.max_sky_extent = self.galaxy.max_skymap
                
                ## Set the parameterized spatial model function
                self.compute_skymap = self.galaxy.mw_mapmaker_2par
                
                ## mask maps to maximum allowed spatial extent
                self.mask = self.galaxy.max_skymap > (1/np.e**4)*np.max(self.galaxy.max_skymap)
                self.masked_skymap = self.galaxy.max_skymap * self.mask
                self.mask_idx = np.flatnonzero(self.mask)
                
                ## ensure normalization
                self.masked_skymap = self.masked_skymap/(np.sum(self.masked_skymap)*self.dOmega)
                
                
                ## set response kwargs
                response_kwargs['masked_skymap'] = self.masked_skymap
                
                self.spatial_parameters = [r'$r_\mathrm{h}$',r'$z_\mathrm{h}$']
                self.prior = self.mw2parameter_prior
                self.cov = self.compute_cov_parameterized_asgwb
            
            else:
                raise ValueError("Parameterized astrophysical spatial submodel type not found. Did you add a new model to the list at the top of this section?")
            
            self.response_mat = self.response(f0,tsegmid,**response_kwargs)
            
        else:
            raise ValueError("Invalid specification of spatial model name ('{}').".format(self.spatial_model_name))
        
        #############################
        ##      FOR TESTING        ##
        #############################
#        np.save(self.params['out_dir']+'/response_'+submodel_full_name+'.npy',self.response_mat) 
#        print("Saving response array to "+self.params['out_dir']+'/response_'+submodel_full_name+'.npy')
        ## store final parameter list and count
        self.parameters = self.parameters + self.spectral_parameters + self.spatial_parameters
        if not injection:               
            self.Npar = len(self.parameters)
        ## store response kwargs for use elsewhere as needed
        self.response_kwargs = response_kwargs
        ## add suffix to parameter names and trueval keys, if desired
        ## (we need this in the multi-model or duplicate model case)
        if suffix != '':
            if injection:
                updated_truevals = {parameter+suffix:self.truevals[parameter] for parameter in self.parameters}
                self.truevals = updated_truevals
            updated_spectral_parameters = [parameter+suffix for parameter in self.spectral_parameters]
            updated_spatial_parameters = [parameter+suffix for parameter in self.spatial_parameters]
            updated_parameters = updated_spectral_parameters+updated_spatial_parameters
            if len(updated_parameters) != len(self.parameters):
                raise ValueError("If you've added a new variety of parameters above, you'll need to update this bit of code too!")
            self.spectral_parameters = updated_spectral_parameters
            self.spatial_parameters = updated_spatial_parameters
            self.parameters = updated_parameters
            
        
        return
    

    #############################
    ##    Spectral Functions   ##
    #############################
    def powerlaw_spectrum(self,fs,alpha,log_omega0):
        '''
        Function to calculate a simple power law spectrum.
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        alpha (float)        : slope of the power law
        log_omega0 (float)   : power law amplitude in units of log dimensionless GW energy density at f_ref
        
        Returns
        -----------
        spectrum (array of floats) : the resulting power law spectrum
        
        '''
        return 10**(log_omega0)*(fs/self.params['fref'])**alpha
    
    def twothirdspowerlaw_spectrum(self,fs,log_omega0):
        '''
        Function to calculate a simple power law spectrum, fixed to the alpha=2/3 prediction for the stellar origin binary background.
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        log_omega0 (float)   : power law amplitude in units of log dimensionless GW energy density at f_ref
        
        Returns
        -----------
        spectrum (array of floats) : the resulting power law spectrum
        
        '''
        return 10**(log_omega0)*(fs/self.params['fref'])**(2/3)
    
    def broken_powerlaw_spectrum(self,fs,alpha_1,log_omega0,alpha_2,log_fbreak):
        '''
        Function to calculate a broken power law spectrum.
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        alpha_1 (float)      : slope of the first power law
        log_omega0 (float)   : power law amplitude of the first power law in units of log dimensionless GW energy density at f_ref
        alpha_2 (float)      : slope of the second power law
        log_fbreak (float)   : log of the break frequency ("knee") in Hz
        
        Returns
        -----------
        spectrum (array of floats) : the resulting broken power law spectrum
        
        '''
        delta = 0.1
        fbreak = 10**log_fbreak
        norm = (fbreak/self.params['fref'])**alpha_1 ## this normalizes the broken powerlaw such that its first leg matches the equivalent standard power law
        return norm * (10**log_omega0)*(fs/fbreak)**(alpha_1) * (1+(fs/fbreak)**(1/delta))**((alpha_1-alpha_2)*delta)
    
    def broken_powerlaw_fixed_a1_spectrum(self,fs,log_omega0,alpha_2,log_fbreak,delta):
        '''
        Function to calculate a broken power law spectrum, with a fixed low-frequency slope and variable turnover scale.
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        log_omega0 (float)   : power law amplitude of the first power law in units of log dimensionless GW energy density at f_ref
        alpha_2 (float)      : slope of the second power law
        log_fbreak (float)   : log of the break frequency ("knee") in Hz
        
        Returns
        -----------
        spectrum (array of floats) : the resulting broken power law spectrum
        
        '''
        fbreak = 10**log_fbreak
        norm = (fbreak/self.params['fref'])**self.fixedvals['alpha_1'] ## this normalizes the broken powerlaw such that its first leg matches the equivalent standard power law
        return norm * (10**log_omega0)*(fs/fbreak)**(self.fixedvals['alpha_1']) * ((1+(fs/fbreak)**(1/delta)))**((self.fixedvals['alpha_1']-alpha_2)*delta)
    
    def truncated_powerlaw_4par_spectrum(self,fs,alpha,log_omega0,log_fcut,log_fscale):
        '''
        Function to calculate a tanh-truncated power law spectrum.
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        alpha (float)        : slope of the power law
        log_omega0 (float)   : power law amplitude of the power law in units of log dimensionless GW energy density at f_ref (if left un-truncated)
        log_fcut (float)     : log of the cut frequency ("knee") in Hz
        log_fscale           : log of the cutoff scale factor in Hz
        
        Returns
        -----------
        spectrum (array of floats) : the resulting truncated power law spectrum
        
        '''
        fcut = 10**log_fcut
        fscale = 10**log_fscale
        return 0.5 * (10**log_omega0)*(fs/self.params['fref'])**(alpha) * (1+jnp.tanh((fcut-fs)/fscale))
    
    def truncated_powerlaw_3par_spectrum(self,fs,alpha,log_omega0,log_fcut):
        '''
        Function to calculate a tanh-truncated power law spectrum with a set truncation scale.
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        alpha (float)        : slope of the power law
        log_omega0 (float)   : power law amplitude of the power law in units of log dimensionless GW energy density at f_ref (if left un-truncated)
        log_fcut (float)     : log of the cut frequency ("knee") in Hz
        
        Returns
        -----------
        spectrum (array of floats) : the resulting truncated power law spectrum
        
        '''
        fcut = 10**log_fcut
        fscale = 10**self.fixedvals['log_fscale']
        return 0.5 * (10**log_omega0)*(fs/self.params['fref'])**(alpha) * (1+jnp.tanh((fcut-fs)/fscale))
    
    def truncated_powerlaw_fixedalpha_spectrum(self,fs,log_omega0,log_fcut,log_fscale):
        '''
        Function to calculate a tanh-truncated power law spectrum with a set low-f slope.
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        log_omega0 (float)   : power law amplitude of the power law in units of log dimensionless GW energy density at f_ref (if left un-truncated)
        log_fcut (float)     : log of the cut frequency ("knee") in Hz
        log_fscale           : log of the cutoff scale factor in Hz
        
        Returns
        -----------
        spectrum (array of floats) : the resulting truncated power law spectrum
        
        '''
        fcut = 10**log_fcut
        fscale = 10**log_fscale
        return 0.5 * (10**log_omega0)*(fs/self.params['fref'])**(self.fixedvals['alpha']) * (1+jnp.tanh((fcut-fs)/fscale))
    
    def truncated_powerlaw_2par_spectrum(self,fs,log_omega0,log_fcut):
        '''
        Function to calculate a tanh-truncated power law spectrum with a set truncation scale and low-f slope.
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        log_omega0 (float)   : power law amplitude of the power law in units of log dimensionless GW energy density at f_ref (if left un-truncated)
        log_fcut (float)     : log of the cut frequency ("knee") in Hz
        
        Returns
        -----------
        spectrum (array of floats) : the resulting truncated power law spectrum
        
        '''
        fcut = 10**log_fcut
        fscale = 10**self.fixedvals['log_fscale']
        return 0.5 * (10**log_omega0)*(fs/self.params['fref'])**(self.fixedvals['alpha']) * (1+jnp.tanh((fcut-fs)/fscale))
    
    def fixed_truncated_powerlaw_spectrum(self,fs):
        '''
        Function to calculate a tanh-truncated power law spectrum with all parameters fixed.
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        
        Returns
        -----------
        spectrum (array of floats) : the resulting truncated power law spectrum
        
        '''
        fcut = 10**self.fixedvals['log_fcut']
        fscale = 10**self.fixedvals['log_fscale']
        return 0.5 * (10**self.fixedvals['log_omega0'])*(fs/self.params['fref'])**(self.fixedvals['alpha']) * (1+jnp.tanh((fcut-fs)/fscale))
    
    def compute_Sgw(self,fs,omegaf_args):
        '''
        Wrapper function to generically calculate the associated stochastic gravitational wave PSD (S_gw)
            for a spectral model given in terms of the dimensionless GW energy density Omega(f)
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        omegaf_args (list)   : list of arguments for the relevant Omega(f) function
        
        Returns
        -----------
        Sgw (array of floats) : the resulting GW PSD
        
        '''
        H0 = 2.2*10**(-18) ## Hubble constant, = 67.88 km/s/Mpc
        Omegaf = self.omegaf(fs,*omegaf_args)
        Sgw = Omegaf*(3/(4*fs**3))*(H0/jnp.pi)**2
        return Sgw
    
    
    #############################
    ##          Priors         ##
    #############################
    def isotropic_prior(self,theta):
        '''
        Isotropic prior transform. Just serves as a wrapper for the spectral prior, as no additional foofaraw is necessary.
        
        Arguments
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled for the spectral parameters.
            
        '''
        return self.spectral_prior(theta)
    
    def fixedsky_prior(self,theta):
        '''
        Fixed sky prior transform. Just serves as a wrapper for the spectral prior, as no additional foofaraw is necessary.
        
        Arguments
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled for the spectral parameters.
            
        '''
        return self.spectral_prior(theta)
    
    def sph_prior(self,theta):
        '''
        Spherical harmonic anisotropic prior transform. Combines a generic spectral prior function with the spherical harmonic priors for the desired lmax.
        
        Arguments
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled for both the spectral and spatial parameters.
        '''
        
        ## spectral prior takes everything up to 
        spectral_theta = self.spectral_prior(theta[:self.blm_start])
        
        # Calculate lmax from the size of theta blm arrays. The shape is
        # given by size = (lmax + 1)**2 - 1. The '-1' is because b00 is
        # an independent parameter
#        lmax = jnp.sqrt( len(theta[self.blm_start:]) + 1 ) - 1
#        lmax = self.lmax
        
        
#        ## theta indices for m == 0 blms
#        blm_m0_idx = self.blm_m0_idx + self.blm_start
#        ## theta indices for m != 0 *amplitude* parameters
#        blm_amp_idx = self.blm_amp_idx + self.blm_start
#        ## theta indices for m != 0 *phase* parameters
#        blm_phase_idx = self.blm_phase_idx + self.blm_start
        
        sph_base = jnp.zeros(len(theta[self.blm_start:]))
#        sph_theta = [0 for ii in theta[self.blm_start:]]
        
        for ii in self.blm_m0_idx:
            sph_base = sph_base.at[ii].set(6*theta[ii+self.blm_start] - 3)
        for ii in self.blm_amp_idx:
            sph_base = sph_base.at[ii].set(3*theta[ii+self.blm_start])
        for ii in self.blm_phase_idx:
#            sph_base = sph_base.at[ii].set(2*jnp.pi*theta[ii+self.blm_start] - jnp.pi)
            sph_base = sph_base.at[ii].set(jnp.remainder(2*jnp.pi*theta[ii+self.blm_start],2*jnp.pi) - jnp.pi)
        
        sph_theta = [draw for draw in sph_base]
        
        ## removing the lmax safety check to be compatible with JAX/jit.
#        if lmax.is_integer():
#            lmax = int(lmax)
#        else:
#            raise ValueError('Illegitimate theta size passed to the spherical harmonic prior')
        
#        lmax = int(lmax)
        
        # The rest of the priors define the blm parameter space
#        sph_theta = []
#
#        ## counter for the rest of theta
#        cnt = self.blm_start
#
#        for lval in range(1, lmax + 1):
#            for mval in range(lval + 1):
#
#                if mval == 0:
#                    sph_theta.append(6*theta[cnt] - 3)
#                    cnt = cnt + 1
#                else:
#                    ## prior on amplitude, phase
#                    sph_theta.append(3* theta[cnt])
#                    sph_theta.append(2*jnp.pi*theta[cnt+1] - jnp.pi)
#                    cnt = cnt + 2

        return spectral_theta+sph_theta
    
    def mw1parameter_prior(self,theta):
        '''
        Hierarchical anisotropic prior transform. Combines a generic spectral prior function with the hierarchical astrophysical prior.
        
        Arguments
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled for both the spectral and spatial parameters.
        '''
        spectral_theta = self.spectral_prior(theta[:self.spatial_start])
        
        zh = 1.45*theta[self.spatial_start] + 0.05
        
        mw_theta = [zh]
        
        return spectral_theta+mw_theta
    
    def mw2parameter_prior(self,theta):
        '''
        Hierarchical anisotropic prior transform. Combines a generic spectral prior function with the hierarchical astrophysical prior.
        
        Arguments
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled for both the spectral and spatial parameters.
        '''
        spectral_theta = self.spectral_prior(theta[:self.spatial_start])
        
        rh = 2*theta[self.spatial_start] + 2
        zh = 1.45*theta[self.spatial_start+1] + 0.05
        
        mw_theta = [rh,zh]
        
        return spectral_theta+mw_theta
        
        
    def instr_noise_prior(self,theta):


        '''
        Prior function for only instrumental noise

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref, Np and Na

        '''


        # Unpack: Theta is defined in the unit cube
        log_Np, log_Na = theta

        # Transform to actual priors
        log_Np = -5*log_Np - 39
        log_Na = -5*log_Na - 46

        return [log_Np, log_Na]
    
    def powerlaw_prior(self,theta):


        '''
        Prior function for an isotropic stochastic backgound analysis.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha and log(Omega0)

        '''


        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        alpha       =  10*theta[0] - 5
        log_omega0  = -26*theta[1] + 12
        
        return [alpha, log_omega0]
    
    def fixedpowerlaw_prior(self,theta):


        '''
        Prior function for a power law with fixed slope.
        
        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha and log(Omega0)

        '''


        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        log_omega0  = -26*theta[0] + 12
        
        return [log_omega0]
    
    def broken_powerlaw_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a broken power law spectral model.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha_1, log(Omega_0), alpha_2, and log(f_break).

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        alpha_1 = 10*theta[0] - 4
        log_omega0 = -22*theta[1] + 8
        alpha_2 = 40*theta[2]
        log_fbreak = -2*theta[3] - 2

        return [alpha_1, log_omega0, alpha_2, log_fbreak]
    
    def broken_powerlaw_fixed_a1_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a 4-parameter broken power law spectral model.
        Fixed low-frequency slope.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_0), log(f_cut), log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        
        log_omega0 = -10*theta[0] - 4
        alpha_2 = 4*theta[1] + self.fixedvals['alpha_1'] ## must be greater than alpha_1
        log_fbreak = -2*theta[2] - 2
        delta = 0.99*theta[3] + 0.01

        return [log_omega0, alpha_2, log_fbreak, delta]
    
    def truncated_powerlaw_4par_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a 4-parameter truncated power law spectral model.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_0), log(f_cut), and log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        alpha = 10*theta[0] - 5
        log_omega0 = -22*theta[1] + 8
        log_fcut = -2*theta[2] - 2
        log_fscale = -2*theta[3] - 2
        

        return [alpha, log_omega0, log_fcut, log_fscale]
    
    def truncated_powerlaw_3par_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a 3-parameter truncated power law spectral model.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_0), and log(f_cut)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        alpha = 10*theta[0] - 5
        log_omega0 = -22*theta[1] + 8
        log_fcut = -2*theta[2] - 2
        

        return [alpha, log_omega0, log_fcut]
    
    def truncated_powerlaw_2par_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a 2-parameter truncated power law spectral model.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as log(Omega_0) and log(f_cut)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        log_omega0 = -22*theta[0] + 8
        log_fcut = -2*theta[1] - 2
        

        return [log_omega0, log_fcut]
    
    def mwspec_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a 2-parameter truncated power law spectral model.
        
        Bounds are astrophysically-motivated and tailored to expectations of the MW foreground.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_0), and log(f_cut)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        log_omega0 = -2*theta[0] - 4
        log_fcut = -0.7*theta[1] - 2.4
        log_fscale = -2*theta[2] - 2
        
        
        return [log_omega0, log_fcut, log_fscale]
    
    def lmcspec_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a 3-parameter truncated power law spectral model.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_0), log(f_cut), log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        log_omega0 = -4*theta[0] - 8
        log_fcut = -1*theta[1] - 2
        log_fscale = -1*theta[2] - 3
        

        return [log_omega0, log_fcut, log_fscale]
    
    def lmcspecbpl_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a 4-parameter broken power law spectral model.
        Tailored for the LMC DWD SGWB spectrum.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_0), log(f_cut), log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        
        log_omega0 = -4*theta[0] - 8
        alpha_2 = 2*theta[1] + self.fixedvals['alpha_1'] ## must be greater than alpha_1
        log_fbreak = -1*theta[2] - 2
        delta = 0.99*theta[3] + 0.01

        return [log_omega0, alpha_2, log_fbreak, delta]
    
    def fixed_model_wrapper_prior(self,theta):


        '''
        Wrapper function to allow fixed "models" to function within Model.prior

        Parameters
        -----------

        theta   : float
            Shoudl always be None or [].

        Returns
        ---------

        theta   :   float
            empty list

        '''
        return []
    
    
    #############################
    ## Covariance Calculations ##
    #############################
    def compute_cov_noise(self,theta):
        '''
        Computes the noise covariance for a given draw of log_Np, log_Na
        
        Arguments
        ----------
        theta (float)   :  A list or numpy array containing samples from a unit cube.
        
        Returns
        ----------
        cov_noise (array) : The corresponding 3 x 3 x frequency x time covariance matrix for the detector noise submodel.
        
        '''
        ## unpack priors
        log_Np, log_Na = theta

        Np, Na =  10**(log_Np), 10**(log_Na)

        ## Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fs,self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = jnp.repeat(cov_noise[:, :, :, jnp.newaxis], self.time_dim, axis=3)
        
        return cov_noise
    
    def compute_cov_isgwb(self,theta):
        '''
        Computes the covariance matrix contribution from a generic isotropic stochastic GW signal.
        
        Arguments
        ----------
        theta (float)   :  A list or numpy array containing samples from a unit cube.
        
        Returns
        ----------
        cov_sgwb (array) : The corresponding 3 x 3 x frequency x time covariance matrix for an isotropic SGWB submodel.
        
        '''
        ## Signal PSD
        Sgw = self.compute_Sgw(self.fs,theta)

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*self.response_mat
        
        return cov_sgwb
    
    def compute_cov_asgwb(self,theta):
        '''
        Computes the covariance matrix contribution from a generic anisotropic stochastic GW signal.
        
        Arguments
        ----------
        theta (float)   :  A list or numpy array containing samples from a unit cube.
        
        Returns
        ----------
        cov_sgwb (array) : The corresponding 3 x 3 x frequency x time covariance matrix for an anisotropic SGWB submodel.
        
        '''
        ## Signal PSD
        Sgw = self.compute_Sgw(self.fs,theta[:self.blm_start])
        
        ## get skymap and integrate over alms
        summ_response_mat = self.compute_summed_response(self.compute_skymap_alms(theta[self.blm_start:]))

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*summ_response_mat
        
        return cov_sgwb

    def compute_cov_fixed_asgwb(self,theta):
        '''
        Computes the covariance matrix contribution from an anisotropic stochastic GW signal with a known (assumed) sky distribution.
        
        Arguments
        ----------
        theta (float)   :  A list or numpy array containing samples from a unit cube.
        
        Returns
        ----------
        cov_sgwb (array) : The corresponding 3 x 3 x frequency x time covariance matrix for an anisotropic SGWB submodel.
        
        '''
        ## Signal PSD
        Sgw = self.compute_Sgw(self.fs,theta)
        
        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        ## the response has been preconvolved with the assumed sky distribution
        cov_sgwb = Sgw[None, None, :, None]*self.summ_response_mat
        
        return cov_sgwb
    
    def compute_cov_parameterized_asgwb(self,theta):
        '''
        Computes the covariance matrix contribution from a explicitly parameterized (i.e. not a generic spherical harmonic model), pixel-basis anisotropic stochastic GW signal.
        
        Arguments
        ----------
        theta (float)   :  A list or numpy array containing samples from a unit cube.
        
        Returns
        ----------
        cov_sgwb (array) : The corresponding 3 x 3 x frequency x time covariance matrix for an anisotropic SGWB submodel.
        
        '''
        ## Signal PSD
        Sgw = self.compute_Sgw(self.fs,theta[:self.spatial_start])
        
        ## get skymap and integrate over alms
        summ_response_mat = self.compute_summed_pixel_response(self.mask_and_norm_pixel_skymap(self.compute_skymap(*theta[self.spatial_start:])))

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*summ_response_mat
        
        return cov_sgwb
    
    def compute_cov_fixedspec_parameterized_asgwb(self,theta):
        '''
        Computes the covariance matrix contribution from a explicitly parameterized (i.e. not a generic spherical harmonic model), pixel-basis anisotropic stochastic GW signal.
        
        Assumes a fixed spectral model. Only compatible with fixedspec spectral models.
        
        Arguments
        ----------
        theta (float)   :  A list or numpy array containing samples from a unit cube.
        
        Returns
        ----------
        cov_sgwb (array) : The corresponding 3 x 3 x frequency x time covariance matrix for an anisotropic SGWB submodel.
        
        '''
        ## Signal PSD
        Sgw = self.fixed_Sgw
        
        ## get skymap and integrate over alms
        summ_response_mat = self.compute_summed_pixel_response(self.mask_and_norm_pixel_skymap(self.compute_skymap(*theta[self.spatial_start:])))

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*summ_response_mat
        
        return cov_sgwb
    
    def compute_cov_fixed(self,dummy_theta):
        '''
        Wrapper to allow for "models" that are fixed a priori.
        
        Arguments
        ----------
        dummy_theta (NoneType)   :  Should always be None; meant to allow for wrapper to integrate with Model.Likelihood
        
        Returns
        ----------
        cov_fixed (array) : The precomputed 3 x 3 x frequency x time covariance matrix for the fixed model.
        
        '''
        return self.cov_fixed
    
    ##########################################
    ##   Skymap and Response Calculations   ##
    ##########################################
    
    def compute_skymap_alms(self,blm_params):
        '''
        Function to compute the anisotropic skymap a_lms from the blm parameters.
        
        Arguments
        ----------
        blm_params (array of complex floats) : the blm parameters
        
        Returns
        ----------
        alm_vals (array of complex floats) : the corresponding alms
        
        '''
        ## Spatial distribution
        blm_vals = self.blm_params_2_blms(blm_params)
        alm_vals = self.blm_2_alm(blm_vals)

        ## normalize and return
        return alm_vals/(alm_vals[0] * jnp.sqrt(4*jnp.pi))
    
    def compute_summed_response(self,alms):
        '''
        Function to compute the integrated, skymap-convolved anisotropic response
        
        Arguments
        ----------
        alms (array of complex floats) : the spherical harmonic alms
        
        Returns
        ----------
        summ_response_mat (array) : the sky/alm-integrated response (3 x 3 x frequency x time)
        
        '''
        return jnp.einsum('ijklm,m', self.response_mat, alms)
    
    def compute_summed_pixel_response(self,pixelmap):
        '''
        Function to compute the integrated, skymap-convolved anisotropic response for an arbitrary skymap in the pixel basis.
        
        Arguments
        ----------
        pixelmap (healpy array) : the pixel-basis skymap
        
        Returns
        ----------
        summ_response_mat (array) : the sky-integrated response (3 x 3 x frequency x time)
        
        '''
        return (self.dOmega/(4*jnp.pi))*jnp.einsum('ijklm,m', self.response_mat, pixelmap)
    
    def process_astro_skymap_injection(self,skymap):
        '''
        
        Function that takes in an astrophysical pixel skymap and:
            - calculates all associated sph quantities
            - computes corresponding blm parameter truevals
            - convolves with response
            
        Arguments
        -----------
        skymap (healpy array) : pixel-basis astrophysical skymap
        
        '''
        ## transform to blms
        self.astro_blms = astro.skymap_pix2sph(skymap,self.lmax)
        ## get corresponding truevals
        inj_blms = self.blms_2_blm_params(self.astro_blms)

        blm_parameters = gen_blm_parameters(self.lmax)

        for param, val in zip(blm_parameters,inj_blms):
            self.truevals[param] = val
        
        self.alms_inj = np.array(self.blm_2_alm(self.astro_blms))
        self.alms_inj = self.alms_inj/(self.alms_inj[0] * np.sqrt(4*np.pi))
        self.sph_skymap = hp.alm2map(self.alms_inj[0:hp.Alm.getsize(self.almax)],self.params['nside'])
        ## get response integrated over the Ylms
        self.summ_response_mat = self.compute_summed_response(self.alms_inj)
        ## create a wrapper b/c isotropic and anisotropic injection responses are different
        self.inj_response_mat = self.summ_response_mat
        
        return
    
    def process_astro_skymap_model(self,skymap):
        '''
        
        Function that takes in an astrophysical pixel skymap and:
            - calculates all associated sph quantities
            - convolves with response
            - sets sample-time response to be the map-convolved 
        
        This is intended for use with models that assume a fixed spatial distribution (e.g., fixedgalaxy, hotpixel).
            
        Arguments
        -----------
        skymap (healpy array) : pixel-basis astrophysical skymap
        
        '''
        ## transform to blms
        self.astro_blms = astro.skymap_pix2sph(skymap,self.lmax)
        ## and then to alms        
        self.astro_alms = np.array(self.blm_2_alm(self.astro_blms))
        self.astro_alms = self.astro_alms/(self.astro_alms[0] * np.sqrt(4*np.pi))
        self.sph_skymap = hp.alm2map(self.astro_alms[0:hp.Alm.getsize(self.almax)],self.params['nside'])
        ## get response integrated over the Ylms
        self.summ_response_mat = self.compute_summed_response(self.astro_alms)
        ## backup the unconvolved response matrix and set the default response to the skymap-convolved one
        self.unconvolved_response_mat = self.response_mat
        self.response_mat = self.summ_response_mat
        
        return
    
    def mask_and_norm_pixel_skymap(self,skymap):
        '''
        
        Function that takes in a modeled astrophysical skymap and:
            - masks to the pixels where we have computed the response
            - normalizes the masked map such that the sky integral = 1
        
        Arguments
        -----------
        skymap (healpy array) : pixel-basis astrophysical skymap
        
        '''
        masked_map = skymap[self.mask_idx]
        
        return masked_map/(np.sum(masked_map)*self.dOmega)
    
#    def process_astro_skymap_pixel_model(self,skymap):
#        '''
#        
#        Function that takes in an astrophysical pixel skymap and:
#            - convolves with response
#            - sets sample-time response to be the map-convolved 
#        
#        This is intended for use with models that assume a fixed spatial distribution (e.g., fixedgalaxy, hotpixel).
#            
#        Arguments
#        -----------
#        skymap (healpy array) : pixel-basis astrophysical skymap
#        
#        '''
#        ## transform to blms
##        self.astro_blms = astro.skymap_pix2sph(skymap,self.lmax)
#        ## and then to alms        
##        self.astro_alms = np.array(self.blm_2_alm(self.astro_blms))
##        self.astro_alms = self.astro_alms/(self.astro_alms[0] * np.sqrt(4*np.pi))
##        self.sph_skymap = hp.alm2map(self.astro_alms[0:hp.Alm.getsize(self.almax)],self.params['nside'])
#        ## get response integrated over the Ylms
#        self.summ_response_mat = self.compute_summed_pixel_response(skymap)
#        ## backup the unconvolved response matrix and set the default response to the skymap-convolved one
#        self.unconvolved_response_mat = self.response_mat
#        self.response_mat = self.summ_response_mat
#        
#        return
    
    def recompute_response(self,f0=None,tsegmid=None):
        '''
        Function to recompute the LISA response matrices if needed.
        
        When we save the Injection object, we delete the LISA response of each injection, as to do otherwise takes up egregious amounts of disk space.
        This allows us to recompute them identically as desired.
        
        Arguments
        -------------
        f0 (array)      : LISA-characteristic-frequency-scaled frequency array at which to compute the response (f0=fs/(2*fstar))
        tsegmid (array)     : array of time segment midpoints at which to compute the response
        
        Returns
        --------------
        response_mat (array) : The associated response for this submodel. 
        '''
        ## allow for respecification of frequency/time grid, but avoid needless computation of extant response matrices
        fsame = True
        tsame = True
        if f0 is not None:
            if f0.shape != self.f0.shape:
                fsame = False
            elif not np.all(f0==self.f0):
                fsame = False
        else:
            f0 = self.f0
        if tsegmid is not None:
            if tsegmid.shape != self.tsegmid.shape:
                tsame = False
            elif not np.all(tsegmid==self.tsegmid):
                tsame = False
        else:
            tsegmid = self.tsegmid
        
        tf_same = tsame and fsame
        
        ## if we're using the same frequencies and times, first check to see if there's already a response connected to the submodel:
        if tf_same and hasattr(self,'response_mat'):
            print("Attempted to recompute response matrix, but there is already an attached response matrix at these times and frequencies. Returning the original...")
            return self.response_mat
        else:
            return self.response(f0,tsegmid,**self.response_kwargs)



###################################################
###      UNIFIED MODEL PRIOR & LIKELIHOOD       ###
###################################################

@register_pytree_node_class
class Model():
    '''
    Class to house all model attributes in a modular fashion.
    '''
    def __init__(self,params,inj,fs,f0,tsegmid,rmat):
        
        '''
        Model() parses a Model string from the params file. This is of the form of an arbitrary number of "+"-delimited submodel types.
        Each submodel should be defined as "[spectral]_[spatial]", save for the noise model, which is just "noise".
        
        e.g., "noise+powerlaw_isgwb+truncated-powerlaw_sph" defines a model with noise, an isotropic SGWB with a power law spectrum,
            and a (spherical harmonic model for) an anisotropic SGWB with a truncated power law spectrum.
        
        Arguments
        ------------
        params, inj (dict)  : params and inj config dictionaries as generated in run_blip.py
        fs, f0 (array)      : frequency array and its LISA-characteristic-frequency-scaled counterpart (f0=fs/(2*fstar))
        tsegmid (array)     : array of time segment midpoints
        rmat (array)        : the data correllation matrix for all LISA arms
        
        Returns
        ------------
        Model (object) : Unified Model comprised of an arbitrary number of noise/signal submodels, with a corresponding unified prior and likelihood.
        
        '''
        
        self.fs = fs
        self.f0 = f0
        self.tsegmid = tsegmid
        self.params = params
        self.inj = inj
        ## separate into submodels
        self.submodel_names = params['model'].split('+')
        
        ## separate into submodels
        base_component_names = params['model'].split('+')
        
        ## check for and differentiate duplicate injections
        ## this will append 1 (then 2, then 3, etc.) to any duplicate submodel names
        ## we will also generate appropriate variable suffixes to use in plots, etc..
        self.submodel_names = catch_duplicates(base_component_names)
        suffixes = gen_suffixes(base_component_names)
        
        ## initialize submodels
        self.submodels = {}
        self.Npar = 0
        self.parameters = {}
        all_parameters = []
        spectral_parameters = []
        spatial_parameters = []
        self.blm_phase_idx = []
        for submodel_name, suffix in zip(self.submodel_names,suffixes):
            sm = submodel(params,inj,submodel_name,fs,f0,tsegmid,suffix=suffix)
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
        
        ## update colors as needed
        catch_color_duplicates(self)
        
        ## assign reference to data for use in likelihood
        self.rmat = rmat
    
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

    
class Injection():#geometry,sph_geometry):
    '''
    Class to house all injection attributes in a modular fashion.
    '''
    def __init__(self,params,inj,fs,f0,tsegmid):
        '''
        Injection() parses a Injection string from the params file. This is of the form of an arbitrary number of "+"-delimited submodel types.
        Each submodel should be defined as "[spectral]_[spatial]", save for the noise model, which is just "noise".
        
        e.g., "noise+powerlaw_isgwb+truncated-powerlaw_sph" defines an injection with noise, an isotropic SGWB with a power law spectrum,
            and a (spherical harmonic description of) an anisotropic SGWB with a truncated power law spectrum.
        
        Arguments
        ------------
        params, inj (dict)  : params and inj config dictionaries as generated in run_blip.py
        fs, f0 (array)      : frequency array and its LISA-characteristic-frequency-scaled counterpart (f0=fs/(2*fstar))
        tsegmid (array)     : array of time segment midpoints
        
        Returns
        ------------
        Injection (object)  : Unified Injection comprised of an arbitrary number of noise/signal injection components, with a variety of helper functions to aid in the BLIP injection procedure.
        
        '''
        self.params = params
        self.inj = inj
        
        self.frange = fs
        self.f0 = f0
        self.tsegmid = tsegmid
        
        ## separate into components
        self.component_names = inj['injection'].split('+')
        N_inj = len(self.component_names)
        
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
        

        ## activate multithreading if desired
        if inj['parallel_inj'] and inj['inj_nthread']>1:
            name_args = [(cmn,suff) for cmn, suff in zip(self.component_names,suffixes)]
            print("Building all injection components in parallel. Number of threads: {}.".format(inj['inj_nthread']))
            with Pool(inj['inj_nthread']) as pool:
                component_list = list(pool.imap(self.add_component,name_args))
            for cm, component_name in zip(component_list,self.component_names):
                self.components[component_name] = cm
                self.truevals[component_name] = cm.truevals
                if cm.has_map:
                    self.plot_skymaps(component_name)
        elif inj['parallel_inj'] and inj['response_nthread']>1:
            for i, (component_name, suffix) in enumerate(zip(self.component_names,suffixes)):
                print("Building injection for {} (component {} of {})...".format(component_name,i+1,N_inj))
                t1 = time.time()
                cm = submodel(params,inj,component_name,fs,f0,tsegmid,injection=True,suffix=suffix,parallel_response=True)
                t2 = time.time()
                print("Time elapsed for component {} is {} s.".format(component_name,t2-t1))
                self.components[component_name] = cm
                self.truevals[component_name] = cm.truevals

                if cm.has_map:
                    self.plot_skymaps(component_name)
        else:
            for i, (component_name, suffix) in enumerate(zip(self.component_names,suffixes)):
                print("Building injection for {} (component {} of {})...".format(component_name,i+1,N_inj))
                t1 = time.time()
                cm = submodel(params,inj,component_name,fs,f0,tsegmid,injection=True,suffix=suffix)
                t2 = time.time()
                print("Time elapsed for component {} is {} s.".format(component_name,t2-t1))
                self.components[component_name] = cm
                self.truevals[component_name] = cm.truevals

                if cm.has_map:
                    self.plot_skymaps(component_name)
            
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
                hp.mollview(Omegamap_pix, coord=coord, title='Injected pixel map $\Omega (f = 1 mHz)$', unit="$\\Omega(f= 1mHz)$", cmap=self.params['colormap'])
                hp.graticule()
            
            if save_figures:
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
                hp.mollview(Omegamap_inj, coord=coord, title='Injected angular distribution map $\Omega (f = 1 mHz)$', unit="$\\Omega(f= 1mHz)$", cmap=self.params['colormap'])
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

def gen_blm_parameters(blmax):
    '''
    Function to make the blm parameter name strings for all blms of a given lmax, in the correct order.
    
    Arguments
    -----------
    blmax (int) : lmax for the blms
    
    Returns
    -----------
    blm_parameters (list of str) : Ordered list of blm parameter name strings
    
    '''
    
    blm_parameters = []
    for lval in range(1, blmax + 1):
        for mval in range(lval + 1):

            if mval == 0:
                blm_parameters.append(r'$b_{' + str(lval) + str(mval) + '}$' )
            else:
                blm_parameters.append(r'$|b_{' + str(lval) + str(mval) + '}|$' )
                blm_parameters.append(r'$\phi_{' + str(lval) + str(mval) + '}$' )
    
    return blm_parameters

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
