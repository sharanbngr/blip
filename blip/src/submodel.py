from enum import Enum
from dataclasses import dataclass

import blip.src.astro as astro
from blip.src.astro import Population
from blip.src.clebschGordan import clebschGordan
from blip.src.fast_geometry import fast_geometry
from blip.src.geometry import geometry
from blip.src.instrNoise import instrNoise
from blip.src.utils import get_robson19_shape_pars_from_tobs


import healpy as hp
import jax.numpy as jnp
import numpy as np


class SubmodelKind(Enum):
    """
    Kind of submodel specifier string.

    Most are 'spectral_spatial'.

    There are only two exceptions:
    - 'population', which is shorthand for 'population_population';
    - 'noise' and 'fixednoise', because there is no spatial model for noise.
    """
    NOISE = 1
    """'noise' or 'fixednoise' submodel."""
    POPULATION = 2
    """'population' submodel, shorthand for 'population_population'."""
    SPECTRAL_SPATIAL = 3
    """General spectral_spatial submodel specifier. Example: 'powerlaw_fixedgalaxy'."""


@dataclass
class SubmodelSpec:
    """
    Parsed (hence valid) specification of a submodel. Use this in submodel.__init__().
    
    To get a specification from a string, use blip.config.parse_model_spec().
    """
    name: str
    """Submodel name without duplicate count. Example: 'population-1' ->
    'population'."""
    kind: SubmodelKind
    """Kind of specifier string."""
    is_injection: bool
    """True if the submodel is intended to be used as part of an injection. False if it
    is intended for analysis."""
    spectral: str | None
    """Spectral model."""
    spatial: str | None
    """Spatial model."""
    count: str
    """Duplicate count enclosed by parenthesis. Example: 'powerlaw_isgwb-2' -> '(2)'."""
    raw_name: str
    """Complete submodel name. Used as unique id."""
    truevals: dict
    """True parameter values. For analysis submodels, this is derived from the aliased
    injection submodel."""

    # only for analysis submodels
    fixedvals: dict  # empty dict if missing
    """Fixed values of parameters. Only for analysis (not injection) submodels."""
    alias: str | None
    """Raw name of aliased injection submodel. Only used in analysis submodels."""


class submodel(fast_geometry,clebschGordan,instrNoise):
    '''
    Modular class that can represent either an injection or an analysis model. Will have different attributes depending on use case.

    Includes all information required to generate an injection or a likelihood/prior.

    New models (injection or analysis) should be added here.

    '''
    def __init__(self,params,inj,spec: SubmodelSpec,fs,f0,tsegmid,injection=False,suffix='',parallel_response=False):
        '''
        Each submodel should be defined as "[spectral]_[spatial]", save for the noise model, which is just "noise".

        e.g., "powerlaw_isgwb" defines a submodel with an isotropic spatial distribution and a power law spectrum.

        Resulting objects has different attributes depending on if it is to be used as an Injection component or part of our unified multi-signal Model.

        Arguments
        ------------
        params, inj (dict)  : params and inj config dictionaries as generated in run_blip.py
        spec (SubmodelSpec) : submodel specification, as parsed by config.parse_spec_analysis or parse_spec_injection
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
        self.injection = injection

        if 'lmax' in params.keys() and not injection:
            clebschGordan.__init__(self,params['lmax'])
        elif 'inj_lmax' in inj.keys() and injection:
            clebschGordan.__init__(self,inj['inj_lmax'])
        elif 'inj_lmax' not in inj.keys() and 'lmax' in params.keys() and injection:
            clebschGordan.__init__(self,params['lmax'])

        self.name = spec.raw_name
        self.kind = spec.kind
        self.name_split = (spec.name, spec.count)
        self.spectral_model_name = spec.spectral
        self.spatial_model_name = spec.spatial
        self.injvals = spec.truevals
        self.truevals = {}  # truevals keys will be processed and renamed case-by-case
        self.fixedvals = spec.fixedvals
        self.alias = spec.alias

        ## set some preliminary flags; these may get overwritten later
        self.has_map = False
        self.parameterized_map = False


        submodel_count = spec.count

        ## plot kwargs dict to allow for case-by-case exceptions to our usual plotting approach
        ## e.g., the population spectra look real weird as dotted lines.
        self.plot_kwargs = {}

        self.parameters = []
        self.spectral_parameters = []
        self.spatial_parameters = []

        if spec.kind == SubmodelKind.NOISE and spec.name == "noise":
            self.spectral_parameters = [r'$\log_{10} (N_p)$'+suffix, r'$\log_{10} (N_a)$'+suffix]
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
                self.truevals[r'$\log_{10} (N_p)$'] = self.injvals['log_Np']
                self.truevals[r'$\log_{10} (N_a)$'] = self.injvals['log_Na']
                ## save the frozen noise spectra
                self.frozen_spectra = self.instr_noise_spectrum(self.fs,self.f0,Np=10**self.injvals['log_Np'],Na=10**self.injvals['log_Na'])

            return

        elif spec.kind == SubmodelKind.NOISE and spec.name == "fixednoise":
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


        ###################################################
        ###            BUILD NEW MODELS HERE            ###
        ###################################################

        ## assignment of spectrum
        if self.spectral_model_name == 'powerlaw':
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha$', r'$\log_{10} (\Omega_{\rm ref})$']
            self.omegaf = self.powerlaw_spectrum
            self.fancyname = "Power Law"+submodel_count
            if not injection:
                self.spectral_prior = self.powerlaw_prior
            else:
                self.truevals[r'$\alpha$'] = self.injvals['alpha']
                self.truevals[r'$\log_{10} (\Omega_{\rm ref})$'] = self.injvals['log_omega0']
        elif self.spectral_model_name == 'twothirdspowerlaw':
            ## it may be worth implementing a more general fixed powerlaw model
            ## but this suffices for investigating the effects of the stellar-origin binary background
            self.spectral_parameters = self.spectral_parameters + [r'$\log_{10} (\Omega_{\rm ref})$']
            self.omegaf = self.twothirdspowerlaw_spectrum
            self.fancyname = r'$\alpha=2/3$'+" Power Law"+submodel_count
            if not injection:
                self.spectral_prior = self.fixedpowerlaw_prior
            else:
                self.truevals[r'$\log_{10} (\Omega_{\rm ref})$'] = self.injvals['log_omega0']

        elif self.spectral_model_name == 'fixedalphapowerlaw':
            if injection:
                raise ValueError("Fixed-value submodels are not supported for injections. Please use the 'powerlaw' submodel instead.")
            ## ensure alpha value is provided
            if 'alpha' not in self.fixedvals.keys():
                raise ValueError("The 'fixedalphapowerlaw' submodel requires the following parameters to be provided to the fixedvals dict: alpha.")
            self.spectral_parameters = self.spectral_parameters + [r'$\log_{10} (\Omega_{\rm ref})$']
            self.omegaf = self.fixedpowerlaw_spectrum
            self.fancyname = r'$\alpha='+'{}$'.format(self.fixedvals['alpha'])+" Power Law"+submodel_count
            self.spectral_prior = self.fixedpowerlaw_prior

        elif self.spectral_model_name == 'brokenpowerlaw':
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha_1$',r'$\log_{10} (\Omega_{\rm ref})$',r'$\alpha_2$',r'$\log_{10} (f_{\rm break})$']
            self.omegaf = self.broken_powerlaw_spectrum
            self.fancyname = "Broken Power Law"+submodel_count
            if not injection:
                self.spectral_prior = self.broken_powerlaw_prior
            else:
                self.truevals[r'$\alpha_1$'] = self.injvals['alpha1']
                self.truevals[r'$\log_{10} (\Omega_{\rm ref})$'] = self.injvals['log_omega0']
                self.truevals[r'$\alpha_2$'] = self.injvals['alpha2']
                self.truevals[r'$\log_{10} (f_{\rm break})$'] = self.injvals['log_fbreak']

        elif self.spectral_model_name == 'fixedalpha1brokenpowerlaw':
            self.spectral_parameters = self.spectral_parameters + [r'$\log_{10} (\Omega_{\rm ref})$',r'$\alpha_2$',r'$\log_{10} (f_{\rm break})$',r'$\delta$']
            self.omegaf = self.broken_powerlaw_fixed_a1_spectrum
            self.fancyname = "Broken Power Law"+submodel_count
            if not injection:
                self.spectral_prior = self.broken_powerlaw_fixed_a1_prior
                if 'alpha_1' not in self.fixedvals.keys():
                    raise KeyError("Fixed-alpha_1 broken power law spectral model selected, but no low-frequeny slope parameter (alpha_1) was provided to the fixedvals dict.")
            else:
                self.fixedvals[r'$\alpha_1$'] = self.injvals['alpha1']
                self.truevals[r'$\log_{10} (\Omega_{\rm ref})$'] = self.injvals['log_omega0']
                self.truevals[r'$\alpha_2$'] = self.injvals['alpha2']
                self.truevals[r'$\log_{10} (f_{\rm break})$'] = self.injvals['log_fbreak']
                self.truevals[r'$\delta$'] = self.injvals['delta']

        elif self.spectral_model_name == 'truncatedpowerlaw':
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha$', r'$\log_{10} (\Omega_{\rm ref})$', r'$\log_{10} (f_{\mathrm{cut}})$']
            self.omegaf = self.truncated_powerlaw_3par_spectrum
            self.fancyname = "Truncated Power Law"+submodel_count
            if not injection:
                if 'log_fscale' not in self.fixedvals.keys():
                    print("Warning: Truncated power law spectral model selected, but no scaling parameter (fscale) was provided to the fixedvals dict. Defaulting to fscale=4e-4 Hz.")
                    self.fixedvals['log_fscale'] = np.log10(4e-4)
                self.spectral_prior = self.truncated_powerlaw_3par_prior
            else:
                self.truevals[r'$\alpha$'] = self.injvals['alpha']
                self.truevals[r'$\log_{10} (\Omega_{\rm ref})$'] = self.injvals['log_omega0']
                self.truevals[r'$\log_{10} (f_{\mathrm{cut}})$'] = self.injvals['log_fcut']
                self.truevals[r'$\log_{10} (f_{\mathrm{scale}})$'] = np.log10(4e-4)
                ## this is a bit hacky but oh well. Solves an issue that comes up if you use the 3par TPL for an injection.
                self.fixedvals = {'log_fscale':np.log10(4e-4)}

        elif self.spectral_model_name == 'truncatedpowerlaw4par':
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha$', r'$\log_{10} (\Omega_{\rm ref})$', r'$\log_{10} (f_{\mathrm{cut}})$',r'$\log_{10} (f_{\mathrm{scale}})$']
            self.omegaf = self.truncated_powerlaw_4par_spectrum
            self.fancyname = "4-Parameter Truncated Power Law"+submodel_count
            if not injection:
                self.spectral_prior = self.truncated_powerlaw_4par_prior
            else:
                self.truevals[r'$\alpha$'] = self.injvals['alpha']
                self.truevals[r'$\log_{10} (\Omega_{\rm ref})$'] = self.injvals['log_omega0']
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
            self.spectral_parameters = self.spectral_parameters + [r'$\log_{10} (\Omega_{\rm ref})$', r'$\log_{10} (f_{\mathrm{cut}})$',r'$\log_{10} (f_{\mathrm{scale}})$']
            self.omegaf = self.truncated_powerlaw_fixedalpha_spectrum
            self.fancyname = "MW Foreground"+submodel_count
            if not injection:
                if 'alpha' not in self.fixedvals.keys():
                    print("Warning: No low-frequency slope (alpha) specified for MWspec spectral model. Defaulting to alpha=2/3.")
                    self.fixedvals['alpha'] = 2/3
                self.spectral_prior = self.mwspec_prior
            else:
                raise ValueError("mwspec is an inference-only spectral submodel. Use the truncatedpowerlaw submodel for injections.")

        elif self.spectral_model_name == 'mwspec3par':
            ## this is a more flexible spectral model tailored to analyses of the MW foreground
            # it is a 3-parameter truncated power law with astrophysically-motivated prior bounds
            # fscale parameter is fixed
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha$', r'$\log_{10} (\Omega_{\rm ref})$', r'$\log_{10} (f_{\mathrm{cut}})$']
            self.omegaf = self.truncated_powerlaw_3par_spectrum
            self.fancyname = "MW Foreground"+submodel_count
            if not injection:
                if 'log_fscale' not in self.fixedvals.keys():
                    print("Warning: 3-parameter MWspec (truncated power law + astrophysical priors) spectral model selected, but no scaling parameter (fscale) was provided to the fixedvals dict. Defaulting to astrophysically-motivated fscale=1.25 mHz.")
                    self.fixedvals['log_fscale'] = -2.907
                self.spectral_prior = self.mwspec3par_prior
            else:
                raise ValueError("mwspec is an inference-only spectral submodel. Use the truncatedpowerlaw submodel for injections.")

        elif self.spectral_model_name == 'mwspec4par':
            ## this is a more flexible spectral model tailored to analyses of the MW foreground
            # it is a 4-parameter truncated power law with astrophysically-motivated prior bounds
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha$', r'$\log_{10} (\Omega_{\rm ref})$', r'$\log_{10} (f_{\mathrm{cut}})$',r'$\log_{10} (f_{\mathrm{scale}})$']
            self.omegaf = self.truncated_powerlaw_4par_spectrum
            self.fancyname = "MW Foreground"+submodel_count
            if not injection:
                self.spectral_prior = self.mwspec4par_prior
            else:
                raise ValueError("mwspec is an inference-only spectral submodel. Use the truncatedpowerlaw submodel for injections.")

        elif self.spectral_model_name == 'doublebrokentruncpowerlaw':
            ## implementation of a broken, truncated power law foreground model.
            ## this is a variation of the tanh-truncated foreground, but with
            ## additional, time-dependent shape parameters due to subtraction of resolved systems
            ## for the BLIP implementation, it has been recast into Omega_GW space
            ## NOTE: This version has no fixed parameters.
                #def truncated_doublebroken_powerlaw_spectrum(self,fs,alpha_1, alpha_2, alpha_3, delta1, delta2, log_omega0, log_fcut, log_fscale, log_fbreak1, log_fbreak2):


            self.spectral_parameters = self.spectral_parameters + [r'$\alpha_1$', r'$\alpha_2$', r'$\alpha_3$', r'$\delta_1$', r'$\delta_2$', r'$\log_{10}\Omega_{\rm GW}$',r'$\log_{10}f_{\rm cut}$', r'$\log_{10}f_{\rm scale}$', r'$\log_{10}f_{\rm break1}$', r'$\log_{10}f_{\rm break2}$']
            self.omegaf = self.truncated_doublebroken_powerlaw_spectrum
            self.fancyname = "MW Foreground"+submodel_count
            if not injection:
                self.spectral_prior = self.truncated_doublebroken_powerlaw_prior
            else:
                ## define truevals
                self.truevals[r'$\alpha_1$'] = self.injvals['alpha_1']
                self.truevals[r'$\alpha_2$'] = self.injvals['alpha_2']
                self.truevals[r'$\alpha_3$'] = self.injvals['alpha_3']
                self.truevals[r'$\delta_1$'] = self.injvals['delta1']
                self.truevals[r'$\delta_2$'] = self.injvals['delta2']
                self.truevals[r'$\log_{10}\Omega_{\rm GW}$'] = jnp.log10(self.injvals['Omega_GW'])
                if 'fcut' in self.injvals.keys():
                    self.truevals[r'$\log_{10}f_{\rm cut}$'] = jnp.log10(self.injvals['fcut'])
                elif 'log_fcut' in self.injvals.keys():
                    self.truevals[r'$\log_{10}f_{\rm cut}$'] = self.injvals['log_fcut']
                if 'fscale' in self.injvals.keys():
                    self.truevals[r'$\log_{10}f_{\rm scale}$'] = jnp.log10(self.injvals['fscale'])
                elif 'log_fscale' in self.injvals.keys():
                    self.truevals[r'$\log_{10}f_{\rm scale}$'] = self.injvals['log_fscale']
                if 'fbreak1' in self.injvals.keys():
                    self.truevals[r'$\log_{10}f_{\rm break1}$'] = jnp.log10(self.injvals['fbreak1'])
                elif 'log_fbreak1' in self.injvals.keys():
                    self.truevals[r'$\log_{10}f_{\rm break1}$'] = self.injvals['log_fbreak1']
                if 'fbreak2' in self.injvals.keys():
                    self.truevals[r'$\log_{10}f_{\rm break2}$'] = jnp.log10(self.injvals['fbreak2'])
                elif 'log_fbreak2' in self.injvals.keys():
                    self.truevals[r'$\log_{10}f_{\rm break2}$'] = self.injvals['log_fbreak2']
                self.fixedvals = self.truevals

        elif self.spectral_model_name == 'bdspec':
            # A spectral model for either the MW bulge or the MW disk.
            self.spectral_parameters = self.spectral_parameters + [r'$\log_{10} (\Omega_{\rm ref})$', r'$\log_{10} (f_{\mathrm{cut}})$',r'$\log_{10} (f_{\mathrm{scale}})$']
            self.omegaf = self.truncated_powerlaw_fixedalpha_spectrum
            self.fancyname = "MW component"+submodel_count
            if not injection:
                if 'alpha' not in self.fixedvals.keys():
                    print("Warning: No low-frequency slope (alpha) specified for bdspec spectral model. Defaulting to alpha=0.5.")
                    self.fixedvals['alpha'] = 0.5
                self.spectral_prior = self.bdspec_prior
            else:
                raise ValueError("bdspec is an inference-only spectral submodel. Use the truncatedpowerlaw submodel for injections.")


        elif self.spectral_model_name == 'robson19foreground':
            ## implementation of the Robson+19 analytic foreground model.
            ## this is a variation of the tanh-truncated foreground, but with
            ## additional, time-dependent shape parameters due to subtraction of resolved systems
            ## for the BLIP implementation, it has been recast into Omega_GW space
            self.spectral_parameters = self.spectral_parameters + [r'$\log_{10}A$']
            self.omegaf = self.robson19_foreground_spectrum
            self.fancyname = "MW Foreground"+submodel_count
            if not injection:
                self.spectral_prior = self.robson19foreground_prior
                if 'T_obs' not in self.fixedvals.keys():
                    shape_fixedvals = get_robson19_shape_pars_from_tobs(self.params['dur']/3.154e7)
                    self.fixedvals |= shape_fixedvals
                else:
                    shape_fixedvals = get_robson19_shape_pars_from_tobs(self.fixedvals['T_obs'])
                    self.fixedvals |= shape_fixedvals
            else:
                ## define truevals
                if 'T_obs' not in self.injvals.keys():
                    raise ValueError("When simulated data with the Robson+19 foreground spectral model, you must specify T_obs as a trueval.")
                else:
                    shape_fixedvals = get_robson19_shape_pars_from_tobs(self.injvals['T_obs'])
                    self.truevals |= shape_fixedvals

                self.truevals[r'$\log_{10}A$'] = jnp.log10(self.injvals['A'])
                self.fixedvals = self.truevals

        elif self.spectral_model_name == 'robson19foregroundvaried':
            ## implementation of the Robson+19 analytic foreground model.
            ## this is a variation of the tanh-truncated foreground, but with
            ## additional, time-dependent shape parameters due to subtraction of resolved systems
            ## for the BLIP implementation, it has been recast into Omega_GW space
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha$',r'$\log_{10}A$',r'$\log_{10}f_{\rm knee}$']
            self.omegaf = self.robson19_foreground_varied_spectrum
            self.fancyname = "MW Foreground"+submodel_count
            if not injection:
                self.spectral_prior = self.robson19foregroundvaried_prior
                if 'T_obs' not in self.fixedvals.keys():
                    shape_fixedvals = get_robson19_shape_pars_from_tobs(self.params['dur']/3.154e7)
                    self.fixedvals |= shape_fixedvals
                else:
                    shape_fixedvals = get_robson19_shape_pars_from_tobs(self.fixedvals['T_obs'])
                    self.fixedvals |= shape_fixedvals
            else:
                ## define truevals
                if 'T_obs' not in self.truevals.keys():
                    raise ValueError("When simulated data with the Robson+19 foreground spectral model, you must specify T_obs as a trueval.")
                else:
                    shape_fixedvals = get_robson19_shape_pars_from_tobs(self.truevals['T_obs'])
                    self.truevals |= shape_fixedvals

                self.truevals[r'$\alpha$'] = self.injvals['alpha']
                self.truevals[r'$A$'] = self.injvals['A']
                if 'fknee' in self.injvals.keys():
                    self.truevals[r'$\log_{10}f_{\rm knee}$'] = jnp.log10(self.injvals['fknee'])
                elif 'log_fknee' in self.injvals.keys():
                    self.truevals[r'$\log_{10}f_{\rm knee}$'] = self.injvals['log_fknee']
                self.truevals[r'$\log_{10}A$'] = jnp.log10(self.injvals['A'])
                self.fixedvals = self.truevals


        elif self.spectral_model_name == 'lmcspec':
            ## this is a spectral model tailored to analyses of the LMC SGWB
            # it is a broken power law with alpha_1 = 2/3
            # and astrophysically-motivated prior bounds
            self.spectral_parameters = self.spectral_parameters + [r'$\log_{10} (\Omega_{\rm ref})$',r'$\alpha_2$',r'$\log_{10} (f_{\rm break})$']#,r'$\delta$']
            self.omegaf = self.broken_powerlaw_fixed_a1delta_spectrum
            self.fancyname = "LMC Spectrum"+submodel_count
            if not injection:
                self.fixedvals['alpha_1'] = 2/3
                self.spectral_prior = self.lmcspecbplad_prior
            else:
                raise ValueError("lmcspec is an inference-only spectral submodel. Use the truncatedpowerlaw submodel for injections.")
        elif self.spectral_model_name == 'lmcspecv2':
            ## this is a spectral model tailored to analyses of the LMC SGWB
            # it is a broken power law with both alphas free
            # and astrophysically-motivated prior bounds
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha_1$',r'$\log_{10} (\Omega_{\rm ref})$',r'$\alpha_2$',r'$\log_{10} (f_{\rm break})$']
            self.omegaf = self.broken_powerlaw_spectrum
            self.fancyname = "LMC Spectrum"+submodel_count
            if not injection:
                self.spectral_prior = self.lmcspecfbpl_prior
            else:
                raise ValueError("lmcspecv2 is an inference-only spectral submodel. Use the truncatedpowerlaw submodel for injections.")
        elif self.spectral_model_name == 'sobbhspec':
            ## spectral model tailored to analyses of the SOBBH ISGWB
            ## a fixed alpha=2/3 power law
            ## with astrophysical priors
            self.spectral_parameters = self.spectral_parameters + [r'$\log_{10} (\Omega_{\rm ref})$']
            self.omegaf = self.twothirdspowerlaw_spectrum
            self.fancyname = "SOBBH Power law"+submodel_count
            if not injection:
                self.spectral_prior = self.sobbh_powerlaw_prior
            else:
                raise ValueError("sobbhspec is an inference-only spectral submodel. Use the powerlaw submodel for injections.")
        elif self.spectral_model_name == 'lowpowerlaw':
            ## spectral model to search for a low-amplitude power law
            ## this is the power law model with a constrained upper bound on its amplitude prior
            ## useful when performing spectral separation of an e.g., cosmological background
            ## from the higher-amplitude SOBBH background
            if not injection:
                if hasattr(self,"fixedvals") and 'alpha' in self.fixedvals:
                    self.fancyname = r'$\alpha='+'{}$'.format(self.fixedvals['alpha'])+" Low-Amplitude Power Law"+submodel_count
                    self.omegaf = self.fixedpowerlaw_spectrum
                    self.spectral_parameters = self.spectral_parameters + [r'$\log_{10} (\Omega_{\rm ref})$']
                    self.spectral_prior = self.flatlowpowerlaw_prior
                else:
                    self.fancyname = "Low-Amplitude Power Law"+submodel_count
                    self.omegaf = self.powerlaw_spectrum
                    self.spectral_parameters = self.spectral_parameters + [r'$\alpha$', r'$\log_{10} (\Omega_{\rm ref})$']
                    self.spectral_prior = self.lowpowerlaw_prior
            else:
                raise ValueError("lowpowerlaw is an inference-only spectral submodel. Use the powerlaw submodel for injections.")

        elif self.spectral_model_name == 'population':
            if not injection:
                raise ValueError("Populations are injection-only.")
            popdict = self.inj['popdict'][spec.raw_name]
            if popdict['name'] is not None:
                self.fancyname = popdict['name']
            else:
                self.fancyname = "DWD Population"+submodel_count
            self.population = Population(self.params,self.inj,self.fs,popdict,seed=params['seed'])
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
            ## plotting stuff
            self.fancyname = "Isotropic "+self.fancyname
            self.subscript = r"_{\mathrm{I}}"
            self.color='darkorange'
            self.has_map = False
            self.fullsky = True

            if not injection:
                ## prior transform
                self.prior = self.isotropic_prior
                self.cov = self.compute_cov_isgwb
            else:
                ## Tell the submodel how to handle the injection response matrix when it's computed later on
                self.convolve_inj_response_mat = self.wrapper_convolve_inj_response_mat

        ## This is the spherical harmonic spatial model. It is the workhorse of the spherical harmonic anisotropic analysis.
        ## It can also be used to perform arbitrary injections in the spherical harmonic basis via direct specification of the blms.
        elif self.spatial_model_name == 'sph':

            if injection:
                if self.inj['inj_basis'] == 'pixel':
                    print("Warning: the injection basis has been specified as the pixel basis (inj_basis=pixel), but this is a spherical harmonic injection. \
                          Spherical harmonic injections must use the spherical harmonic basis (inj_basis=sph).\
                          Proceeding with the spherical harmonic basis for this component; other components will continue to use the pixel basis.")
                self.lmax = self.inj['inj_lmax']
            else:
                self.lmax = self.params['lmax']

            ## almax is twice the blmax
            self.almax = 2*self.lmax
            response_kwargs['set_almax'] = self.almax

            ## plotting stuff
            self.fancyname = "Anisotropic "+self.fancyname
            self.subscript = r"_{\mathrm{A}}"
            self.color = 'teal'
            self.has_map = True
            self.fullsky = True
            self.basis = 'sph'

            # add the blms
            blm_parameters = gen_blm_parameters(self.lmax)

            ## save the blm start index for the prior, then add the blms to the parameter list
            self.blm_start = len(self.spectral_parameters)
            self.spatial_parameters = self.spatial_parameters + blm_parameters

            if not injection:
                self.fixed_map = False
                self.parameterized_map = True
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
                self.alms_inj = np.array(self.compute_skymap_alms(val_list).tolist())
                ## get sph basis skymap
                self.sph_skymap =  hp.alm2map(self.alms_inj[0:hp.Alm.getsize(self.almax)],self.params['nside'])
                ## Tell the submodel how to handle the injection response matrix when it's computed later on
                self.convolve_inj_response_mat = self.sph_convolve_inj_response_mat

        ## Handle all the static (non-inferred) astrophysical spatial distributions together due to their similarities
        elif self.spatial_model_name in ['galaxy','mwdisk','mwbulge','dwarfgalaxy','lmc','pointsource','twopoints','pointsources','population','fixedgalaxy','fixedlmc','fixeddisk','fixedbulge','hotpixel','pixiso','popmap']:

            ## some of the astrophysical spatial models are injection-only.
            if self.spatial_model_name in ['galaxy','mwdisk','mwbulge','dwarfgalaxy','lmc','pointsource','twopoints','population'] and not injection:
                raise ValueError("This model is injection-only.")

            self.has_map = True

            if (injection and inj['inj_basis']=='pixel') or (not injection and params['model_basis']=='pixel'):
                basis = 'pixel'
                self.fullsky = False
            else:
                basis = 'sph'
                self.fullsky = True
            self.basis = basis

            ## set lmax for sph case & define responses
            if basis == 'sph':
                ## almax is twice the blmax
                self.lmax = self.inj['inj_lmax']
                self.almax = 2*self.lmax
                response_kwargs['set_almax'] = self.almax

            ## model-specific quantities
            if self.spatial_model_name == 'galaxy':
                ## store the high-level MW truevals for the hierarchical analysis
                self.truevals[r'$r_{\mathrm{h}}$'] = self.injvals['rh']
                self.truevals[r'$z_{\mathrm{h}}$'] = self.injvals['zh']
                ## plotting stuff
                self.fancyname = "Galactic Foreground"
                self.subscript = r"_{\mathrm{G}}"
                self.color = 'mediumorchid'
                ## generate skymap
                self.skymap = astro.generate_galactic_foreground(self.injvals['rh'],self.injvals['zh'],self.params['nside'])
                ## mask to only the first four scale heights
                mask = self.skymap > (1/np.e**4)*np.max(self.skymap)
                self.skymap = self.skymap * mask
            elif self.spatial_model_name == 'mwdisk':
                ## store the high-level MW truevals for the hierarchical analysis
                self.truevals[r'$r_{\mathrm{h}}$'] = self.injvals['rh']
                self.truevals[r'$z_{\mathrm{h}}$'] = self.injvals['zh']
                ## plotting stuff
                self.fancyname = "Galactic Disk"
                self.subscript = r"_{\mathrm{MWD}}"
                self.color = 'mediumorchid'
                ## generate skymap
                self.skymap = astro.generate_galactic_disk(self.injvals['rh'],self.injvals['zh'],self.params['nside'])
                ## mask to only the first four scale heights
                mask = self.skymap > (1/np.e**4)*np.max(self.skymap)
                self.skymap = self.skymap * mask
            elif self.spatial_model_name == 'mwbulge':
                ## plotting stuff
                self.fancyname = "Galactic Bulge"
                self.subscript = r"_{\mathrm{MWB}}"
                self.color = 'teal'
                ## generate skymap
                self.skymap = astro.generate_galactic_bulge(self.params['nside'])
                ## mask to only the first four scale heights
                mask = self.skymap > (1/np.e**4)*np.max(self.skymap)
                self.skymap = self.skymap * mask
            elif self.spatial_model_name == 'lmc':
                ## plotting stuff
                self.fancyname = "LMC"
                self.subscript = r"_{\mathrm{LMC}}"
                self.color = 'darkmagenta'
                ## generate skymap
                self.skymap = astro.generate_sdg(self.params['nside']) ## sdg defaults are for the LMC
            elif self.spatial_model_name == 'dwarfgalaxy':
                ## plotting stuff
                self.fancyname = "Dwarf Galaxy"+submodel_count
                self.subscript = r"_{\mathrm{DG}}"
                self.color = 'maroon'
                ## generate skymap
                self.skymap = astro.generate_sdg(self.params['nside'],ra=self.injvals['sdg_RA'], dec=self.injvals['sdg_DEC'], D=self.injvals['sdg_dist'], r=self.injvals['sdg_rad'], N=self.injvals['sdg_N'])
            elif self.spatial_model_name == 'pointsource':
                ## plotting stuff
                self.fancyname = "Point Source"+submodel_count
                self.subscript = r"_{\mathrm{1P}}"
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
                self.subscript = r"_{\mathrm{NP}}"
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
                self.subscript = r"_{\mathrm{2P}}"
                self.color = 'gold'
                ## generate skymap
                self.skymap = astro.generate_two_point_source(self.injvals['theta_1'],self.injvals['phi_1'],self.injvals['theta_2'],self.injvals['phi_2'],self.params['nside'])
            elif self.spatial_model_name == 'population':
                ## flag the fact that we have a population skymap
                self.skypop = True
                ## plotting stuff
                self.subscript = r"_{\mathrm{P}}"
                self.color = 'midnightblue'
                if self.spectral_model_name != 'population':
                    ## generate population if still needed
                    popdict = self.inj['popdict'][spec.raw_name]
                    if popdict['name'] is not None:
                        self.fancyname = popdict['name']
                    else:
                        self.fancyname = "DWD Population"+submodel_count
                    self.population = Population(self.params,self.inj,self.fs,popdict,seed=params['seed'])
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
                self.subscript = r"_{\mathrm{G}}"
                self.color = 'mediumorchid'
                ## generate skymap
                self.skymap = astro.generate_galactic_foreground(rh,zh,self.params['nside'])
                ## mask to only the first four scale heights
                mask = self.skymap > (1/np.e**4)*np.max(self.skymap)
                self.skymap = self.skymap * mask
                self.fixed_map = True
            elif self.spatial_model_name == 'fixeddisk':
                ## get the fixed values
                if 'rh' in self.fixedvals.keys():
                    rh = self.fixedvals['rh']
                else:
                    print("Warning: Using fixeddisk spatial model but no 'rh' fixed value was provided. Defaulting to Breivik+2020 thin disk galaxy (rh = 2.9 kpc.)")
                    rh = 2.9
                if 'zh' in self.fixedvals.keys():
                    zh = self.fixedvals['zh']
                else:
                    print("Warning: Using fixeddisk spatial model but no 'zh' fixed value was provided. Defaulting to Breivik+2020 thin disk galaxy (zh = 0.3 kpc).")
                    zh = 0.3
                ## plotting stuff
                self.fancyname = "Galactic Disk"
                self.subscript = r"_{\mathrm{MWD}}"
                self.color = 'mediumorchid'
                ## generate skymap
                self.skymap = astro.generate_galactic_disk(rh,zh,self.params['nside'])
                ## mask to only the first four scale heights
                mask = self.skymap > (1/np.e**4)*np.max(self.skymap)
                self.skymap = self.skymap * mask
                self.fixed_map = True
            elif self.spatial_model_name == 'fixedbulge':
                print("Using fixedbulge spatial model with the default bulge values given in Criswell+25a.")
                ## plotting stuff
                self.fancyname = "Galactic Bulge"
                self.subscript = r"_{\mathrm{MWB}}"
                self.color = 'teal'
                ## generate skymap
                self.skymap = astro.generate_galactic_bulge(self.params['nside'])
                ## mask to only the first four scale heights
                mask = self.skymap > (1/np.e**4)*np.max(self.skymap)
                self.skymap = self.skymap * mask
                self.fixed_map = True
            elif self.spatial_model_name == 'fixedlmc':
                ## plotting stuff
                self.fancyname = "LMC"
                self.subscript = r"_{\mathrm{LMC}}"
                self.color = 'darkmagenta'
                ## generate skymap
                self.skymap = astro.generate_sdg(self.params['nside']) ## sdg defaults are for the LMC
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
                self.subscript = r"_{\mathrm{1P}}"
                self.color = 'forestgreen'
                self.skymap = astro.generate_point_source(coord1,coord2,self.params['nside'],convention=convention,pad=True)
                self.fixed_map = True

            elif self.spatial_model_name == 'pixiso':
                self.fancyname = "Pixel Isotropic"
                self.subscript = r"_{\mathrm{PI}}"
                self.color = 'forestgreen'
                self.skymap = np.ones(hp.nside2npix(self.params['nside']))
                self.fixed_map = True

            elif self.spatial_model_name == 'popmap':
                self.fancyname = "Population Skymap"
                self.subscript = r"_{\mathrm{PM}}"
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

            ## set skymap
            if basis == 'pixel':
                response_kwargs['skymap_inj'] = self.skymap #/(np.sum(self.skymap)*hp.nside2pixarea(self.params['nside']))


            ## process skymap, indicate how to compute the response functions later
            if not injection:
                if basis == 'sph':
                    self.process_astro_skymap_model(self.skymap)
                    self.prior = self.fixedsky_prior
                    self.cov = self.compute_cov_fixed_asgwb
                elif basis=='pixel':
                    self.prior = self.fixedsky_prior
                    self.cov = self.compute_cov_fixed_asgwb
                else:
                    raise TypeError("Basis was not defined, or was incorrectly defined.")
            else:
                if basis == 'sph':
                    self.process_astro_skymap_injection(self.skymap)
                    ## Tell the submodel how to handle the injection response matrix when it's computed later on
                    self.convolve_inj_response_mat = self.sph_convolve_inj_response_mat
                elif basis == 'pixel':
                    ## Tell the submodel how to handle the injection response matrix when it's computed later on
                    self.convolve_inj_response_mat = self.wrapper_convolve_inj_response_mat
                else:
                    raise TypeError("Basis was not defined, or was incorrectly defined.")


        ## Parameterized astrophysical spatial distributions.
        ## Distinct from the fixedsky/injection-only models as we need spatial inference infrastructure
        ## pixel-basis only
        elif self.spatial_model_name in ['1parametermw','2parametermw']:

            ## enforce pixel basis
            if params["model_basis"] != "pixel":
                raise ValueError("Parameterized astrophysical spatial submodels are only supported in the pixel basis. (You have set basis={}.)".format(params["model_basis"]))
            self.basis = "pixel"

            ## calculate pixel area
            self.dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])

            ## set starting index for spatial model parameters
            self.spatial_start = len(self.spectral_parameters)

            ## we won't convolve this response function with anything ahead of time, so set the wrapper
            self.convolve_inj_response_mat = self.wrapper_convolve_inj_response_mat

            ## 2-parameter Milky Way model
            if self.spatial_model_name == '1parametermw':
                ## model to infer the Milky Way spatial distribution, using a simplified 1-parameter model of the Galaxy
                ## only infers the vertical scale height z_h
                ## plotting stuff
                self.fancyname = "1-Parameter Milky Way"
                self.subscript = r"_{\mathrm{G}}"
                self.color = 'mediumorchid'
                self.has_map = True
                self.fixed_map = False
                self.parameterized_map = True

                ## Initialize the galaxy grid
                self.galaxy = astro.Galaxy_Model(self.params['nside'],grid_res=0.1,
                                                 gal_rad=14,gal_height=6,max_rh=3.5,max_zh=1.5,fix_rh=self.fixedvals['rh'])
                self.max_sky_extent = self.galaxy.max_skymap

                ## Set the parameterized spatial model function
                self.compute_skymap = self.galaxy.mw_mapmaker_1par

                ## mask maps to maximum allowed spatial extent
                self.mask = self.galaxy.max_skymap > (1/np.e**4)*np.max(self.galaxy.max_skymap)
                self.masked_skymap = self.galaxy.max_skymap * self.mask
                self.mask_idx = np.flatnonzero(self.mask)

                ## ensure normalization
                self.masked_skymap = self.masked_skymap/(np.sum(self.masked_skymap)*self.dOmega)

                ## alias as needed for response function calculations
                self.skymap = self.masked_skymap

                ## set response kwargs
                response_kwargs['masked_skymap'] = self.masked_skymap


                self.spatial_parameters = [r'$z_{\mathrm{h}}$']
                self.prior = self.mw1parameter_prior
                self.cov = self.compute_cov_parameterized_asgwb

            elif self.spatial_model_name == '2parametermw':
                ## model to infer the Milky Way spatial distribution, using a basic 2-parameter model of the Galaxy
                ## plotting stuff
                self.fancyname = "2-Parameter Milky Way"
                self.subscript = r"_{\mathrm{G}}"
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

                ## alias as needed for response function calculations
                self.skymap = self.masked_skymap

                ## set response kwargs
                response_kwargs['masked_skymap'] = self.masked_skymap

                self.spatial_parameters = [r'$r_{\mathrm{h}}$',r'$z_{\mathrm{h}}$']
                self.prior = self.mw2parameter_prior
                self.cov = self.compute_cov_parameterized_asgwb

            else:
                raise ValueError("Parameterized astrophysical spatial submodel type not found. Did you add a new model to the list at the top of this section?")

        else:
            raise ValueError("Invalid specification of spatial model name ('{}').".format(self.spatial_model_name))

        #############################
        ##      FOR TESTING        ##
        #############################
#        np.save(self.params['out_dir']+'/response_'+spec.raw_name+'.npy',self.response_mat) 
#        print("Saving response array to "+self.params['out_dir']+'/response_'+spec.raw_name+'.npy')
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
                updated_truevals = {suffix+' '+parameter:self.truevals[parameter] for parameter in self.parameters}
                self.truevals = updated_truevals
            updated_spectral_parameters = [suffix+' '+parameter for parameter in self.spectral_parameters]
            updated_spatial_parameters = [suffix+' '+parameter for parameter in self.spatial_parameters]
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

    def fixedpowerlaw_spectrum(self,fs,log_omega0):
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
        return 10**(log_omega0)*(fs/self.params['fref'])**(self.fixedvals['alpha'])

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

    def broken_powerlaw_fixed_a1delta_spectrum(self,fs,log_omega0,alpha_2,log_fbreak):
        '''
        Function to calculate a broken power law spectrum, with a fixed low-frequency slope and turnover scale.

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
        delta = 0.1
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

    def truncated_doublebroken_powerlaw_spectrum(self,fs,alpha_1, alpha_2, alpha_3, delta1, delta2, log_omega0, log_fcut, log_fscale, log_fbreak1, log_fbreak2):
            '''
            Function to calculate a tanh-truncated power law spectrum.
            Arguments
            -----------
            fs (array of floats) : frequencies at which to evaluate the spectrum
            alpha_1 (float)      : slope of the first power law
            alpha_2 (float)      : slope of the second power law
            alpha_3 (float)      : slope of the third power law
            delta1 (float)       : smoothing parameter between the first and second powerlaws.
            delta2 (float)       : smoothing parameter between the second and third powerlaws.
            log_omega0 (float)   : power law amplitude of the power law in units of log dimensionless GW energy density at f_ref (if left un-truncated)
            log_fcut (float)     : log of the cut frequency ("knee") in Hz
            log_fscale           : log of the cutoff scale factor in Hz
            log_fbreak1 (float)  : log of the first break frequency ("knee") in Hz
            log_fbreak2 (float)  : log of the second break frequency ("knee") in Hz
            Returns
            -----------
            spectrum (array of floats) : the resulting truncated power law spectrum
            '''
            fcut = 10**log_fcut
            fscale = 10**log_fscale
            fbreak1 = 10**log_fbreak1
            fbreak2 = 10**log_fbreak2
            norm = (fbreak1/self.params['fref'])**alpha_1 ## this normalizes the broken powerlaw such that its first leg matches the equivalent standard power law

            brokenpl1 = (1+(fs/fbreak1)**(1/delta1))**((alpha_1-alpha_2)*delta1)
            brokenpl2 = (1+(fs/fbreak2)**(1/delta2))**((alpha_2-alpha_3)*delta2)
            tanhtrunc = (1+jnp.tanh((fcut-fs)/fscale))


            return 0.5 * norm * (10**log_omega0)*(fs/fbreak1)**(alpha_1) * tanhtrunc * brokenpl1 * brokenpl2

    def robson19_foreground_spectrum(self,fs,logA):
        '''
        Function to calculate an analytical spectrum for the Galactic foreground of the form given in Robson et al. (2019) (arXiv:1803.01944)

        NOTE: this is given in terms of PSD amplitude A, as opposed to the usual units used in BLIP (dimensionless GW energy density)

        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        A (float)   : power law amplitude of the power law in units of **PSD** at f_ref (if left un-truncated)

        Returns
        -----------
        spectrum (array of floats) : the resulting analytical foreground spectrum

        '''

        alpha_shape = self.fixedvals[r'$\alpha_{\rm shape}$']
        beta_shape = self.fixedvals[r'$\beta$']
        kappa = self.fixedvals[r'$\kappa$']
        gamma = self.fixedvals[r'$\gamma$']
        log_fknee = jnp.log10(self.fixedvals[r'$f_{\rm knee}$'])


        Sgw = 10**logA * (fs/self.params['fref'])**(-7/3) * jnp.exp(-fs**alpha_shape + beta_shape*fs*jnp.sin(kappa*fs)) * (1 + jnp.tanh(gamma*(10**log_fknee - fs)))
        ## defined in terms of Sgw, so need to convert to be in terms of Omegaf
        return self.compute_Omega0_from_Sgw(fs,Sgw)

    def robson19_foreground_varied_spectrum(self,fs,alpha,logA,log_fknee):
        '''
        Function to calculate an analytical spectrum for the Galactic foreground of the form given in Robson et al. (2019) (arXiv:1803.01944)

        This version also varies the slope alpha and break ("knee") frequency f_knee.

        NOTE: this is given in terms of PSD amplitude A, as opposed to the usual units used in BLIP (dimensionless GW energy density)

        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        A (float)   : power law amplitude of the power law in units of **PSD** at f_ref (if left un-truncated)
        alpha (float) : power law slope above the truncation
        log_fknee (float) : Log10 of the break ("knee") frequency.

        Returns
        -----------
        spectrum (array of floats) : the resulting analytical foreground spectrum

        '''

        alpha_shape = self.fixedvals[r'$\alpha_{\rm shape}$']
        beta_shape = self.fixedvals[r'$\beta$']
        kappa = self.fixedvals[r'$\kappa$']
        gamma = self.fixedvals[r'$\gamma$']

        Sgw = 10**logA * (fs/self.params['fref'])**(alpha) * jnp.exp(-fs**alpha_shape + beta_shape*fs*jnp.sin(kappa*fs)) * (1 + jnp.tanh(gamma*(10**log_fknee - fs)))
        ## defined in terms of Sgw, so need to convert to be in terms of Omegaf
        return self.compute_Omega0_from_Sgw(fs,Sgw)

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

    def compute_Sgw(self,fs,omegaf_args):  # pylint: disable=method-hidden
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

    def compute_Omega0_from_Sgw(self,fs,Sgw):
        '''
        Wrapper function to generically calculate the associated stochastic gravitational wave dimensionless GW energy density Omega(f)
            for a spectral model given in terms of the PSD (S_gw)

        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        sgw_args (list)   : list of arguments for the relevant Omega(f) function

        Returns
        -----------
        Sgw (array of floats) : the resulting GW PSD

        '''
        H0 = 2.2*10**(-18) ## Hubble constant, = 67.88 km/s/Mpc
        Omegaf = Sgw*((3/(4*fs**3))*(H0/jnp.pi)**2)**-1
        return Omegaf

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
        log_omega0  = -30*theta[1] + 9

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
        log_omega0  = -30*theta[0] + 9

        return [log_omega0]

    def sobbh_powerlaw_prior(self,theta):


        '''
        Prior function for a power law with fixed slope, with astrophysical prior bounds tailored to the expected SOBBH ISGWB amplitude (see, e.g., Babak+2023)

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
        log_omega0  = -1.5*theta[0] - 11

        return [log_omega0]

    def lowpowerlaw_prior(self,theta):


        '''
        Prior function for an isotropic stochastic backgound analysis. Imposes an upper constraint on the prior to aid in spectral separation.

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
        alpha       =  6*theta[0] - 3
        log_omega0  = -11*theta[1] - 12.6

        return [alpha, log_omega0]

    def flatlowpowerlaw_prior(self,theta):


        '''
        Prior function for an isotropic stochastic backgound analysis. Imposes an upper constraint on the prior to aid in spectral separation.

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
        log_omega0  = -11*theta[0] - 12.6

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
            theta with each element rescaled. The elements are  interpreted as alpha_1, log(Omega_ref), alpha_2, and log(f_break).

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        alpha_1 = 10*theta[0] - 4
        log_omega0 = -22*theta[1]
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
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_ref), log(f_cut), log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors

        log_omega0 = -22*theta[0]
        alpha_2 = 4*theta[1] + self.fixedvals['alpha_1'] ## must be greater than alpha_1
        log_fbreak = -2*theta[2] - 2
        delta = 0.99*theta[3] + 0.01

        return [log_omega0, alpha_2, log_fbreak, delta]

    def broken_powerlaw_fixed_a1delta_prior(self,theta):


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
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_ref), log(f_cut), log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors

        log_omega0 = -22*theta[0]
        alpha_2 = 4*theta[1] #+ self.fixedvals['alpha_1'] ## must be greater than alpha_1
        log_fbreak = -2*theta[2] - 2

        return [log_omega0, alpha_2, log_fbreak]

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
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_ref), log(f_cut), and log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        alpha = 10*theta[0] - 5
        log_omega0 = -22*theta[1] + 5
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
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_ref), and log(f_cut)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        alpha = 10*theta[0] - 5
        log_omega0 = -22*theta[1] + 5
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
            theta with each element rescaled. The elements are  interpreted as log(Omega_ref) and log(f_cut)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        log_omega0 = -22*theta[0] + 5
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
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_ref), and log(f_cut)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        log_omega0 = -3*theta[0] - 7
        log_fcut = -0.7*theta[1] - 2.4
        log_fscale = -2*theta[2] - 2


        return [log_omega0, log_fcut, log_fscale]

    def mwspec3par_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a 3-parameter truncated power law spectral model.

        Bounds are astrophysically-motivated and tailored to expectations of the MW foreground.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_ref), log(f_cut), and log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        alpha = 2*theta[0]
        log_omega0 = -3*theta[1] - 7
        log_fcut = -0.7*theta[2] - 2.4


        return [alpha, log_omega0, log_fcut]

    def mwspec4par_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a 4-parameter truncated power law spectral model.

        Bounds are astrophysically-motivated and tailored to expectations of the MW foreground.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_ref), log(f_cut), and log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        alpha = 2*theta[0]
        log_omega0 = -3*theta[1] - 7
        log_fcut = -0.7*theta[2] - 2.4
        log_fscale = -2*theta[3] - 2


        return [alpha, log_omega0, log_fcut, log_fscale]

    def truncated_doublebroken_powerlaw_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a truncated doublebroken power law spectral model.
        Parameters
        -----------
        theta   : float
            A list or numpy array containing samples from a unit cube.
        Returns
        ---------
        theta   :   float
            theta with each element rescaled. 
            The elements are  interpreted as alpha_1, alpha_2, alpha_3, delta1, delta2, log_omega0, log_fcut, log_fscale, log_fbreak1, log_fbreak2
        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        alpha_1 = 10*theta[0] - 4
        alpha_2 = 40*theta[1]
        alpha_3 = (34+44)*theta[2] - 44 # LSS [34, 44] so alpha2-alpha3 is [-44, 6], equivalent to alpha1-alpha2 range.
        delta1 = 0.99*theta[3] + 0.01
        delta2 = 0.99*theta[4] + 0.01
        log_omega0 = -22*theta[5]
        # LSS  .8-10 mHz in log10 is -3.0969100130080562 to -2.0
        log_fcut = -(1.0969100130080562*theta[6] + 2) 
        log_fscale = -2*theta[7] - 2
        # LSS .6-.8mHz in log10 is -3.2218487496163566 to -3.0969100130080562
        log_fbreak1 = (-3.0969100130080562+3.2218487496163566)*theta[8] - 3.2218487496163566 
        # LSS .2-.599 mHz in log10 is -3.6989700043360187 to -3.2225731776106885
        log_fbreak2 = (-3.2225731776106885+3.6989700043360187)*theta[9] - 3.6989700043360187 

        return [alpha_1, alpha_2, alpha_3, delta1, delta2, log_omega0, log_fcut, log_fscale, log_fbreak1, log_fbreak2]

    def bdspec_prior(self,theta):
        """Prior for spectrum of bulge or disk of MW (bdspec).

        Parameters
        ----------
        theta : array (3,)
            samples from unit cube

        Returns
        -------
        theta
            rescaled samples in order: log(Omega_ref), log(f_cut), and log(f_scale)
        """
        assert len(theta) == 3
        log_omega0 = -3*theta[0] - 8
        log_fcut = -0.7*theta[1] - 2.4
        log_fscale = -2*theta[2] - 2
        return [log_omega0, log_fcut, log_fscale]

    def robson19foreground_prior(self,theta):

        logA = -20*theta[0] - 30

        return [logA]

    def robson19foregroundvaried_prior(self,theta):

        alpha = 3*theta[0] - 3
        logA = -20*theta[1] - 30
        log_fknee = -0.5*theta[2] - 2.75

        return [alpha,logA,log_fknee]

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
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_ref), log(f_cut), log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        log_omega0 = -3*theta[0] - 9
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
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_ref), log(f_cut), log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors

        log_omega0 = -3*theta[0] - 9
        alpha_2 = 4*theta[1] #+ self.fixedvals['alpha_1'] ## must be greater than alpha_1
        log_fbreak = -1*theta[2] - 2
        delta = 0.99*theta[3] + 0.01

        return [log_omega0, alpha_2, log_fbreak, delta]

    def lmcspecbplad_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a 3-parameter broken power law spectral model.
        Tailored for the LMC DWD SGWB spectrum.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_ref), log(f_cut), log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors

        log_omega0 = -3*theta[0] - 9
        alpha_2 = 4*theta[1] #+ self.fixedvals['alpha_1'] ## must be greater than alpha_1
        log_fbreak = -1*theta[2] - 2

        return [log_omega0, alpha_2, log_fbreak]

    def lmcspecfbpl_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a 4-parameter broken power law spectral model.
        Tailored for the LMC DWD SGWB spectrum. In contrast to lmcspecbpl, this variant fixes the smoothing parameter delta and allows alpha_1 to vary.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_ref), log(f_cut), log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors

        alpha_1 = 2*theta[0]
        log_omega0 = -3*theta[1] - 9
        alpha_2 = 4*theta[2]
        log_fbreak = -1*theta[2] - 2

        return [alpha_1,log_omega0,alpha_2,log_fbreak]

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
        cov_sgwb = Sgw[None, None, :, None]*self.response_mat

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

    def wrapper_convolve_inj_response_mat(self,fdata_flag=False):
        '''
        A wrapper function for the ISGWB and pixel basis cases, the skymaps are convolved implicitly when calculating the response.

        Arguments
        -----------
        fdata_flag (bool) : Whether to compute the convolution for injection frequencies (False, default) or data frequencies (True).

        Returns
        -----------
        (none)

        '''

        # create a wrapper b/c isotropic and anisotropic injection responses are handled differently w.r.t. skymap convolution
        if not fdata_flag:
            self.inj_response_mat = self.response_mat
            self.summ_response_mat = self.response_mat
        else:
            self.fdata_response_mat = self.unconvolved_fdata_response_mat

        return

    def sph_convolve_inj_response_mat(self,fdata_flag=False):
        '''
        Function to convolve the sph response matrix with an injected spherical harmonic skymap.

        Arguments
        -----------
        fdata_flag (bool) : Whether to compute the convolution for injection frequencies (False, default) or data frequencies (True).

        Returns
        -----------
        (none)

        '''

        if not fdata_flag:
            ## get response integrated over the Ylms
            self.summ_response_mat = self.compute_summed_response(self.alms_inj)
            ## create a wrapper b/c isotropic and anisotropic injection responses are different
            self.inj_response_mat = self.summ_response_mat
        else:
            self.fdata_response_mat = jnp.einsum('ijklm,m', self.unconvolved_fdata_response_mat, self.alms_inj)

        return

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
        ## sacrifice einsum efficiency for memory usage here
        ## due to the extreme memory requirement of the pixel-basis unconvolved anisotropic responses
        convolved_response = jnp.zeros(self.response_mat.shape[:-1])
        for ii in range(len(pixelmap)):
            convolved_response = convolved_response + self.response_mat[:,:,:,:,ii]*pixelmap[ii]
        return self.dOmega*convolved_response
    #        return (self.dOmega)*jnp.einsum('ijklm,m', self.response_mat, pixelmap)

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
#        self.summ_response_mat = self.compute_summed_response(self.alms_inj)
#        ## create a wrapper b/c isotropic and anisotropic injection responses are different
#        self.inj_response_mat = self.summ_response_mat

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
