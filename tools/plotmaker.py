import sys, os
sys.path.append(os.getcwd()) ## this lets python find src
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from chainconsumer import ChainConsumer
import healpy as hp
from healpy import Alm
from astropy import units as u
import pickle, argparse
import logging
from src.populations import populations
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def mapmaker(post, params, parameters, inj, Model, Injection, saveto=None, coord=None):
    
    
    sph_models = []
    hierarchical_models = []
    for submodel_name in Model.submodel_names:
        ## spatial type will be the latter part of the name
        ## also catch duplicates (with -N appended to them_)
        spatial_name = submodel_name.split('_')[-1].split('-')[0]
        if spatial_name == 'sph':
            sph_models.append(submodel_name)
        elif spatial_name == 'hierarchical':
            hierarchical_models.append(submodel_name)
    if (len(sph_models)==0 ) and (len(hierarchical_models)==0):
        print("Called mapmaker but none of the recovery models have a non-isotropic spatial model. Skipping...")
        return
    
    
    ## FIX THE REST LATER
    
#    for name in sph_models+hierarchical_models:
#        if name in sph_models:
#            blm_start = Model[name].blm_start
#        elif name in hierarchical_models:
#            hyperparam_start = 
    
    nside = params['nside']

    npix = hp.nside2npix(nside)

        
    start_idx = 0   
    for submodel_name in Model.submodel_names:
        ## grab submodel
        sm = Model.submodels[submodel_name]
        
        # Initialize power skymap
        omega_map = np.zeros(npix)
        
        ## only make a map if there's a map to make (this is also good life advice)
        if submodel_name in sph_models+hierarchical_models:
            ## select relevant posterior columns
            post_i = post[:,start_idx:(start_idx+sm.Npar)]
            
            print("Computing marginalized posterior skymap for submodel: {}...".format(submodel_name))
            for ii in range(post.shape[0]):
                ## get Omega(f=1mHz)
                Omega_1mHz = sm.omegaf(1e-3,*post_i[ii,:sm.blm_start])
                
                ## convert blm params to full blms
                blm_vals = sm.blm_params_2_blms(post_i[ii,sm.blm_start:])
                
                ## normalize, convert to map, and sum
                norm = np.sum(blm_vals[0:(sm.lmax + 1)]**2) + np.sum(2*np.abs(blm_vals[(sm.lmax + 1):])**2)
                
                prob_map  = (1.0/norm) * (hp.alm2map(blm_vals, nside))**2
                
                omega_map = omega_map + Omega_1mHz * prob_map

            omega_map = omega_map/post.shape[0]
            
            # setting coord back to E, if parameter isn't specified
            if 'projection' in params.keys():
                coord = params['projection']
            else:
                coord = 'E'
            
            ## HEALpy is really, REALLY noisy sometimes. This stops that.
            logger = logging.getLogger()
            logger.setLevel(logging.ERROR)
            
            # generating skymap, switches to specified projection if not 'E'
            if coord=='E':
                hp.mollview(omega_map, coord=coord, title='Marginalized posterior skymap of $\\Omega(f= 1mHz)$', unit="$\\Omega(f= 1mHz)$")
            else:
                hp.mollview(omega_map, coord=['E',coord], title='Marginalized posterior skymap of $\\Omega(f= 1mHz)$', unit="$\\Omega(f= 1mHz)$")
           
            hp.graticule()
            
            ## switch logging level back to normal so we get our own status updates
            logger.setLevel(logging.INFO)
            
            if saveto is not None:
                plt.savefig(saveto + '/{}_post_skymap.png'.format(submodel_name), dpi=150)
                logger.info('Saving posterior skymap at ' +  saveto + '/{}_post_skymap.png'.format(submodel_name))
        
            else:
                plt.savefig(params['out_dir'] + '/{}_post_skymap.png'.format(submodel_name), dpi=150)
                logger.info('Saving posterior skymap at ' +  params['out_dir'] + '/{}_post_skymap.png'.format(submodel_name))
            plt.close()
            
            ## now do the median skymap
            print("Computing median posterior skymap for submodel {}...".format(submodel_name))
            # median values of the posteriors
            med_vals = np.median(post_i, axis=0)
            
            # Omega(f=1mHz)
            Omega_1mHz_median = sm.omegaf(1e-3,*med_vals[:sm.blm_start])
            ## blms.
            blms_median = np.append([1], med_vals[sm.blm_start:])
            
            blm_median_vals = sm.blm_params_2_blms(blms_median)
        
            norm = np.sum(blm_median_vals[0:(sm.lmax + 1)]**2) + np.sum(2*np.abs(blm_median_vals[(sm.lmax + 1):])**2)

            Omega_median_map  =  Omega_1mHz_median * (1.0/norm) * (hp.alm2map(blm_median_vals, nside))**2
            
            ## HEALpy is really, REALLY noisy sometimes. This stops that.
            logger.setLevel(logging.ERROR)
            
            if coord=='E':
                hp.mollview(Omega_median_map, coord=coord, title='Median skymap of $\\Omega(f= 1mHz)$', unit="$\\Omega(f= 1mHz)$")
            else:
                hp.mollview(Omega_median_map, coord=['E',coord], title='Median skymap of $\\Omega(f= 1mHz)$', unit="$\\Omega(f= 1mHz)$")
            
            hp.graticule()
            
            ## switch logging level back to normal so we get our own status updates
            logger.setLevel(logging.INFO)
            
            if saveto is not None:
                plt.savefig(saveto + '/post_median_skymap.png', dpi=150)
                logger.info('Saving injected skymap at ' +  saveto + '/post_median_skymap.png')
        
            else:
                plt.savefig(params['out_dir'] + '/post_median_skymap.png', dpi=150)
                logger.info('Saving injected skymap at ' +  params['out_dir'] + '/post_median_skymap.png')
        
            plt.close()
        
            
        
        ## increment start regardless of if we made a map
        start_idx += sm.Npar
    
    
    return
    
    
    
    
    
    
    
#    
#    
#    
#    
#    
#    
#    blm_start = len(Model.parameters['all']) - len(Model.parameters['blms'])
#    
#    if type(parameters) is dict:
#        blm_start = len(parameters['noise']) + len(parameters['signal'])
#        
#    elif type(parameters) is list:
#        print("Warning: using a depreciated parameter format. Number of non-b_lm parameters is unknown, defaulting to n=4.")
#        blm_start = 4
#    else:
#        raise TypeError("parameters argument is not dict or list.")
#    
#    # size of the blm array
#    blm_size = Alm.getsize(params['lmax'])
#
#    ## we will plot with a larger nside than the analysis for finer plots
##    nside = 2*params['nside']
#    ## no we won't
#    nside = params['nside']
#
#    npix = hp.nside2npix(nside)
#
#    # Initialize power skymap
#    omega_map = np.zeros(npix)
#
#    blmax = params['lmax']
#    
#    print("Computing marginalized posterior skymap...")
#    for ii in range(post.shape[0]):
#
#        sample = post[ii, :]
#
#        # Omega at 1 mHz
#        # handle various spectral models, but default to power law
#        if 'spectrum_model' in params.keys():
#            if params['spectrum_model'] == 'powerlaw':
#                alpha = sample[2]
#                log_Omega0 = sample[3]
#                Omega_1mHz = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)
#            elif params['spectrum_model']=='broken_powerlaw':
#                log_A1 = sample[2]
#                alpha_1 = sample[3]
#                log_A2 = sample[4]
#                alpha_2 = sample[3] - 0.667
#                Omega_1mHz= ((10**log_A1)*(1e-3/params['fref'])**alpha_1)/(1 + (10**log_A2)*(1e-3/params['fref'])**alpha_2)
#            elif params['spectrum_model']=='broken_powerlaw_2':
#                delta = 0.1
#                log_Omega0 = sample[2]
#                alpha_1 = sample[3]
#                alpha_2 = sample[4]
#                f_break = 10**sample[5]
#                Omega_1mHz = (10**log_Omega0)*(1e-3/f_break)**(alpha_1) * (0.5*(1+(1e-3/f_break)**(1/delta)))**((alpha_1-alpha_2)*delta)
#            elif params['spectrum_model']=='free_broken_powerlaw':
#                log_A1 = sample[2]
#                alpha_1 = sample[3]
#                log_A2 = sample[4]
#                alpha_2 = sample[5]
#                Omega_1mHz= ((10**log_A1)*(1e-3/params['fref'])**alpha_1)/(1 + (10**log_A2)*(1e-3/params['fref'])**alpha_2)
#            elif params['spectrum_model']=='truncated_broken_powerlaw':
#                delta = 0.1
#                log_Omega0 = sample[2]
#                alpha_1 = sample[3]
#                alpha_2 = sample[4]
#                f_break = 10**sample[5]
#                f_cut = f_break
#                f_scale = 10**sample[6]
#                Omega_1mHz = 0.5 * (10**log_Omega0)*(1e-3/f_break)**(alpha_1) * (0.5*(1+(1e-3/f_break)**(1/delta)))**((alpha_1-alpha_2)*delta) * (1+np.tanh((f_cut-1e-3)/f_scale))
#            elif params['spectrum_model']=='truncated_powerlaw':
#                delta = 0.1
#                log_Omega0 = sample[2]
#                alpha = sample[3]
#                f_break = 10**sample[4]
#                f_scale = 10**sample[5]
#                Omega_1mHz = 0.5 * (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha) * (1+np.tanh((f_break-1e-3)/f_scale))
#            elif params['spectrum_model']=='multi_atpl_ipl':
#                delta = 0.1
#                log_Omega0_a = sample[2]
#                alpha_a = sample[3]
#                f_cut_a = 10**sample[4]
#                f_scale_a = 10**sample[5]
##                log_Omega0_i = sample[6]
##                alpha_i = sample[7]
#                Omega_1mHz = 0.5 * (10**(log_Omega0_a)) * (1e-3/params['fref'])**(alpha_a) * (1+np.tanh((f_cut_a-1e-3)/f_scale_a))
#            else:
#                if ii==0:
#                    print("Unknown spectral model. Defaulting to power law...")
#                alpha = sample[2]
#                log_Omega0 = sample[3]
#                Omega_1mHz = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)
#        else:
#            print("Warning: running on older output without specification of spectral model.")
#            print("Warning: defaulting to power law spectral model. This may result in unintended behavior.")
#            alpha = sample[2]
#            log_Omega0 = sample[3]
#            Omega_1mHz = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)
#        ## blms.
#        blms = np.append([1], sample[blm_start:])
#
#        ## Complex array of blm values for both +ve m values
#        blm_vals = np.zeros(blm_size, dtype='complex')
#
#        ## this is b00, alsways set to 1
#        blm_vals[0] = 1
#        norm, cnt = 1, 1
#
#        for lval in range(1, blmax + 1):
#            for mval in range(lval + 1):
#
#                idx = Alm.getidx(blmax, lval, mval)
#
#                if mval == 0:
#                    blm_vals[idx] = blms[cnt]
#                    cnt = cnt + 1
#                else:
#                    ## prior on amplitude, phase
#                    blm_vals[idx] = blms[cnt] * np.exp(1j * blms[cnt+1])
#                    cnt = cnt + 2
#
#        norm = np.sum(blm_vals[0:(blmax + 1)]**2) + np.sum(2*np.abs(blm_vals[(blmax + 1):])**2)
#
#        prob_map  = (1.0/norm) * (hp.alm2map(blm_vals, nside))**2
#
#        ## add to the omega map
#        omega_map = omega_map + Omega_1mHz * prob_map
#
#    omega_map = omega_map/post.shape[0]
#
#    # setting coord back to E, if parameter isn't specified
#    if 'projection' in params.keys():
#        coord = params['projection']
#    else:
#        coord = 'E'
#    
#    ## HEALpy is really, REALLY noisy sometimes. This stops that.
#    logger = logging.getLogger()
#    logger.setLevel(logging.ERROR)
#    
#    # generating skymap, switches to specified projection if not 'E'
#    if coord=='E':
#        hp.mollview(omega_map, coord=coord, title='Marginalized posterior skymap of $\\Omega(f= 1mHz)$', unit="$\\Omega(f= 1mHz)$")
#    else:
#        hp.mollview(omega_map, coord=['E',coord], title='Marginalized posterior skymap of $\\Omega(f= 1mHz)$', unit="$\\Omega(f= 1mHz)$")
#   
#    # hp.mollview(omega_map, coord=coord, title='Posterior predictive skymap of $\\Omega(f= 1mHz)$')
#
#    hp.graticule()
#    
#    ## switch logging level back to normal so we get our own status updates
#    logger.setLevel(logging.INFO)
#    
#    if saveto is not None:
#        plt.savefig(saveto + '/post_skymap.png', dpi=150)
#        logger.info('Saving posterior skymap at ' +  saveto + '/post_skymap.png')
#
#    else:
#        plt.savefig(params['out_dir'] + '/post_skymap.png', dpi=150)
#        logger.info('Saving posterior skymap at ' +  params['out_dir'] + '/post_skymap.png')
#    plt.close()
#
#
#    #### ------------ Now plot median value
#    print("Computing median posterior skymap...")
#    # median values of the posteriors
#    med_vals = np.median(post, axis=0)
#
#    ## blms.
#    blms_median = np.append([1], med_vals[blm_start:])
#
#    # Omega at 1 mHz
#    # handle various spectral models, but default to power law
#    ## include backwards compatability check (to be depreciated later)
#    if 'spectrum_model' in params.keys():
#        if params['spectrum_model'] == 'powerlaw':
#            alpha = med_vals[2]
#            log_Omega0 = med_vals[3]
#            Omega_1mHz_median = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)
#        elif params['spectrum_model']=='broken_powerlaw':
#            log_A1 = med_vals[2]
#            alpha_1 = med_vals[3]
#            log_A2 = med_vals[4]
#            alpha_2 = med_vals[3] - 0.667
#            Omega_1mHz_median= ((10**log_A1)*(1e-3/params['fref'])**alpha_1)/(1 + (10**log_A2)*(1e-3/params['fref'])**alpha_2)
#        elif params['spectrum_model']=='broken_powerlaw_2':
#            delta = 0.1
#            log_Omega0 = med_vals[2]
#            alpha_1 = med_vals[3]
#            alpha_2 = med_vals[4]
#            f_break = 10**med_vals[5]
#            Omega_1mHz_median = (10**log_Omega0)*(1e-3/f_break)**(alpha_1) * (0.5*(1+(1e-3/f_break)**(1/delta)))**((alpha_1-alpha_2)*delta)
#        elif params['spectrum_model']=='free_broken_powerlaw':
#            log_A1 = med_vals[2]
#            alpha_1 = med_vals[3]
#            log_A2 = med_vals[4]
#            alpha_2 = med_vals[5]
#            Omega_1mHz_median= ((10**log_A1)*(1e-3/params['fref'])**alpha_1)/(1 + (10**log_A2)*(1e-3/params['fref'])**alpha_2)
#        elif params['spectrum_model']=='truncated_broken_powerlaw':
#            delta = 0.1
#            log_Omega0 = med_vals[2]
#            alpha_1 = med_vals[3]
#            alpha_2 = med_vals[4]
#            f_break = 10**med_vals[5]
#            f_cut = f_break
#            f_scale = 10**med_vals[6]
#            Omega_1mHz_median = 0.5 * (10**log_Omega0)*(1e-3/f_break)**(alpha_1) * (0.5*(1+(1e-3/f_break)**(1/delta)))**((alpha_1-alpha_2)*delta) * (1+np.tanh((f_cut-1e-3)/f_scale))
#        elif params['spectrum_model']=='truncated_powerlaw':
#            delta = 0.1
#            log_Omega0 = med_vals[2]
#            alpha = med_vals[3]
#            f_break = 10**med_vals[4]
#            f_scale = 10**med_vals[5]
#            Omega_1mHz_median = 0.5 * (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha) * (1+np.tanh((f_break-1e-3)/f_scale))
#        elif params['spectrum_model']=='multi_atpl_ipl':
#            delta = 0.1
#            log_Omega0_a = med_vals[2]
#            alpha_a = med_vals[3]
#            f_cut_a = 10**med_vals[4]
#            f_scale_a = 10**med_vals[5]
##                log_Omega0_i = sample[6]
##                alpha_i = sample[7]
#            Omega_1mHz_median = 0.5 * (10**(log_Omega0_a)) * (1e-3/params['fref'])**(alpha_a) * (1+np.tanh((f_cut_a-1e-3)/f_scale_a))
#        else:
#            print("Unknown spectral model. Defaulting to power law...")
#            alpha = med_vals[2]
#            log_Omega0 = med_vals[3]
#            Omega_1mHz_median = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)
#        
#    else:
#        print("Warning: running on older output without specification of spectral model.")
#        print("Warning: defaulting to power law spectral model. This may result in unintended behavior.")
#        alpha = med_vals[2]
#        log_Omega0 = med_vals[3]
#        Omega_1mHz_median = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)
#
#    ## Complex array of blm values for both +ve m values
#    blm_median_vals = np.zeros(blm_size, dtype='complex')
#
#    ## this is b00, alsways set to 1
#    blm_median_vals[0] = 1
#    cnt = 1
#
#    for lval in range(1, blmax + 1):
#        for mval in range(lval + 1):
#
#            idx = Alm.getidx(blmax, lval, mval)
#
#            if mval == 0:
#                blm_median_vals[idx] = blms_median[cnt]
#                cnt = cnt + 1
#            else:
#                ## prior on amplitude, phase
#                blm_median_vals[idx] = blms_median[cnt] * np.exp(1j * blms_median[cnt+1])
#                cnt = cnt + 2
#
#    norm = np.sum(blm_median_vals[0:(blmax + 1)]**2) + np.sum(2*np.abs(blm_median_vals[(blmax + 1):])**2)
#
#    Omega_median_map  =  Omega_1mHz_median * (1.0/norm) * (hp.alm2map(blm_median_vals, nside))**2
#    
#    ## HEALpy is really, REALLY noisy sometimes. This stops that.
#    logger.setLevel(logging.ERROR)
#    
#    if coord=='E':
#        hp.mollview(Omega_median_map, coord=coord, title='Median skymap of $\\Omega(f= 1mHz)$', unit="$\\Omega(f= 1mHz)$")
#    else:
#        hp.mollview(Omega_median_map, coord=['E',coord], title='Median skymap of $\\Omega(f= 1mHz)$', unit="$\\Omega(f= 1mHz)$")
#    
#    hp.graticule()
#    
#    ## switch logging level back to normal so we get our own status updates
#    logger.setLevel(logging.INFO)
#    
#    if saveto is not None:
#        plt.savefig(saveto + '/post_median_skymap.png', dpi=150)
#        logger.info('Saving injected skymap at ' +  saveto + '/post_median_skymap.png')
#
#    else:
#        plt.savefig(params['out_dir'] + '/post_median_skymap.png', dpi=150)
#        logger.info('Saving injected skymap at ' +  params['out_dir'] + '/post_median_skymap.png')
#
#    plt.close()
#
#    return

def fitmaker(post,params,parameters,inj,Model,Injection,plot_convolved=True):
    
    '''
    Make a plot of the spectral fit from the samples generated by the mcmc/nested sampling algorithm.

    Parameters
    -----------

    params : dictionary
        Dictionary of config params

    parameters: string
        Array or list of strings with names of the parameters

    inj : dictionary
        Dictionary of injection params
    '''
    
    print("Computing spectral fit median and 95% CI...")
    ## get samples
#    post = np.loadtxt(params['out_dir'] + "/post_samples.txt")
    
    notation_legend_elements = [Line2D([0], [0], color='k', ls='--', label='Injection'),
                                Line2D([0], [0], color='k', ls='-', label='Median Fit'),
                                Patch(color='k',alpha=0.25,label='$95\%$ C.I.')]
    
    
    ## get frequencies
    frange = Injection.frange
    ffilt = (frange>params['fmin'])*(frange<params['fmax'])
    ## commenting for testing version
#    fs = frange[ffilt][::10]
    fs = frange[ffilt]
    fs = fs.reshape(-1,1)

    
    ## make the deconvolved spectral fit plot
    plt.figure()
    
    ## plot our recovered spectra
    if 'noise' in Model.submodel_names:
        start_idx = 2
    else:
        start_idx = 0
    
    model_legend_elements = []
    ymins = []
    ## loop over submodels
    signal_model_names = [sm_name for sm_name in Model.submodel_names if sm_name!='noise']
    for i, sm_name in enumerate(signal_model_names):
        sm = Model.submodels[sm_name]
        
        model_legend_elements.append(Line2D([0],[0],color=sm.color,lw=3,label=sm.fancyname))
#            post_sm = [post[:,start_idx:(start_idx+sm.Npar)]
        ## this grabs the relevant bits of the posterior vector for each model
        ## will need to fix this for the anisotropic case later...
        post_sm = [post[:,idx] for idx in range(start_idx,start_idx+sm.Npar)]
#        post_sm = post[:,start_idx:start_idx+sm.Npar]
        ## handle any additional spatial variables (will need to fix this when I introduce hierarchical models)
        if hasattr(sm,"blm_start"):
            post_sm = post_sm[sm.blm_start:]
        start_idx += sm.Npar
        ## the spectrum of every sample
        Sgw = sm.compute_Sgw(fs,post_sm)
        ## get summary statistics
        ## median and 95% C.I.
        Sgw_median = np.median(Sgw,axis=1)
        Sgw_upper95 = np.quantile(Sgw,0.975,axis=1)
        Sgw_lower95 = np.quantile(Sgw,0.025,axis=1)
        ymins.append(Sgw_median.min())
        ymins.append(Sgw_lower95.min())
        ## plot
        plt.loglog(fs,Sgw_median,color=sm.color)
        plt.fill_between(fs.flatten(),Sgw_lower95,Sgw_upper95,alpha=0.25,color=sm.color)
        
    if not params['mldc']:
        ## plot the injected spectra, if known
        for component_name in Injection.component_names:
            if component_name != 'noise':
                Injection.plot_injected_spectra(component_name,legend=False,color=Injection.components[component_name].color,ls='--')
    
    ## avoid plot squishing due to signal spectra with cutoffs, etc.
    ymin = np.min(ymins)
    if ymin < 1e-43:
        plt.ylim(bottom=1e-43)
    ax = plt.gca()
    model_legend = ax.legend(handles=model_legend_elements,loc='upper right')
    ax.add_artist(model_legend)
    N_models = len(model_legend_elements)
    notation_legend = ax.legend(handles=notation_legend_elements,loc='upper right',bbox_to_anchor=(1,0.9825-0.056*N_models))
    ax.add_artist(notation_legend)
    
    plt.title("Fit vs. Injection (Astrophysical)")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [1/Hz]')
    plt.savefig(params['out_dir'] + '/spectral_fit_astro.png', dpi=150)
    print("Astrophysical spectral fit plot saved to " + params['out_dir'] + "spectral_fit_astro.png")
    plt.close()
    
    ## plot our recovered convolved spectra if desired
    if plot_convolved:
        model_legend_elements = []
        ymins = []
        plt.figure()
#        ifs = Injection.components['noise'].fs
#        ifilt = (ifs>params['fmin'])*(ifs<params['fmax'])
#        ifs = ifs[ifilt]
#        if0s = Injection.components['noise'].f0[ifilt]
        ## switch back to a flat frequency array, as due to the added memory cost of response convolution, we need to do this as a loop
#        fs = fs.flatten()
#        if 'noise' in Model.submodel_names:
#            fstar = 3e8/(2*np.pi*Model.submodels['noise'].armlength)
#            f0 = fs/(2*fstar)
        start_idx = 0
        ## loop over submodels
#        noise_only = ['noise']
#        for i, sm_name in enumerate(Model.submodel_names):
        for sm_name in Model.submodel_names:
#        for sm_name in noise_only:
            sm = Model.submodels[sm_name]
            
            model_legend_elements.append(Line2D([0],[0],color=sm.color,lw=3,label=sm.fancyname))
            
            fdata = sm.fs
            filt = (fdata>params['fmin'])*(fdata<params['fmax'])
            fdata = fdata[filt]
            f0 = sm.f0[filt]
    #            post_sm = [post[:,start_idx:(start_idx+sm.Npar)]
            ## this grabs the relevant bits of the posterior vector for each model
            ## will need to fix this for the anisotropic case later...
#            import pdb; pdb.set_trace()
            ## the spectrum of every sample
            ## for memory's sake, this needs to be a for loop
            Sgw = np.zeros((post.shape[0],len(fdata)))
            for jj in range(post.shape[0]):
#                post_sm = [post[jj,idx] for idx in range(start_idx,start_idx+sm.Npar)]
                post_sm = post[jj,start_idx:start_idx+sm.Npar]
                ## handle any additional spatial variables (will need to fix this when I introduce hierarchical models)
                if hasattr(sm,"blm_start"):
                    post_sm = post_sm[sm.blm_start:]
                ## handle noise and gw differently, but they all ended up named Sgw. Oh well.
                if sm_name == 'noise':
                    Np = 10**post_sm[0]
                    Na = 10**post_sm[1]
                    Sgw_j = sm.instr_noise_spectrum(fdata,f0,Np=Np,Na=Na)[2,2,:]
#                    import pdb; pdb.set_trace()
                else:
#                    import pdb; pdb.set_trace()
                    Sgw_j = np.mean(sm.compute_Sgw(fdata,post_sm)[:,None] * sm.response_mat[0,0,filt,:],axis=1)
                
                Sgw[jj,:] = np.real(Sgw_j)
            start_idx += sm.Npar
            ## get summary statistics
            ## median and 95% C.I.
            Sgw_median = np.median(Sgw,axis=0)
            Sgw_upper95 = np.quantile(Sgw,0.975,axis=0)
            Sgw_lower95 = np.quantile(Sgw,0.025,axis=0)
            ymins.append(Sgw_median.min())
            ymins.append(Sgw_lower95.min())
            ## plot
            plt.loglog(fdata,Sgw_median,color=sm.color)
            plt.fill_between(fdata,Sgw_lower95,Sgw_upper95,alpha=0.25,color=sm.color)
            
            
        ## now make the convolved spectral fit
        
        if not params['mldc']:
            ## plot the injected spectra, if known
            for component_name in Injection.component_names:
                if component_name == 'noise':
                    Injection.plot_injected_spectra(component_name,channels='22',ls='--',color=Injection.components[component_name].color)
                else:
                    Injection.plot_injected_spectra(component_name,convolved=True,ls='--',color=Injection.components[component_name].color)
        
        ## avoid plot squishing due to signal spectra with cutoffs, etc.
        ymin = np.min(ymins)
        if ymin < 1e-43:
            plt.ylim(bottom=1e-43)
        
        ax = plt.gca()
        model_legend = ax.legend(handles=model_legend_elements,loc='upper right')
        ax.add_artist(model_legend)
        N_models = len(model_legend_elements)
        notation_legend = ax.legend(handles=notation_legend_elements,loc='upper right',bbox_to_anchor=(1,0.9825-0.056*N_models))
        ax.add_artist(notation_legend)
        plt.title("Fit vs. Injection (in Detector)")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [1/Hz]')
        plt.savefig(params['out_dir'] + '/spectral_fit_detector.png', dpi=150)
        print("Detector spectral fit plot saved to " + params['out_dir'] + "spectral_fit_detector.png")
        plt.close()
    
#    
#    Nperseg=int(params['fs']*params['dur'])
#    frange = np.fft.rfftfreq(Nperseg, 1.0/params['fs'])[1:]
#    ffilt = (frange>params['fmin'])*(frange<params['fmax'])
#    ## filter and downsample 
#    fs = frange[ffilt].reshape(-1,1)
#    fs = frange[ffilt][::10]
#    fs = fs.reshape(-1,1)
#    ## need to ensure population construction uses same frequencies as in BLIP
#    if inj['spectral_inj']=='population':
#        fs_inj = frange
#    else:
#        fs_inj = fs
#    
#    if params['spectrum_model'] == 'powerlaw':
#        alpha = post[:,2]
#        log_Omega0 = post[:,3]
#    elif params['spectrum_model']=='broken_powerlaw':
#        log_A1 = post[:,2]
#        alpha_1 = post[:,3]
#        log_A2 = post[:,4]
#        alpha_2 = post[:,3] - 0.667
#    elif params['spectrum_model'] == 'broken_powerlaw_2':
#        log_Omega0 = post[:,2]
#        alpha_1 = post[:,3]
#        alpha_2 = post[:,4]
#        log_fbreak = post[:,5]
#        fbreak = 10**log_fbreak
#        delta = 0.1
#    elif params['spectrum_model']=='free_broken_powerlaw':
#        log_A1 = post[:,2]
#        alpha_1 = post[:,3]
#        log_A2 = post[:,4]
#        alpha_2 = post[:,5]
#    elif params['spectrum_model'] == 'truncated_broken_powerlaw':
#        log_Omega0 = post[:,2]
#        alpha_1 = post[:,3]
#        alpha_2 = post[:,4]
#        log_fbreak = post[:,5]
#        log_fscale= post[:,6]
#        fbreak = 10**log_fbreak
#        fcut = fbreak
#        fscale = 10**log_fscale
#        delta = 0.1
#    elif params['spectrum_model'] == 'truncated_powerlaw':
#        log_Omega0 = post[:,2]
#        alpha = post[:,3]
#        log_fbreak = post[:,4]
#        log_fscale= post[:,5]
#        fbreak = 10**log_fbreak
#        fscale = 10**log_fscale
#    elif params['spectrum_model']=='multi_atpl_ipl':
#        delta = 0.1
#        log_Omega0_a = post[:,2]
#        alpha_a = post[:,3]
#        f_cut_a = 10**post[:,4]
#        f_scale_a = 10**post[:,5]
#        log_Omega0_i = post[:,6]
#        alpha_i = post[:,7]
#    else:
#        print("Unknown spectral model. Exiting without creating plots...")
#        return
#        
#    ## H0 def (SI)
#    H0 = 2.2*10**(-18)
#    
#    ## get injected spectrum
#    if not params['mldc']:
#        if inj['injtype']=='multi':
#            Omegaf_inj_a = (10**inj['log_omega0_a'])*(fs_inj/(params['fref']))**inj['alpha_a'] \
#                    * 0.5 * (1+np.tanh((inj['f_cut_a'] - fs_inj)/inj['f_scale_a']))
#            Sgw_inj_a = Omegaf_inj_a*(3/(4*fs_inj**3))*(H0/np.pi)**2 
#            Omegaf_inj_i = (10**inj['log_omega0_i'])*(fs_inj/(params['fref']))**inj['alpha_i']
#            Sgw_inj_i = Omegaf_inj_i*(3/(4*fs_inj**3))*(H0/np.pi)**2  
#        elif inj['spectral_inj']=='powerlaw':
#            Omegaf_inj = (10**inj['log_omega0'])*(fs_inj/(params['fref']))**inj['alpha']
#            Sgw_inj = Omegaf_inj*(3/(4*fs_inj**3))*(H0/np.pi)**2  
#        elif inj['spectral_inj']=='broken_powerlaw':
#            Omegaf_inj = ((10**inj['log_A1'])*(fs_inj/params['fref'])**inj['alpha1'])/(1 + (10**inj['log_A2'])*(fs/params['fref'])**(inj['alpha1']-0.667))
#            Sgw_inj = Omegaf_inj*(3/(4*fs_inj**3))*(H0/np.pi)**2  
#        elif inj['spectral_inj'] == 'broken_powerlaw_2':
#            delta = 0.1
#            Omegaf_inj = (10**inj['log_omega0'])*(fs_inj/inj['f_break'])**(inj['alpha1']) \
#                    * (0.5*(1+(fs_inj/inj['f_break'])**(1/delta)))**((inj['alpha1']-inj['alpha2'])*delta)
#            Sgw_inj = Omegaf_inj*(3/(4*fs_inj**3))*(H0/np.pi)**2 
#        elif inj['spectral_inj']=='free_broken_powerlaw':
#            Omegaf_inj = ((10**inj['log_A1'])*(fs_inj/params['fref'])**inj['alpha1'])/(1 + (10**inj['log_A2'])*(fs_inj/params['fref'])**(inj['alpha2']))
#            Sgw_inj = Omegaf_inj*(3/(4*fs_inj**3))*(H0/np.pi)**2  
#        elif inj['spectral_inj'] == 'truncated_broken_powerlaw':
#            delta = 0.1
#            Omegaf_inj = (10**inj['log_omega0'])*(fs_inj/inj['f_break'])**(inj['alpha1']) \
#                    * (0.5*(1+(fs_inj/inj['f_break'])**(1/delta)))**((inj['alpha1']-inj['alpha2'])*delta) \
#                    * 0.5 * (1+np.tanh((inj['f_cut'] - fs_inj)/inj['f_scale']))
#            Sgw_inj = Omegaf_inj*(3/(4*fs_inj**3))*(H0/np.pi)**2 
#        elif inj['spectral_inj'] == 'truncated_powerlaw':
#            Omegaf_inj = (10**inj['log_omega0'])*(fs_inj/(params['fref']))**inj['alpha'] \
#                    * 0.5 * (1+np.tanh((inj['f_cut'] - fs_inj)/inj['f_scale']))
#            Sgw_inj = Omegaf_inj*(3/(4*fs_inj**3))*(H0/np.pi)**2 
#        elif inj['spectral_inj']=='population':
#            pop = populations(params,inj)
#            Sgw_inj = pop.pop2spec(inj['popfile'],fs_inj,params['dur']*u.s,return_median=True,names=inj['columns'],sep=inj['delimiter'])*4
#            ## filter to analysis band
#            fs_inj = fs_inj[ffilt]
#            Sgw_inj = Sgw_inj[ffilt]
#        else:
#            print("Other injection types not yet supported, sorry! (Currently supported: powerlaw, broken_powerlaw)")
#            return
#    
#    ## get recovered spectrum
#    if params['spectrum_model']=='powerlaw':
#        Omegaf = (10**log_Omega0)*(fs/(params['fref']))**alpha
#    elif params['spectrum_model']=='broken_powerlaw' or params['spectrum_model']=='free_broken_powerlaw':
#        Omegaf = ((10**log_A1)*(fs/params['fref'])**alpha_1)/(1 + (10**log_A2)*(fs/params['fref'])**alpha_2)
#    elif params['spectrum_model']=='broken_powerlaw_2':
#        Omegaf = (10**log_Omega0)*(fs/fbreak)**(alpha_1) * (0.5*(1+(fs/fbreak)**(1/delta)))**((alpha_1-alpha_2)*delta)
#    elif params['spectrum_model']=='truncated_broken_powerlaw':
#        Omegaf = 0.5 * (10**log_Omega0)*(fs/fbreak)**(alpha_1) * (0.5*(1+(fs/fbreak)**(1/delta)))**((alpha_1-alpha_2)*delta) * (1+np.tanh((fcut-fs)/fscale))
#    elif params['spectrum_model']=='truncated_powerlaw':
#        Omegaf = 0.5 *(10**log_Omega0)*(fs/(params['fref']))**alpha * (1+np.tanh((fbreak-fs)/fscale))
#    elif params['spectrum_model']=='multi_atpl_ipl':
#        Omegaf_a = 0.5 *(10**log_Omega0_a)*(fs/(params['fref']))**alpha_a * (1+np.tanh((f_cut_a-fs)/f_scale_a))
#        Omegaf_i = (10**log_Omega0_i)*(fs/(params['fref']))**alpha_i
#    else:
#        print("Unknown spectral model. Exiting without creating plots...")
#        return
#    
#    if params['modeltype'] == 'multi':
#        Sgw_a = Omegaf_a*(3/(4*fs**3))*(H0/np.pi)**2
#
#        ## median and 95% C.I.
#        Sgw_median_a = np.median(Sgw_a,axis=1)
#        Sgw_upper95_a = np.quantile(Sgw_a,0.975,axis=1)
#        Sgw_lower95_a = np.quantile(Sgw_a,0.025,axis=1)
#        
#        Sgw_i = Omegaf_i*(3/(4*fs**3))*(H0/np.pi)**2
#
#        ## median and 95% C.I.
#        Sgw_median_i = np.median(Sgw_i,axis=1)
#        Sgw_upper95_i = np.quantile(Sgw_i,0.975,axis=1)
#        Sgw_lower95_i = np.quantile(Sgw_i,0.025,axis=1)
#        
#        plt.figure()
#        if not params['mldc']:
#            plt.loglog(fs_inj,Sgw_inj_i,label='Injected Isotropic Spectrum',color='darkorange',ls='--')
#        plt.loglog(fs,Sgw_median_i,label='Median Recovered Isotropic Spectrum',color='darkorange')
#        plt.fill_between(fs.flatten(),Sgw_lower95_i,Sgw_upper95_i,alpha=0.5,label='Isotropic 95% C.I.',color='moccasin')
#        if not params['mldc']:
#            plt.loglog(fs_inj,Sgw_inj_a,label='Injected Anisotropic Spectrum',color='maroon',ls='--')
#        plt.loglog(fs,Sgw_median_a,label='Median Recovered Anisotropic Spectrum',color='maroon')
#        plt.fill_between(fs.flatten(),Sgw_lower95_a,Sgw_upper95_a,alpha=0.5,label='Anisotropic 95% C.I.',color='lightpink')
#    else:
#        Sgw = Omegaf*(3/(4*fs**3))*(H0/np.pi)**2
#    
#        ## median and 95% C.I.
#        Sgw_median = np.median(Sgw,axis=1)
#        Sgw_upper95 = np.quantile(Sgw,0.975,axis=1)
#        Sgw_lower95 = np.quantile(Sgw,0.025,axis=1)
#        
#        plt.figure()
#        if not params['mldc']:
#            plt.loglog(fs_inj,Sgw_inj,label='Injected Spectrum',color='steelblue')
#        plt.loglog(fs,Sgw_median,label='Median Recovered Spectrum',color='darkorange')
#        plt.fill_between(fs.flatten(),Sgw_lower95,Sgw_upper95,alpha=0.5,label='95% C.I.',color='moccasin')
#    plt.legend()
#    plt.title("Fit vs. Injection")
#    plt.xlabel('Frequency [Hz]')
#    plt.ylabel('PSD [1/Hz]')
#    plt.savefig(params['out_dir'] + '/spectral_fit.png', dpi=150)
#    print("Spectral fit plot saved to " + params['out_dir'] + "spectral_fit.png")
#    plt.close()
    
    return #Sgw_median, Sgw_upper95, Sgw_lower95, Sgw_inj, Sgw, fs.flatten()
    

  
def plotmaker(post, params,parameters, inj, Model, Injection):

    '''
    Make posterior plots from the samples generated by tge mcmc/nested sampling algorithm.

    Parameters
    -----------
    post : array
        Collection of posterior samples.

    params : dictionary
        Dictionary of config params

    parameters: string or dict
        Dictionary or list of strings with names of the parameters

    npar : int
        Dimensionality of the parameter space
    '''


#    
#    
#    ## adding this for compatibility with previous runs
#    ## should eventually be depreciated
#    if type(parameters) is dict:
#        all_parameters = parameters['all']
#        ## temp fix
##        if r'$\log_{10} (f_{\mathrm{cut}})$' in all_parameters:
##            all_parameters.remove(r'$\log_{10} (f_{\mathrm{cut}})$')
#    elif type(parameters) is list:
#        all_parameters = parameters
#    else:
#        raise TypeError("parameters argument is not dict or list.")
#    ## if modeltype is sph, first call the mapmaker.
#    if params['modeltype'] not in ['isgwb','isgwb_only','noise_only']:
#        if 'healpy_proj' in params.keys():
#            mapmaker(params,post,parameters,coord=params['healpy_proj'])
#        else:
#            mapmaker(params, post,parameters)
#            
#    ## if spectral fit type is supported, call the fitmaker.
#    if 'spectrum_model' in params.keys():
#        fitmaker(params,parameters,inj)


    ## setup the truevals dict/list
    ## temporary soln... should have all injection methods implemented as dict in future
#    if inj['injtype'] == 'astro':
#        truevals = {}
#    else:
#        truevals = []
#
#    if inj['injtype']=='isgwb':
#
#        truevals.append(inj['log_Np'])
#        truevals.append( inj['log_Na'])
#        truevals.append( inj['alpha'] )
#        truevals.append( inj['log_omega0'] )
#
#    elif inj['injtype']=='noise_only':
#
#        truevals.append(inj['log_Np'])
#        truevals.append( inj['log_Na'])
#
#    elif inj['injtype'] =='isgwb_only':
#
#        truevals.append( inj['alpha'] )
#        truevals.append( inj['log_omega0'] )
#
#    elif inj['injtype']=='sph_sgwb':
#
#        truevals.append(inj['log_Np'])
#        truevals.append( inj['log_Na'])
#        truevals.append( inj['alpha'] )
#        truevals.append( inj['log_omega0'] )

        ## get blms
#        for lval in range(1, params['lmax'] + 1):
#            for mval in range(lval + 1):
#
#                idx = Alm.getidx(params['lmax'], lval, mval)
#
#                if mval == 0:
#                    truevals.append(np.real(inj['blms'][idx]))
#                else:
#                    truevals.append(np.abs(inj['blms'][idx]))
#                    truevals.append(np.angle(inj['blms'][idx]))
    all_parameters = Model.parameters['all']
    
    ## get truevals if not using an external injection
    if not params['mldc']:
        
        inj_truevals = Injection.truevals
        
        truevals = {param:inj_truevals[param] for param in all_parameters if param in inj_truevals.keys()}
#        
#        truevals = {}
#        ## make sure we only plot truevals we have a way of knowing
#        mystery_list = []
#        if inj['spectral_inj']=='population':
#            mystery_list.extend(parameters['signal'])
#        if inj['injtype']=='astro':
#            if inj['injbasis']!='sph':
#                mystery_list.extend(parameters['blm'])
#        if inj['injtype']=='multi':
#            mystery_list.extend(parameters['blm'])
#        param_list = [p for p in parameters['all'] if p not in mystery_list]
#        
#        val_list = [inj['log_Np'],inj['log_Na']]
#        
#        if inj['injtype'] == 'multi':
#            val_list.append( inj['log_omega0_a'] )
#            val_list.append( inj['alpha_a'] )
#            val_list.append( np.log10(inj['f_cut_a']) )
#            val_list.append( np.log10(inj['f_scale_a']) )
#            val_list.append( inj['log_omega0_i'] )
#            val_list.append( inj['alpha_i'] )
#        elif inj['spectral_inj']=='powerlaw':
#            val_list.append( inj['alpha'] )
#            val_list.append( inj['log_omega0'] )
#        elif inj['spectral_inj']=='broken_powerlaw':
#            val_list.append( inj['log_A1'] )
#            val_list.append( inj['alpha1'] )
#            val_list.append( inj['log_A2'] )
#        elif inj['spectral_inj']=='free_broken_powerlaw':
#            val_list.append( inj['log_A1'] )
#            val_list.append( inj['alpha1'] )
#            val_list.append( inj['log_A2'] )
#            val_list.append( inj['alpha2'] )
#        elif inj['spectral_inj']=='broken_powerlaw_2':
#            val_list.append( inj['log_omega0'] )
#            val_list.append( inj['alpha1'] )
#            val_list.append( inj['alpha2'] )
#            val_list.append( inj['f_break'] )
#        elif inj['spectral_inj']=='truncated_broken_powerlaw':
#            val_list.append( inj['log_omega0'] )
#            val_list.append( inj['alpha1'] )
#            val_list.append( inj['alpha2'] )
#            val_list.append( inj['f_break'] )
#            val_list.append( inj['f_scale'] )
#        elif inj['spectral_inj']=='truncated_powerlaw':
#            val_list.append( inj['log_omega0'] )
#            val_list.append( inj['alpha'] )
#            val_list.append( inj['f_break'] )
#            val_list.append( inj['f_scale'] )
#    
#        ## get blms
#        if inj['injtype']=='sph_sgwb':
#            for lval in range(1, params['lmax'] + 1):
#                for mval in range(lval + 1):
#    
#                    idx = Alm.getidx(params['lmax'], lval, mval)
#    
#                    if mval == 0:
#                        val_list.append(np.real(inj['blms'][idx]))
#                    else:
#                        val_list.append(np.abs(inj['blms'][idx]))
#                        val_list.append(np.angle(inj['blms'][idx]))
#        elif inj['injtype']=='astro':
#            if inj['injbasis'] == 'sph':
#                pass
##                for lval in range(1, params['lmax'] + 1):
##                    for mval in range(lval + 1):
##        
##                        idx = Alm.getidx(params['lmax'], lval, mval)
##        
##                        if mval == 0:
##                            val_list.append(np.real(inj['astro_blms'][idx]))
##                        else:
##                            val_list.append(np.abs(inj['astro_blms'][idx]))
##                            val_list.append(np.angle(inj['astro_blms'][idx]))
#        print(param_list)
#        print(val_list)
#        for param, val in zip(param_list,val_list):
#            truevals[param] = val
#        print(truevals)
#        print(len(truevals))
##        ## temporary
##        knowTrue = 0
        
        if len(truevals) > 0:
            knowTrue = 1 ## Bit for whether we know the true vals or not
        else:
            knowTrue = 0
    else:
        knowTrue = 0

    npar = Model.Npar

#    plotrange = [0.999]*npar

    if params['out_dir'][-1] != '/':
        params['out_dir'] = params['out_dir'] + '/'
        
    ## Make chainconsumer corner plots
    cc = ChainConsumer()
    cc.add_chain(post, parameters=all_parameters)
    cc.configure(smooth=False, kde=False, max_ticks=2, sigmas=np.array([1, 2]), label_font_size=18, tick_font_size=18, \
            summary=False, statistics="max_central", spacing=2, summary_area=0.95, cloud=False, bins=1.2)
    cc.configure_truth(color='g', ls='--', alpha=0.7)

    if knowTrue:
        fig = cc.plotter.plot(figsize=(16, 16), truth=truevals)
    else:
        fig = cc.plotter.plot(figsize=(16, 16))

    ## make axis labels to be parameter summaries
    sum_data = cc.analysis.get_summary()
    axes = np.array(fig.axes).reshape((npar, npar))

    # Adjust axis labels
    for ii in range(npar):
        ax = axes[ii, ii]

        # get the right summary for the parameter ii
        sum_ax = sum_data[all_parameters[ii]]
        err =  [sum_ax[2] - sum_ax[1], sum_ax[1]- sum_ax[0]]

        if np.abs(sum_ax[1]) <= 1e-3:
            mean_def = '{0:.3e}'.format(sum_ax[1])
            eidx = mean_def.find('e')
            base = float(mean_def[0:eidx])
            exponent = int(mean_def[eidx+1:])
            mean_form = str(base)
            exp_form = ' \\times ' + '10^{' + str(exponent) + '}'
        else:
            mean_form = '{0:.3f}'.format(sum_ax[1])
            exp_form = ''

        if np.abs(err[0]) <= 1e-2:
            err[0] = '{0:.4f}'.format(err[0])
        else:
            err[0] = '{0:.2f}'.format(err[0])

        if np.abs(err[1]) <= 1e-2:
            err[1] = '{0:.4f}'.format(err[1])
        else:
            err[1] = '{0:.2f}'.format(err[1])

        label =  all_parameters[ii][:-1] + ' = ' + mean_form + '^{+' + err[0] + '}_{-' + err[1] + '}'+exp_form+'$'

        ax.set_title(label, {'fontsize':18}, loc='left')


    ## Save posterior
    plt.savefig(params['out_dir'] + 'corners.png', dpi=150)
    print("Posteriors plots printed in " + params['out_dir'] + "corners.png")
    plt.close()



if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser(prog='plotmaker', usage='%(prog)s [options] rundir', description='run plotmaker')

    # Add arguments
    parser.add_argument('rundir', metavar='rundir', type=str, help='The path to the run directory')

    parser.add_argument('--nofit', action='store_true', help="Disable spectral fit reconstruction plots.")
    parser.add_argument('--nomap', action='store_true', help="Disable skymaps.")
    parser.add_argument('--nocorner', action='store_true', help="Disable corner plots.")
    
    # execute parser
    args = parser.parse_args()

    with open(args.rundir + '/config.pickle', 'rb') as paramfile:
        ## things are loaded from the pickle file in the same order they are put in
        params = pickle.load(paramfile)
        inj = pickle.load(paramfile)
        parameters = pickle.load(paramfile)
    
    ## grab the model and injection
    with open(args.rundir + '/model.pickle', 'rb') as modelfile:
        Model = pickle.load(modelfile)
    with open(args.rundir + '/injection.pickle', 'rb') as injectionfile:
        Injection = pickle.load(injectionfile)
    
    
    post = np.loadtxt(params['out_dir'] + "/post_samples.txt")
    
    if not args.nocorner:
        plotmaker(post, params, parameters, inj, Model, Injection)    
    if not args.nofit:
        fitmaker(post, params, parameters, inj, Model, Injection)
    if not args.nomap:
        if 'healpy_proj' in params.keys():
            mapmaker(post, params, parameters, inj, Model, Injection, coord=params['healpy_proj'])
        else:
            mapmaker(post, params, parameters, inj, Model, Injection)
    
    
#    plotmaker(params, parameters, inj, model, injection)
