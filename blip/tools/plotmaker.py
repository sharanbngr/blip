import sys, os, shutil
sys.path.append(os.getcwd()) ## this lets python find src
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import ScalarFormatter
from chainconsumer import ChainConsumer
import healpy as hp
from healpy import Alm
from astropy import units as u
import pickle, argparse
import logging
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def mapmaker(post, params, parameters, Model, saveto=None, coord=None, cmap=None, post_map_kwargs={}, med_map_kwargs={}, plot_data_path=None):
    '''
    Function to create skymaps from the anisotropic search posteriors.
    
    Arguments
    ---------------
    post (array) : posterior samples
    params (dict) : params dictionary
    inj (dict) : injection params dictionary
    Model (Model object) : Combined Model used for the analysis.
    
    saveto (str) : /path/to/save/skymaps/ (Defaults to params['rundir']).
    coord (str) : Healpy coordinate choice. Defaults to 'E'.
    cmap (matplolib colormap) : Colormap to use for the skymaps.
    post_map_kwargs (dict) : kwargs to be passed to the marginalized posterior skymap mollview plot.
    med_map_kwargs (dict) : kwargs to be passed to the median posterior skymap mollview plot.
    plot_data_path (str) : /path/to/plot_data.pickle; where to save the plot data as a pickle file.
                             Will create the file if it does not exist; otherwise will modify the existing file.
                             Defaults to params['rundir']/plot_data.pickle
    
    '''
    
    ## check for submodels with associated maps
    map_models = []
    for submodel_name in Model.submodel_names:
        sm = Model.submodels[submodel_name]
        if sm.has_map:
            map_models.append(submodel_name)

    if (len(map_models)==0 ):
        print("Called mapmaker but none of the recovery models have a non-isotropic spatial model. Skipping...")
        return
    
    ## load or create plot_data dict
    if plot_data_path is None:
        plot_data_path = params['rundir']+'/plot_data.pickle'
    if os.path.exists(plot_data_path):
        with open(plot_data_path, 'rb') as datafile:
            plot_data = pickle.load(datafile)
            plot_data['map_data'] = {}
    else:
        plot_data = {'map_data':{}}
    
    ## handle projection, kwargs
    # setting coord back to E, if parameter isn't specified
    if coord is None:
        if 'projection' in params.keys():
            coord = ['E',params['projection']]
        else:
            coord = 'E'
    else:
        coord = ['E',coord]
    
    plot_data['map_data']['coord'] = coord
    
    # handling titles, units
    post_base_kwargs = {'title':'Marginalized posterior skymap of $\\Omega(f= 1mHz)$','unit':"$\\Omega(f= 1mHz)$"}
    med_base_kwargs = {'title':'Median skymap of $\\Omega(f= 1mHz)$','unit':"$\\Omega(f= 1mHz)$"}
    
    ## join user-set values to the base settings, overriding when specified
    post_map_kwargs = post_base_kwargs | post_map_kwargs
    med_map_kwargs = med_base_kwargs | med_map_kwargs

    nside = params['nside']
    plot_data['map_data']['nside'] = nside
    npix = hp.nside2npix(nside)

    plot_data['map_data']['maps'] = {}
    
    start_idx = 0   
    for submodel_name in Model.submodel_names:
        ## grab submodel
        sm = Model.submodels[submodel_name]
        
        # Initialize power skymap
        omega_map = np.zeros(npix)
        
        ## only make a map if there's a map to make (this is also good life advice)
        if submodel_name in map_models:
            
            ## kwargs
            post_map_kwargs_i = post_map_kwargs
            
            
            ## HEALpy is really, REALLY noisy sometimes. This stops that.
            logger = logging.getLogger()
            logger.setLevel(logging.ERROR)
            
            ## select relevant posterior columns
            post_i = post[:,start_idx:(start_idx+sm.Npar)]
            

            if hasattr(sm,"fixed_map") and sm.fixed_map:
                ## for analyses with fixed sky distributions
                print("Generating assumed skymap at spectral posterior mean for submodel: {}...".format(submodel_name))
                skip_median = True
                post_map_kwargs_i['title'] = 'Assumed sky distribution evaluated at spectral posterior mean of $\\Omega(f= 1mHz) $'
                if sm.basis=='sph':
                    norm_map = sm.sph_skymap
                elif sm.basis=='pixel':
                    norm_map = sm.skymap / (np.sum(sm.skymap)*(hp.nside2pixarea(nside)/(4*np.pi)))
                else:
                    raise ValueError("Unknown basis specified; can be sph or pixel.")
                    
                for ii in range(post.shape[0]):
                    ## get Omega(f=1mHz)
                    Omega_1mHz = sm.omegaf(1e-3,*post_i[ii,:])
                    omega_map = omega_map + Omega_1mHz * norm_map
            
            elif hasattr(sm,"parameterized_map") and sm.parameterized_map:
                ## for analyses with explicitly parameterized sky distributions (i.e. parameterized but not the more generic sph. harm. model)
                print("Computing marginalized posterior skymap for submodel: {}...".format(submodel_name))
                skip_median = False
                
                for ii in range(post.shape[0]):
                    ## get Omega(f=1mHz)
                    Omega_1mHz = sm.omegaf(1e-3,*post_i[ii,:sm.spatial_start])
                    
                    ## make map from parameterized model
                    skymap_i = sm.compute_skymap(*post_i[ii,sm.spatial_start:])
                    
                    ## mask and norm
                    prob_map = sm.mask_and_norm_pixel_skymap(skymap_i)
                    
                    ## sum on masked pixels
                    omega_map[sm.mask_idx] = omega_map[sm.mask_idx] + Omega_1mHz * prob_map
                    
            else:
                ## sph. harm. analysis
                print("Computing marginalized posterior skymap for submodel: {}...".format(submodel_name))
                skip_median = False
                for ii in range(post.shape[0]):
                    
                    ## get Omega(f=1mHz)
                    Omega_1mHz = sm.omegaf(1e-3,*post_i[ii,:sm.blm_start])
                    
                    ## convert blm params to full blms
                    blm_vals = sm.blm_params_2_blms(post_i[ii,sm.blm_start:])
                    
                    ## normalize, convert to map, and sum
                    norm = np.sum(blm_vals[0:(sm.lmax + 1)]**2) + np.sum(2*np.abs(blm_vals[(sm.lmax + 1):])**2)

                    prob_map  = (1.0/norm) * (hp.alm2map(np.array(blm_vals), nside))**2
                    
                    omega_map = omega_map + Omega_1mHz * prob_map
            
            ## normalize and cast to real to stop Healpy from complaining (all imaginary components are already zero)
            omega_map = np.real(omega_map/post.shape[0])
            
            ## save the map array
            plot_data['map_data']['maps'][submodel_name+'_marginalized'] = omega_map
            
            # generating skymap
            hp.mollview(omega_map, coord=coord, cmap=cmap, **post_map_kwargs_i)
            hp.graticule()
            
            ## switch logging level back to normal so we get our own status updates
            logger.setLevel(logging.INFO)
            
            if saveto is not None:
                fig_path_base = (saveto + '/{}_post_skymap'.format(submodel_name)).replace('//','/')
            else:
                fig_path_base = (params['out_dir'] + '/{}_post_skymap'.format(submodel_name)).replace('//','/')
            
            for ext in ['.png','.pdf']:
                plt.savefig(fig_path_base+ext, dpi=200)
                logger.info('Posterior skymap for submodel {} printed as {} file to  {}'.format(submodel_name,ext,fig_path_base+ext))
            
            plt.close()
            
            
            

            ## now do the median skymap
            if not skip_median:
                print("Computing median posterior skymap for submodel {}...".format(submodel_name))
                
                ## HEALpy is really, REALLY noisy sometimes. This stops that.
                logger.setLevel(logging.ERROR)
                
                # median values of the posteriors
                med_vals = np.median(post_i, axis=0)
                
                if hasattr(sm,"parameterized_map") and sm.parameterized_map:
                    ## explicitly parameterized spatial analyses
                    ## get Omega(f=1mHz)
                    Omega_1mHz_median = sm.omegaf(1e-3,*med_vals[:sm.spatial_start])
                    
                    ## make map from parameterized model
                    skymap_median = sm.compute_skymap(*med_vals[sm.spatial_start:])
                    
                    ## instantiate, mask, and norm
                    ## this ensures the correct pixel ordering is maintained
                    prob_map_median = np.zeros(npix)
                    prob_map_median[sm.mask_idx] = sm.mask_and_norm_pixel_skymap(skymap_median)
                    
                    Omega_median_map = np.real(Omega_1mHz_median * prob_map_median)
                else:
                    ## sph. harm. analysis
                    
                    # Omega(f=1mHz)
                    Omega_1mHz_median = sm.omegaf(1e-3,*med_vals[:sm.blm_start])
                    ## blms.
                    blms_median = np.append([1], med_vals[sm.blm_start:])
                    
                    blm_median_vals = sm.blm_params_2_blms(blms_median)
                
                    norm = np.sum(blm_median_vals[0:(sm.lmax + 1)]**2) + np.sum(2*np.abs(blm_median_vals[(sm.lmax + 1):])**2)
        
                    Omega_median_map  =  np.real(Omega_1mHz_median * (1.0/norm) * (hp.alm2map(np.array(blm_median_vals), nside))**2)
                
                ## save the map array
                plot_data['map_data']['maps'][submodel_name+'_median'] = Omega_median_map
                
                hp.mollview(Omega_median_map, coord=coord, cmap=cmap, **med_map_kwargs)
                
                hp.graticule()
                
                ## switch logging level back to normal so we get our own status updates
                logger.setLevel(logging.INFO)
                
                if saveto is not None:
                    fig_path_base = (saveto + '/{}_post_median_skymap'.format(submodel_name)).replace('//','/')
                else:
                    fig_path_base = (params['out_dir'] + '/{}_post_median_skymap'.format(submodel_name)).replace('//','/')
                
                for ext in ['.png','.pdf']:
                    plt.savefig(fig_path_base+ext, dpi=200)
                    logger.info('Median posterior skymap for submodel {} printed as {} file to  {}'.format(submodel_name,ext,fig_path_base+ext))
            
                plt.close()
            
        
        ## increment start regardless of if we made a map
        start_idx += sm.Npar
    
    ## save map data
    if os.path.exists(plot_data_path):
        ## move to temp file
        temp_file = plot_data_path + ".temp"
        with open(temp_file, "wb") as datafile:
            pickle.dump(plot_data,datafile)
        shutil.move(temp_file, plot_data_path)
    else:
        with open(plot_data_path, 'wb') as datafile:
            plot_data = pickle.dump(plot_data,datafile)
    
    return
    
    
    


def fitmaker(post,params,parameters,inj,Model,Injection=None,saveto=None,plot_convolved=True,astro_kwargs={},det_kwargs={}):
    
    '''
    Make a plot of the spectral fit from the samples generated by the mcmc/nested sampling algorithm.

    Parameters
    -----------

    post : array
        Posterior samples
    
    params : dictionary
        Dictionary of config params

    parameters: string
        Array or list of strings with names of the parameters

    inj : dictionary
        Dictionary of injection params
        
    Model : Model object
        The federated Model used for the analysis
    
    Injection : Injection object
        The federated Injection used to create the data.
    
    *_kwargs : dict
        Keyword argument dictionaries for tweaking the astrophysical/detector plots. Limited number of attributes are supported.
        Supported attributes: figsize, dpi, color_dict, title, title_fontsize, xlabel, xlabel_fontsize, ylabel, ylabel_fontsize, xmin, xmax, ymin, ymax.
        Most of the above are the associated matplotlib argument. The exception is 'color_dict', which should be of the form {'submodel_name':'colorname'}
            and can be used to specify the desired plotting color for specific submodels.
    '''
    
    ## check that an injection was specified if we're not using external data
    if not params['load_data']:
        if Injection is None:
            print("Warning: Not using externally generated data, but no Injection object has been provided to the fitmaker. Returning without making plots...")
            return
    
    ## build the default plot kwargs
    default_kwargs = {'figsize':None,'dpi':150,'color_dict':{},'title':None,'title_fontsize':None,
                      'xlabel':'Frequency [Hz]','xlabel_fontsize':None,'ylabel':'PSD [1/Hz]','ylabel_fontsize':None,
                      'xmin':None,'xmax':None,'ymin':None,'ymax':None}
    ## update astro kwargs
    astro_kwargs = {'title':"Fit vs. Injection (Astrophysical)"} | astro_kwargs
    astro_kwargs = default_kwargs | astro_kwargs
    ## update det kwargs
    det_kwargs = {'title':"Fit vs. Injection (in Detector)"} | det_kwargs
    det_kwargs = default_kwargs | det_kwargs
    
    print("Computing spectral fit median and 95% CI...")
    ## get samples
    
    ## the population injection looks funky with a dashed line, but we still need to make it clear that it's an injection.
    ## this makes the Notation Legend "Injection" label be a split dashed/solid line
    
    if params['load_data']:
        notation_legend_elements = [Line2D([0], [0], color='k', ls='-'),
                                    Patch(color='k',alpha=0.25)]
        notation_legend_labels = ['Median Fit','$95\%$ C.I.']
        notation_handler_map = {}
        notation_handlelength = None
    elif 'population' in Injection.component_names:
        notation_legend_elements = [(Line2D([0], [0], color='k', ls='--'),Line2D([0], [0], color=Injection.components['population'].color,ls='-',lw=0.75,alpha=0.8)),
                                    Line2D([0], [0], color='k', ls='-'),
                                    Patch(color='k',alpha=0.25)]
        notation_legend_labels = ['Injection','Median Fit','$95\%$ C.I.']
        notation_handler_map = {tuple: HandlerTuple(ndivide=None)}
        notation_handlelength = 3
    else:
        notation_legend_elements = [Line2D([0], [0], color='k', ls='--'),
                                    Line2D([0], [0], color='k', ls='-'),
                                    Patch(color='k',alpha=0.25)]
        notation_legend_labels = ['Injection','Median Fit','$95\%$ C.I.']
        notation_handler_map = {}
        notation_handlelength = None
    
    ## get frequencies
    frange = Model.fs
    ffilt = np.logical_and(frange >= params['fmin'], frange <= params['fmax'])
    fs = frange[ffilt]
    fs = fs.reshape(-1,1)

    
    ## make the deconvolved spectral fit plot
    plt.figure(figsize=astro_kwargs['figsize'])
    
    ## plot our recovered spectra
    if 'noise' in Model.submodel_names:
        start_idx = 2
    else:
        start_idx = 0
    
    model_legend_elements = []
    ymins = []
    ymeds = []
    ydevs = []
    ## loop over submodels
    signal_model_names = [sm_name for sm_name in Model.submodel_names if sm_name!='noise']
    if len(signal_model_names) > 0:
        signal_aliases = [Model.submodels[sm_name].alias for sm_name in signal_model_names if hasattr(Model.submodels[sm_name],"alias")]
        for i, sm_name in enumerate(signal_model_names):
            sm = Model.submodels[sm_name]
            model_legend_elements.append(Line2D([0],[0],color=sm.color,lw=3,label=sm.fancyname))
            ## this grabs the relevant bits of the posterior vector for each model
            ## will need to fix this for the anisotropic case later...
            post_sm = [post[:,idx] for idx in range(start_idx,start_idx+sm.Npar)]
            ## handle any additional spatial variables (will need to fix this when I introduce hierarchical models)
            if hasattr(sm,"blm_start"):
                post_sm = post_sm[:sm.blm_start]
            elif hasattr(sm,"spatial_start"):
                post_sm = post_sm[:sm.spatial_start]
            start_idx += sm.Npar
            ## the spectrum of every sample
            Sgw = sm.compute_Sgw(fs,post_sm)
            ## get summary statistics
            ## median and 95% C.I.
            Sgw_median = np.median(Sgw,axis=1)
            Sgw_upper95 = np.quantile(Sgw,0.975,axis=1)
            Sgw_lower95 = np.quantile(Sgw,0.025,axis=1)
            for Sgw_quantile in [Sgw_median,Sgw_lower95]:
                log_Sgw_i = np.log10(Sgw_quantile[np.nonzero(Sgw_quantile)])
                ymins.append(np.min(log_Sgw_i))
                ymeds.append(np.median(log_Sgw_i))
                ydevs.append(np.std(log_Sgw_i))
            ## plot
            plt.loglog(fs,Sgw_median,color=sm.color)
            plt.fill_between(fs.flatten(),Sgw_lower95,Sgw_upper95,alpha=0.25,color=sm.color)

        if not params['load_data']:
            ## plot the injected spectra, if known
            for component_name in Injection.component_names:
                if component_name != 'noise':
                    ## this will overwrite the default linestyle if 'ls' is given in cm.plot_kwargs
                    kwargs = {'ls':'--','color':Injection.components[component_name].color,
                              **Injection.components[component_name].plot_kwargs}
                    ## overwrite color if specified in the the high-level kwargs
                    if component_name in astro_kwargs['color_dict'].keys():
                        kwargs['color'] = astro_kwargs['color_dict'][component_name]
                    inj_Sgw_i = Injection.plot_injected_spectra(component_name,fs_new='data',return_PSD=True,legend=False,ymins=ymins,**kwargs)
                    inj_Sgw_filt_i = inj_Sgw_i[ffilt]
                    log_Sgw_i = np.log10(inj_Sgw_filt_i[np.nonzero(inj_Sgw_filt_i)])
                    ymins.append(np.min(log_Sgw_i))
                    ymeds.append(np.median(log_Sgw_i))
#                    ywmeds.append(weighted_ymed)
                    ydevs.append(np.std(log_Sgw_i))
                    if component_name not in Model.submodel_names and component_name not in signal_aliases:
                        model_legend_elements.append(Line2D([0],[0],color=Injection.components[component_name].color,lw=3,label=Injection.components[component_name].fancyname))

        ## set plot limits, with dynamic scaling for the y-axis to handle spectra with cutoffs, etc.
        if astro_kwargs['ymin'] is None:            
            ylows = [ymed_i - ydev_i for ymed_i,ydev_i in zip(ymeds,ydevs)]
            ylow_min = np.min(ylows)
            plt.ylim(bottom=10**(ylow_min-1))
        else:
            plt.ylim(bottom=astro_kwargs['ymin'])
        plt.ylim(top=astro_kwargs['ymax'])

        ax = plt.gca()
        model_legend = ax.legend(handles=model_legend_elements,loc='upper right')
        ax.add_artist(model_legend)
        N_models = len(model_legend_elements)
        notation_legend = ax.legend(handles=notation_legend_elements,labels=notation_legend_labels,handler_map=notation_handler_map,
                                    handlelength=notation_handlelength,loc='upper right',bbox_to_anchor=(1,0.9825-0.056*N_models))
        ax.add_artist(notation_legend)

        plt.title(astro_kwargs['title'],fontsize=astro_kwargs['title_fontsize'])
        plt.xlabel(astro_kwargs['xlabel'],fontsize=astro_kwargs['xlabel_fontsize'])
        plt.ylabel(astro_kwargs['ylabel'],fontsize=astro_kwargs['ylabel_fontsize'])
        
        ## save astrophysical fit
        if saveto is not None:
            fig_path_base = (saveto + '/spectral_fit_astro').replace('//','/')
        else:
            fig_path_base = (params['out_dir'] + '/spectral_fit_astro').replace('//','/')
        
        for ext in ['.png','.pdf']:
            plt.savefig(fig_path_base+ext, dpi=astro_kwargs['dpi'])
            print("Astrophysical spectral fit plot printed as " + ext + " file to " + fig_path_base+ext)
        
        plt.close()
    
    ## plot our recovered convolved spectra if desired
    if plot_convolved:
        model_legend_elements = []
        ymins = []
        ymeds = []
        ydevs = []

        plt.figure(figsize=det_kwargs['figsize'])

        start_idx = 0
        ## loop over submodels
        for sm_name in Model.submodel_names:
            sm = Model.submodels[sm_name]
            
            model_legend_elements.append(Line2D([0],[0],color=sm.color,lw=3,label=sm.fancyname))
            
            fdata = sm.fs
#            filt = (fdata>params['fmin'])*(fdata<params['fmax'])
            filt = np.logical_and(frange >= params['fmin'], frange <= params['fmax'])
            fdata = fdata[filt]
            f0 = sm.f0[filt]

            ## the spectrum of every sample
            ## for memory's sake, this needs to be a for loop
            Sgw = np.zeros((post.shape[0],len(fdata)))
            for jj in range(post.shape[0]):
                post_sm = post[jj,start_idx:start_idx+sm.Npar]
                ## handle noise and gw differently, but they all ended up named Sgw. Oh well.
                if sm_name == 'noise':
                    Np = 10**post_sm[0]
                    Na = 10**post_sm[1]
                    Sgw_j = sm.instr_noise_spectrum(fdata,f0,Np=Np,Na=Na)[2,2,:]
                ## handle any additional spatial variables (will need to fix this when I introduce hierarchical models)
                elif hasattr(sm,"blm_start"):
                    post_sm_sph = post_sm[sm.blm_start:]
                    post_sm = post_sm[:sm.blm_start]
                    Sgw_j = np.mean(sm.compute_Sgw(fdata,post_sm)[:,None] * sm.compute_summed_response(sm.compute_skymap_alms(post_sm_sph))[0,0,filt,:],axis=1)
                elif hasattr(sm,"spatial_start"):
                    post_sm_spatial = post_sm[sm.spatial_start:]
                    post_sm = post_sm[:sm.spatial_start]
                    Sgw_j = np.mean(sm.compute_Sgw(fdata,post_sm)[:,None] * sm.compute_summed_pixel_response(sm.mask_and_norm_pixel_skymap(sm.compute_skymap(*post_sm_spatial)))[0,0,filt,:],axis=1)
#                    Sgw_j = np.mean(sm.compute_Sgw(fdata,post_sm)[:,None] * sm.compute_summed_response(sm.compute_skymap_alms(post_sm_sph))[0,0,filt,:],axis=1)
                else:
                    Sgw_j = np.mean(sm.compute_Sgw(fdata,post_sm)[:,None] * sm.response_mat[0,0,filt,:],axis=1)
                
                Sgw[jj,:] = np.real(Sgw_j)
            start_idx += sm.Npar
            ## get summary statistics
            ## median and 95% C.I.
            Sgw_median = np.median(Sgw,axis=0)
            Sgw_upper95 = np.quantile(Sgw,0.975,axis=0)
            Sgw_lower95 = np.quantile(Sgw,0.025,axis=0)
            for Sgw_quantile in [Sgw_median,Sgw_lower95]:
                log_Sgw_i = np.log10(Sgw_quantile[np.nonzero(Sgw_quantile)])
                ymins.append(np.min(log_Sgw_i))
                ymeds.append(np.median(log_Sgw_i))
                ydevs.append(np.std(log_Sgw_i))
            ## plot
            plt.loglog(fdata,Sgw_median,color=sm.color)
            plt.fill_between(fdata,Sgw_lower95,Sgw_upper95,alpha=0.25,color=sm.color)
            
            
        ## now make the convolved spectral fit
        
        if not params['load_data']:
            ## plot the injected spectra, if known
            for component_name in Injection.component_names:
                ## this will overwrite the default linestyle if 'ls' is given in cm.plot_kwargs
                kwargs = {'ls':'--','color':Injection.components[component_name].color,
                          **Injection.components[component_name].plot_kwargs}
                ## overwrite color if specified in the the high-level kwargs
                if component_name in det_kwargs['color_dict'].keys():
                    kwargs['color'] = det_kwargs['color_dict'][component_name]
                if component_name == 'noise':
                    Injection.plot_injected_spectra(component_name,fs_new=fdata,channels='22',ymins=ymins,**kwargs)
                else:
                    Injection.plot_injected_spectra(component_name,fs_new='data',convolved=True,ymins=ymins,**kwargs)
                    if component_name not in Model.submodel_names and component_name not in signal_aliases:
                        model_legend_elements.append(Line2D([0],[0],color=Injection.components[component_name].color,lw=3,label=Injection.components[component_name].fancyname))
        
        ## avoid plot squishing due to signal spectra with cutoffs, etc.
        if det_kwargs['ymin'] is None:
            ylows = [ymed_i - ydev_i for ymed_i,ydev_i in zip(ymeds,ydevs)]
            ylow_min = np.min(ylows)
            ## check to see if the diag_spectra() ylim was higher
            ## and use that ylim if so (helps with wonky lower limits)
            if (Injection.plot_ymin is not None) and (Injection.plot_ymin > 10**(ylow_min - 1)):
                ylim_final = Injection.plot_ymin
            else:
                ylim_final = 10**(ylow_min)
            plt.ylim(bottom=ylim_final)

        else:
            plt.ylim(bottom=det_kwargs['ymin'])
        plt.ylim(top=det_kwargs['ymax'])
        
        ax = plt.gca()
        model_legend = ax.legend(handles=model_legend_elements,loc='upper right')
        ax.add_artist(model_legend)
        N_models = len(model_legend_elements)
        notation_legend = ax.legend(handles=notation_legend_elements,labels=notation_legend_labels,handler_map=notation_handler_map,
                                    handlelength=notation_handlelength,loc='upper right',bbox_to_anchor=(1,0.9825-0.056*N_models))
        ax.add_artist(notation_legend)
        
        plt.title(det_kwargs['title'],fontsize=det_kwargs['title_fontsize'])
        plt.xlabel(det_kwargs['xlabel'],fontsize=det_kwargs['xlabel_fontsize'])
        plt.ylabel(det_kwargs['ylabel'],fontsize=det_kwargs['ylabel_fontsize'])
        
        ## save detector fit
        if saveto is not None:
            fig_path_base = (saveto + '/spectral_fit_detector').replace('//','/')
        else:
            fig_path_base = (params['out_dir'] + '/spectral_fit_detector').replace('//','/')
        
        for ext in ['.png','.pdf']:
            plt.savefig(fig_path_base+ext, dpi=det_kwargs['dpi'])
            print("Detector spectral fit plot printed as " + ext + " file to " + fig_path_base+ext)
        
        plt.close()
 
    
    return
    

  
def cornermaker(post, params,parameters, inj, Model, Injection=None, split_by=None, saveto=None):

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

    inj : dictionary
        Dictionary of injection config params
    
    Model : Model() object
        BLIP class with all information about the statistical model
    
    Injection : Injection() object
        BLIP class with all information about the simulated data, if it is BLIP-generated
    
    split_by : str
        How to divvy up the parameters into corner plots. Default (None) places all parameters
        on the same plot. This can get unweildy for high model dimensionality, so this can 
        also be set to "submodel" (makes an individual corner plot for each submodel's parameters)
        or "type" (makes one corner plot for all spectral parameters and one for all spatial parameters).
    
    saveto : str
        Path to save directory. Default None (will save to params['out_dir'])
    '''

    all_parameters = Model.parameters['all']

    if split_by is None:
        parameter_subsets = [all_parameters]
        subset_filts = [np.full(len(all_parameters),True)]
        subset_names = ['all']
    elif split_by == 'type':
        ## we have to be careful here as we can have Models with no spatial parameters whatsoever
        parameter_subsets = []
        subset_names = []
        for subset, name in zip([Model.parameters['spectral'],Model.parameters['spatial']],['spectral','spatial']):
            if len(subset) > 0:
                parameter_subsets.append(subset)
                subset_names.append(name)
        subset_filts = [[(parameter in subset_i) for parameter in all_parameters] for subset_i in parameter_subsets]
    elif split_by == 'submodel':
        parameter_subsets = [Model.parameters[smn] for smn in Model.submodel_names]
        subset_filts = [[(parameter in subset_i) for parameter in all_parameters] for subset_i in parameter_subsets]
        subset_names = Model.submodel_names
        
    else:
        raise ValueError("Unknown specification of 'split_by' ({}). Can be None, 'type', or 'submodel'.".format(split_by))
    
    ## get truevals if not using an external injection
    if not params['load_data']:
        if Injection is None:
            print("Warning: Not using externally generated data, but no Injection object has been provided to the corner plotmaker. Returning without making plots...")
            return
        
        inj_truevals = Injection.truevals
        
        truevals = {}
        for smn in Model.submodel_names:
            for cmn in Injection.component_names:
                if smn == cmn or (hasattr(Model.submodels[smn],"alias") and Model.submodels[smn].alias == cmn):
                    truevals |= {param:inj_truevals[cmn][param] for param in Model.submodels[smn].parameters if param in inj_truevals[cmn].keys()}
                    
        if len(truevals) > 0:
            knowTrue = 1 ## Bit for whether we know the true vals or not
        else:
            knowTrue = 0
    else:
        knowTrue = 0
    
    for i, parameter_subset_i in enumerate(parameter_subsets):
        
        subset_name_i = subset_names[i]
        subset_filt_i = subset_filts[i]
        
        npar = len(parameter_subset_i)
    
        if params['out_dir'][-1] != '/':
            params['out_dir'] = params['out_dir'] + '/'
            
        ## Make chainconsumer corner plots
        cc = ChainConsumer()
        cc.add_chain(post[:,subset_filt_i], parameters=parameter_subset_i)
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
            sum_ax = sum_data[parameter_subset_i[ii]]
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
    
            label =  parameter_subset_i[ii][:-1] + ' = ' + mean_form + '^{+' + err[0] + '}_{-' + err[1] + '}'+exp_form+'$'
    
            ax.set_title(label, {'fontsize':18}, loc='left')
            
            ## chainconsumer has probably messed up the labels on the sides , so reset them
            ## only need to do this for the first column and last row
            if ii in [0,1]:
                ax_left = axes[ii,0]
                ax_left.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                ax_left.set_ylabel(parameter_subset_i[ii])
        
                ax_bottom = axes[-1,ii]
                ax_bottom.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                ax_bottom.set_xlabel(parameter_subset_i[ii])
            
            ## set the labels identical for all axes
            axes[ii,0].set_ylabel(parameter_subset_i[ii],fontsize=18)
            axes[-1,ii].set_xlabel(parameter_subset_i[ii],fontsize=18)
            
        ## plot aesthetics
        fig.align_ylabels()
        fig.align_xlabels()
        
        ## Save posterior
        if saveto is not None:
            fig_path_base = (saveto + '/corners_' + subset_name_i).replace('//','/')
        else:
            fig_path_base = (params['out_dir'] + '/corners_' + subset_name_i).replace('//','/')
        
        for ext in ['.png','.pdf']:
            plt.savefig(fig_path_base+ext, dpi=200, bbox_inches='tight')
            print("Posterior corner plot printed as " + ext + " file to " + fig_path_base+ext)
        plt.close()
        
        if not params['load_data']:    
            # plot walkers
            fig = cc.plotter.plot_walks(truth=truevals, convolve=10)
            plt.savefig(params['out_dir'] + 'plotwalks.png', dpi=200)
            plt.close()


if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser(prog='plotmaker', usage='%(prog)s [options] rundir', description='run plotmaker')

    # Add arguments
    parser.add_argument('rundir', metavar='rundir', type=str, help='The path to the run directory')

    parser.add_argument('--nofit', action='store_true', help="Disable spectral fit reconstruction plots.")
    parser.add_argument('--nomap', action='store_true', help="Disable skymaps.")
    parser.add_argument('--nocorner', action='store_true', help="Disable corner plots.")
    
    parser.add_argument('--cornersplit', type=str, default=None, help="How to split the corner plots. Default None (one corner plot). Can be 'type' or 'submodel'.")
    
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
    if not params['load_data']:
        with open(args.rundir + '/injection.pickle', 'rb') as injectionfile:
            Injection = pickle.load(injectionfile)
    else:
        Injection = None
    
    
    post = np.loadtxt(params['out_dir'] + "/post_samples.txt")
    
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    if not args.nocorner:
        cornermaker(post, params, parameters, inj, Model, Injection=Injection, split_by=args.cornersplit)    
    if not args.nofit:
        fitmaker(post, params, parameters, inj, Model, Injection)
    if not args.nomap:
        if 'healpy_proj' in params.keys():
            mapmaker(post, params, parameters, Model, coord=params['healpy_proj'], cmap=params['colormap'])
        else:
            mapmaker(post, params, parameters, Model, cmap=params['colormap'])

