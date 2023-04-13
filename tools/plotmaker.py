import sys, os
sys.path.append(os.getcwd()) ## this lets python find src
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from chainconsumer import ChainConsumer
import healpy as hp
from healpy import Alm
from astropy import units as u
import pickle, argparse
import logging
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
            
            ## HEALpy is really, REALLY noisy sometimes. This stops that.
            logger = logging.getLogger()
            logger.setLevel(logging.ERROR)
            
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
            
            ## HEALpy is really, REALLY noisy sometimes. This stops that.
            logger.setLevel(logging.ERROR)
            
            # median values of the posteriors
            med_vals = np.median(post_i, axis=0)
            
            # Omega(f=1mHz)
            Omega_1mHz_median = sm.omegaf(1e-3,*med_vals[:sm.blm_start])
            ## blms.
            blms_median = np.append([1], med_vals[sm.blm_start:])
            
            blm_median_vals = sm.blm_params_2_blms(blms_median)
        
            norm = np.sum(blm_median_vals[0:(sm.lmax + 1)]**2) + np.sum(2*np.abs(blm_median_vals[(sm.lmax + 1):])**2)

            Omega_median_map  =  Omega_1mHz_median * (1.0/norm) * (hp.alm2map(blm_median_vals, nside))**2
            
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
    
    ## the population injection looks funky with a dashed line, but we still need to make it clear that it's an injection.
    ## this makes the Notation Legend "Injection" label be a split dashed/solid line
    if 'population' in Injection.component_names:
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
        ## this grabs the relevant bits of the posterior vector for each model
        ## will need to fix this for the anisotropic case later...
        post_sm = [post[:,idx] for idx in range(start_idx,start_idx+sm.Npar)]
        ## handle any additional spatial variables (will need to fix this when I introduce hierarchical models)
        if hasattr(sm,"blm_start"):
            post_sm = post_sm[:sm.blm_start]
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
                ## this will overwrite the default linestyle if 'ls' is given in cm.plot_kwargs
                kwargs = {'ls':'--','color':Injection.components[component_name].color,
                          **Injection.components[component_name].plot_kwargs}
                Injection.plot_injected_spectra(component_name,legend=False,ymins=ymins,**kwargs)
                if component_name not in Model.submodel_names:
                    model_legend_elements.append(Line2D([0],[0],color=Injection.components[component_name].color,lw=3,label=Injection.components[component_name].fancyname))
    
    ## avoid plot squishing due to signal spectra with cutoffs, etc.
    ymin = np.min(ymins)
    if ymin < 1e-43:
        plt.ylim(bottom=1e-43)
    ax = plt.gca()
    model_legend = ax.legend(handles=model_legend_elements,loc='upper right')
    ax.add_artist(model_legend)
    N_models = len(model_legend_elements)
    notation_legend = ax.legend(handles=notation_legend_elements,labels=notation_legend_labels,handler_map=notation_handler_map,
                                handlelength=notation_handlelength,loc='upper right',bbox_to_anchor=(1,0.9825-0.056*N_models))
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

        start_idx = 0
        ## loop over submodels
        for sm_name in Model.submodel_names:
            sm = Model.submodels[sm_name]
            
            model_legend_elements.append(Line2D([0],[0],color=sm.color,lw=3,label=sm.fancyname))
            
            fdata = sm.fs
            filt = (fdata>params['fmin'])*(fdata<params['fmax'])
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
                else:
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
                ## this will overwrite the default linestyle if 'ls' is given in cm.plot_kwargs
                kwargs = {'ls':'--','color':Injection.components[component_name].color,
                          **Injection.components[component_name].plot_kwargs}
                if component_name == 'noise':
                    Injection.plot_injected_spectra(component_name,channels='22',ymins=ymins,**kwargs)
                else:
                    Injection.plot_injected_spectra(component_name,convolved=True,ymins=ymins,**kwargs)
                    if component_name not in Model.submodel_names:
                        model_legend_elements.append(Line2D([0],[0],color=Injection.components[component_name].color,lw=3,label=Injection.components[component_name].fancyname))
        
        ## avoid plot squishing due to signal spectra with cutoffs, etc.
        ymin = np.min(ymins)
        if ymin < 1e-43:
            plt.ylim(bottom=1e-43)
        
        ax = plt.gca()
        model_legend = ax.legend(handles=model_legend_elements,loc='upper right')
        ax.add_artist(model_legend)
        N_models = len(model_legend_elements)
        notation_legend = ax.legend(handles=notation_legend_elements,labels=notation_legend_labels,handler_map=notation_handler_map,
                                    handlelength=notation_handlelength,loc='upper right',bbox_to_anchor=(1,0.9825-0.056*N_models))
        ax.add_artist(notation_legend)
        plt.title("Fit vs. Injection (in Detector)")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [1/Hz]')
        plt.savefig(params['out_dir'] + '/spectral_fit_detector.png', dpi=150)
        print("Detector spectral fit plot saved to " + params['out_dir'] + "spectral_fit_detector.png")
        plt.close()
 
    
    return
    

  
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

    all_parameters = Model.parameters['all']
    
    ## get truevals if not using an external injection
    if not params['mldc']:
        
        inj_truevals = Injection.truevals
        
        truevals = {param:inj_truevals[param] for param in all_parameters if param in inj_truevals.keys()}
        
        if len(truevals) > 0:
            knowTrue = 1 ## Bit for whether we know the true vals or not
        else:
            knowTrue = 0
    else:
        knowTrue = 0

    npar = Model.Npar

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
    plt.savefig(params['out_dir'] + 'corners.png', dpi=200)
    print("Posteriors plots printed in " + params['out_dir'] + "corners.png")
    plt.close()
    
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

