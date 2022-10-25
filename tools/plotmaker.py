import sys, os
sys.path.append(os.getcwd()) ## this lets python find src
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import healpy as hp
from healpy import Alm
from astropy import units as u
import pickle, argparse
import logging
from src.populations import populations
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def mapmaker(params, post, parameters, saveto=None):
    
    if type(parameters) is dict:
        blm_start = len(parameters['noise']) + len(parameters['signal'])
        
    elif type(parameters) is list:
        print("Warning: using a depreciated parameter format. Number of non-b_lm parameters is unknown, defaulting to n=4.")
        blm_start = 4
    else:
        raise TypeError("parameters argument is not dict or list.")
    
    # size of the blm array
    blm_size = Alm.getsize(params['lmax'])

    ## we will plot with a larger nside than the analysis for finer plots
#    nside = 2*params['nside']
    ## no we won't
    nside = params['nside']

    npix = hp.nside2npix(nside)

    # Initialize power skymap
    omega_map = np.zeros(npix)

    blmax = params['lmax']
    
    print("Computing marginalized posterior skymap...")
    for ii in range(post.shape[0]):

        sample = post[ii, :]

        # Omega at 1 mHz
        # handle various spectral models, but default to power law
        if 'spectrum_model' in params.keys():
            if params['spectrum_model'] == 'powerlaw':
                alpha = sample[2]
                log_Omega0 = sample[3]
                Omega_1mHz = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)
            elif params['spectrum_model']=='broken_powerlaw':
                log_A1 = sample[2]
                alpha_1 = sample[3]
                log_A2 = sample[4]
                alpha_2 = sample[3] - 0.667
                Omega_1mHz= ((10**log_A1)*(1e-3/params['fref'])**alpha_1)/(1 + (10**log_A2)*(1e-3/params['fref'])**alpha_2)
            elif params['spectrum_model']=='free_broken_powerlaw':
                log_A1 = sample[2]
                alpha_1 = sample[3]
                log_A2 = sample[4]
                alpha_2 = sample[5]
                Omega_1mHz= ((10**log_A1)*(1e-3/params['fref'])**alpha_1)/(1 + (10**log_A2)*(1e-3/params['fref'])**alpha_2)
            else:
                print("Unknown spectral model. Defaulting to power law...")
                alpha = sample[2]
                log_Omega0 = sample[3]
                Omega_1mHz = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)
        else:
            print("Warning: running on older output without specification of spectral model.")
            print("Warning: defaulting to power law spectral model. This may result in unintended behavior.")
            alpha = sample[2]
            log_Omega0 = sample[3]
            Omega_1mHz = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)
        ## blms.
        blms = np.append([1], sample[blm_start:])

        ## Complex array of blm values for both +ve m values
        blm_vals = np.zeros(blm_size, dtype='complex')

        ## this is b00, alsways set to 1
        blm_vals[0] = 1
        norm, cnt = 1, 1

        for lval in range(1, blmax + 1):
            for mval in range(lval + 1):

                idx = Alm.getidx(blmax, lval, mval)

                if mval == 0:
                    blm_vals[idx] = blms[cnt]
                    cnt = cnt + 1
                else:
                    ## prior on amplitude, phase
                    blm_vals[idx] = blms[cnt] * np.exp(1j * blms[cnt+1])
                    cnt = cnt + 2

        norm = np.sum(blm_vals[0:(blmax + 1)]**2) + np.sum(2*np.abs(blm_vals[(blmax + 1):])**2)

        prob_map  = (1.0/norm) * (hp.alm2map(blm_vals, nside))**2

        ## add to the omega map
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
   
    # hp.mollview(omega_map, coord=coord, title='Posterior predictive skymap of $\\Omega(f= 1mHz)$')

    hp.graticule()
    
    ## switch logging level back to normal so we get our own status updates
    logger.setLevel(logging.INFO)
    
    if saveto is not None:
        plt.savefig(saveto + '/post_skymap.png', dpi=150)
        logger.info('Saving posterior skymap at ' +  saveto + '/post_skymap.png')

    else:
        plt.savefig(params['out_dir'] + '/post_skymap.png', dpi=150)
        logger.info('Saving posterior skymap at ' +  params['out_dir'] + '/post_skymap.png')
    plt.close()


    #### ------------ Now plot median value
    print("Computing median posterior skymap...")
    # median values of the posteriors
    med_vals = np.median(post, axis=0)

    ## blms.
    blms_median = np.append([1], med_vals[blm_start:])

    # Omega at 1 mHz
    # handle various spectral models, but default to power law
    ## include backwards compatability check (to be depreciated later)
    if 'spectrum_model' in params.keys():
        if params['spectrum_model'] == 'powerlaw':
            alpha = med_vals[2]
            log_Omega0 = med_vals[3]
            Omega_1mHz_median = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)
        elif params['spectrum_model']=='broken_powerlaw':
            log_A1 = med_vals[2]
            alpha_1 = med_vals[3]
            log_A2 = med_vals[4]
            alpha_2 = med_vals[3] - 0.667
            Omega_1mHz_median= ((10**log_A1)*(1e-3/params['fref'])**alpha_1)/(1 + (10**log_A2)*(1e-3/params['fref'])**alpha_2)
        elif params['spectrum_model']=='free_broken_powerlaw':
            log_A1 = med_vals[2]
            alpha_1 = med_vals[3]
            log_A2 = med_vals[4]
            alpha_2 = med_vals[5]
            Omega_1mHz_median= ((10**log_A1)*(1e-3/params['fref'])**alpha_1)/(1 + (10**log_A2)*(1e-3/params['fref'])**alpha_2)
        else:
            print("Unknown spectral model. Defaulting to power law...")
            alpha = med_vals[2]
            log_Omega0 = med_vals[3]
            Omega_1mHz_median = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)
        
    else:
        print("Warning: running on older output without specification of spectral model.")
        print("Warning: defaulting to power law spectral model. This may result in unintended behavior.")
        alpha = med_vals[2]
        log_Omega0 = med_vals[3]
        Omega_1mHz_median = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)

    ## Complex array of blm values for both +ve m values
    blm_median_vals = np.zeros(blm_size, dtype='complex')

    ## this is b00, alsways set to 1
    blm_median_vals[0] = 1
    cnt = 1

    for lval in range(1, blmax + 1):
        for mval in range(lval + 1):

            idx = Alm.getidx(blmax, lval, mval)

            if mval == 0:
                blm_median_vals[idx] = blms_median[cnt]
                cnt = cnt + 1
            else:
                ## prior on amplitude, phase
                blm_median_vals[idx] = blms_median[cnt] * np.exp(1j * blms_median[cnt+1])
                cnt = cnt + 2

    norm = np.sum(blm_median_vals[0:(blmax + 1)]**2) + np.sum(2*np.abs(blm_median_vals[(blmax + 1):])**2)

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

    return

def fitmaker(params,parameters,inj):
    
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
    post = np.loadtxt(params['out_dir'] + "/post_samples.txt")
    ## get frequencies
    Nperseg=int(params['fs']*params['dur'])
    frange = np.fft.rfftfreq(Nperseg, 1.0/params['fs'])[1:]
    ffilt = (frange>params['fmin'])*(frange<params['fmax'])
    ## filter and downsample 
    fs = frange[ffilt].reshape(-1,1)
    fs = frange[ffilt][::10]
    fs = fs.reshape(-1,1)
    ## need to ensure population construction uses same frequencies as in BLIP
    if inj['spectral_inj']=='population':
        fs_inj = frange
    else:
        fs_inj = fs
    
    if params['spectrum_model'] == 'powerlaw':
        alpha = post[:,2]
        log_Omega0 = post[:,3]
    elif params['spectrum_model']=='broken_powerlaw':
        log_A1 = post[:,2]
        alpha_1 = post[:,3]
        log_A2 = post[:,4]
        alpha_2 = post[:,3] - 0.667
    elif params['spectrum_model']=='free_broken_powerlaw':
        log_A1 = post[:,2]
        alpha_1 = post[:,3]
        log_A2 = post[:,4]
        alpha_2 = post[:,5]
    else:
        print("Unknown spectral model. Exiting without creating plots...")
        return
        
    ## H0 def (SI)
    H0 = 2.2*10**(-18)
    
    ## get injected spectrum
    if inj['spectral_inj']=='powerlaw':
        Omegaf_inj =(10**inj['log_omega0'])*(fs_inj/(params['fref']))**inj['alpha']
        Sgw_inj = Omegaf_inj*(3/(4*fs_inj**3))*(H0/np.pi)**2  
    elif inj['spectral_inj']=='broken_powerlaw':
        Omegaf_inj = ((10**inj['log_A1'])*(fs_inj/params['fref'])**inj['alpha1'])/(1 + (10**inj['log_A2'])*(fs/params['fref'])**(inj['alpha1']-0.667))
        Sgw_inj = Omegaf_inj*(3/(4*fs_inj**3))*(H0/np.pi)**2  
    elif inj['spectral_inj']=='free_broken_powerlaw':
        Omegaf_inj = ((10**inj['log_A1'])*(fs_inj/params['fref'])**inj['alpha1'])/(1 + (10**inj['log_A2'])*(fs_inj/params['fref'])**(inj['alpha2']))
        Sgw_inj = Omegaf_inj*(3/(4*fs_inj**3))*(H0/np.pi)**2  
    elif inj['spectral_inj']=='population':
        pop = populations(params,inj)
        Sgw_inj = pop.pop2spec(inj['popfile'],fs_inj,params['dur']*u.s,return_median=True,names=inj['columns'],sep=inj['delimiter'])*4
        ## filter to analysis band
        fs_inj = fs_inj[ffilt]
        Sgw_inj = Sgw_inj[ffilt]
    else:
        print("Other injection types not yet supported, sorry! (Currently supported: powerlaw, broken_powerlaw)")
        return
    
    ## get recovered spectrum
    if params['spectrum_model']=='powerlaw':
        Omegaf = (10**log_Omega0)*(fs/(params['fref']))**alpha
    elif params['spectrum_model']=='broken_powerlaw' or params['spectrum_model']=='free_broken_powerlaw':
        Omegaf = ((10**log_A1)*(fs/params['fref'])**alpha_1)/(1 + (10**log_A2)*(fs/params['fref'])**alpha_2)
    else:
        print("Unknown spectral model. Exiting without creating plots...")
        return
    
    Sgw = Omegaf*(3/(4*fs**3))*(H0/np.pi)**2

    ## median and 95% C.I.
    Sgw_median = np.median(Sgw,axis=1)
    Sgw_upper95 = np.quantile(Sgw,0.975,axis=1)
    Sgw_lower95 = np.quantile(Sgw,0.025,axis=1)
    
    plt.figure()
    plt.loglog(fs_inj,Sgw_inj,label='Injected Spectrum',color='steelblue')
    plt.loglog(fs,Sgw_median,label='Median Recovered Spectrum',color='darkorange')
    plt.fill_between(fs.flatten(),Sgw_lower95,Sgw_upper95,alpha=0.5,label='95% C.I.',color='moccasin')
    plt.legend()
    plt.title("Fit vs. Injection")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [1/Hz]')
    plt.savefig(params['out_dir'] + '/spectral_fit.png', dpi=150)
    print("Spectral fit plot saved to " + params['out_dir'] + "spectral_fit.png")
    plt.close()
    
    return #Sgw_median, Sgw_upper95, Sgw_lower95, Sgw_inj, Sgw, fs.flatten()
    

  
def plotmaker(params,parameters, inj):

    '''
    Make posterior plots from the samples generated by tge mcmc/nested sampling algorithm.

    Parameters
    -----------

    params : dictionary
        Dictionary of config params

    parameters: string or dict
        Dictionary or list of strings with names of the parameters

    npar : int
        Dimensionality of the parameter space
    '''

    post = np.loadtxt(params['out_dir'] + "/post_samples.txt")
    ## adding this for compatibility with previous runs
    ## should eventually be depreciated
    if type(parameters) is dict:
        all_parameters = parameters['all']
    elif type(parameters) is list:
        all_parameters = parameters
    else:
        raise TypeError("parameters argument is not dict or list.")
    ## if modeltype is sph, first call the mapmaker.
    if params['modeltype'] not in ['isgwb','isgwb_only','noise_only']:
        if 'healpy_proj' in params.keys():
            mapmaker(params,post,parameters,coord=params['healpy_proj'])
        else:
            mapmaker(params, post,parameters)
            
    ## if spectral fit type is supported, call the fitmaker.
    if 'spectrum_model' in params.keys():
        fitmaker(params,parameters,inj)


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

    truevals = {}
    ## make sure we only plot truevals we have a way of knowing
    mystery_list = []
    if inj['spectral_inj']=='population':
        mystery_list.extend(parameters['signal'])
    if inj['injtype']=='astro':
        if inj['injbasis']!='sph':
            mystery_list.extend(parameters['blm'])
    param_list = [p for p in parameters['all'] if p not in mystery_list]
    
    val_list = [inj['log_Np'],inj['log_Na']]
    
    if inj['spectral_inj']=='powerlaw':
        val_list.append( inj['alpha'] )
        val_list.append( inj['log_omega0'] )
    elif inj['spectral_inj']=='broken_powerlaw':
        val_list.append( inj['log_A1'] )
        val_list.append( inj['alpha1'] )
        val_list.append( inj['log_A2'] )
    elif inj['spectral_inj']=='free_broken_powerlaw':
        val_list.append( inj['log_A1'] )
        val_list.append( inj['alpha1'] )
        val_list.append( inj['log_A2'] )
        val_list.append( inj['alpha2'] )

    ## get blms
    if inj['injtype']=='sph_sgwb':
        for lval in range(1, params['lmax'] + 1):
            for mval in range(lval + 1):

                idx = Alm.getidx(params['lmax'], lval, mval)

                if mval == 0:
                    val_list.append(np.real(inj['blms'][idx]))
                else:
                    val_list.append(np.abs(inj['blms'][idx]))
                    val_list.append(np.angle(inj['blms'][idx]))
    elif inj['injtype']=='astro':
        if inj['injbasis'] == 'sph':
            for lval in range(1, params['lmax'] + 1):
                for mval in range(lval + 1):
    
                    idx = Alm.getidx(params['lmax'], lval, mval)
    
                    if mval == 0:
                        val_list.append(np.real(inj['astro_blms'][idx]))
                    else:
                        val_list.append(np.abs(inj['astro_blms'][idx]))
                        val_list.append(np.angle(inj['astro_blms'][idx]))
    
    for param, val in zip(param_list,val_list):
        truevals[param] = val
        
#        import pdb;pdb.set_trace()
        
    if len(truevals) > 0:
        knowTrue = 1 ## Bit for whether we know the true vals or not
    else:
        knowTrue = 0

    npar = len(all_parameters)

    plotrange = [0.999]*npar

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

    # execute parser
    args = parser.parse_args()

    with open(args.rundir + '/config.pickle', 'rb') as paramfile:
        ## things are loaded from the pickle file in the same order they are put in
        params = pickle.load(paramfile)
        inj = pickle.load(paramfile)
        parameters = pickle.load(paramfile)
    
    plotmaker(params, parameters, inj)
