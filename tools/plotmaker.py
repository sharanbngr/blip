import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import healpy as hp
from healpy import Alm
import pickle, argparse
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def mapmaker(params, post):

    # size of the blm array
    blm_size = Alm.getsize(params['lmax'])

    ## we will plot with a larger nside than the analysis for finer plots
    nside = 2*params['nside']

    npix = hp.nside2npix(nside)

    # Initialize power skymap
    omega_map = np.zeros(npix)

    blmax = params['lmax']

    for ii in range(post.shape[0]):

        sample = post[ii, :]

        # Omega at 1 mHz
        Omega_1mHz = (10**(sample[3])) * (1e-3/25)**(sample[2])

        ## blms.
        blms = np.append([1], sample[4:])

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

        prob_map  = (1.0/norm) * (hp.alm2map(blm_vals, nside , verbose=False))**2

        ## add to the omega map
        omega_map = omega_map + Omega_1mHz * prob_map

    omega_map = omega_map/post.shape[0]

    hp.mollview(omega_map, title='Posterior predictive skymap of $\\Omega(f= 1mHz)$')
    hp.graticule()
    plt.savefig(params['out_dir'] + '/post_skymap.png', dpi=150)
    print('saving injected skymap at ' +  params['out_dir'] + '/post_skymap.png')
    plt.close()


    #### ------------ Now plot median value

    # median values of the posteriors
    med_vals = np.median(post, axis=0)

    ## blms.
    blms_median = np.append([1], med_vals[4:])

    # Omega at 1 mHz
    Omega_1mHz_median = (10**(med_vals[3])) * (1e-3/25)**(med_vals[2])

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

    Omega_median_map  =  Omega_1mHz_median * (1.0/norm) * (hp.alm2map(blm_median_vals, nside , verbose=False))**2

    hp.mollview(omega_map, title='median skymap of $\\Omega(f= 1mHz)$')
    hp.graticule()
    plt.savefig(params['out_dir'] + '/post_median_skymap.png', dpi=150)
    print('saving injected skymap at ' +  params['out_dir'] + '/post_median_skymap.png')
    plt.close()






    return

def plotmaker(params,parameters, inj):

    '''
    Make posterior plots from the samples generated by tge mcmc/nested sampling algorithm.

    Parameters
    -----------

    params : dictionary
        Dictionary of config params

    parameters: string
        Array or list of strings with names of the parameters

    npar : int
        Dimensionality of the parameter space
    '''

    post = np.loadtxt(params['out_dir'] + "/post_samples.txt")

    ## if modeltype is sph, first call the mapmaker.
    if params['modeltype']=='sph_sgwb':
        mapmaker(params, post)


    ## setup the truevals dict
    truevals = []

    if params['modeltype']=='isgwb':

        truevals.append(inj['log_Np'])
        truevals.append( inj['log_Na'])
        truevals.append( inj['alpha'] )
        truevals.append( inj['ln_omega0'] )

    elif params['modeltype']=='noise_only':

        truevals.append(inj['log_Np'])
        truevals.append( inj['log_Na'])

    elif params['modeltype'] =='isgwb_only':

        truevals.append( inj['alpha'] )
        truevals.append( inj['ln_omega0'] )

    elif params['modeltype']=='sph_sgwb':

        truevals.append(inj['log_Np'])
        truevals.append( inj['log_Na'])
        truevals.append( inj['alpha'] )
        truevals.append( inj['ln_omega0'] )

        ## get blms
        for lval in range(1, params['lmax'] + 1):
            for mval in range(lval + 1):

                idx = Alm.getidx(params['lmax'], lval, mval)

                if mval == 0:
                    truevals.append(np.real(inj['blms'][idx]))
                else:
                    truevals.append(np.abs(inj['blms'][idx]))
                    truevals.append(np.angle(inj['blms'][idx]))

    if len(truevals) > 0:
        knowTrue = 1 ## Bit for whether we know the true vals or not
    else:
        knowTrue = 0

    npar = len(parameters)

    plotrange = [0.999]*npar

    if params['out_dir'][-1] != '/':
        params['out_dir'] = params['out_dir'] + '/'

    ## Make chainconsumer corner plots
    cc = ChainConsumer()
    cc.add_chain(post, parameters=parameters)
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
        sum_ax = sum_data[parameters[ii]]
        err =  [sum_ax[2] - sum_ax[1], sum_ax[1]- sum_ax[0]]

        if np.abs(sum_ax[1]) <= 1e-3:
            mean_def = '{0:.3e}'.format(sum_ax[1])
            eidx = mean_def.find('e')
            base = float(mean_def[0:eidx])
            exponent = int(mean_def[eidx+1:])
            mean_form = str(base) + ' \\times ' + '10^{' + str(exponent) + '} '
        else:
            mean_form = '{0:.3f}'.format(sum_ax[1])

        if np.abs(err[0]) <= 1e-2:
            err[0] = '{0:.4f}'.format(err[0])
        else:
            err[0] = '{0:.2f}'.format(err[0])

        if np.abs(err[1]) <= 1e-2:
            err[1] = '{0:.4f}'.format(err[1])
        else:
            err[1] = '{0:.2f}'.format(err[1])

        label =  parameters[ii][:-1] + ' = ' + mean_form + '^{+' + err[0] + '}_{-' + err[1] + '}$'

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


    paramfile = open(args.rundir + '/config.pickle', 'rb')

    ## things are loaded from the pickle file in the same order they are put in
    params = pickle.load(paramfile)
    inj = pickle.load(paramfile)
    parameters = pickle.load(paramfile)

    plotmaker(params, parameters, inj)
