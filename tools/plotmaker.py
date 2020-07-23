import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
from healpy import Alm
import pickle, argparse
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

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

    
    elif params['modeltype'] =='primordial':
        
        truevals.append(inj['log_Np'])
        truevals.append(inj['log_Na'])
        truevals.append(inj['nHat'])
        
        if len(parameters) == 4:
            truevals.append(inj['wHat'])
    
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
    cc.configure(smooth=False, kde=False, max_ticks=3, sigmas=np.array([1, 2]), label_font_size=30, tick_font_size=20, \
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

        ax.set_title(label, {'fontsize':23}, loc='left')


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
