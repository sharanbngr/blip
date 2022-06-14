import sys, os
sys.path.append(os.getcwd()) ## this lets python find src
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
#import healpy as hp
#from healpy import Alm
import pickle, argparse
#import logging
from src.hierarchical import postprocess
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser(prog='postproc', usage='%(prog)s [options] rundir', description='run hierarchical postprocessing')

    # Add arguments
    parser.add_argument('rundir', metavar='rundir', type=str, help='The path to the run directory.')
    parser.add_argument('--outdir', metavar='outdir', type=str, help='The path to the output directory Defaults to rundir.',default=None)
    parser.add_argument('--model', metavar='model', type=str, help='Parameterized spatial model to use.', default='breivik2020')
    parser.add_argument('--Nwalkers', metavar='Nwalkers', type=int, help='Number of walkers.', default=50)
    parser.add_argument('--Nsamples', metavar='Nsamples', type=int, help='Number of desired samples.', default=10000)
    parser.add_argument('--Nburn', metavar='Nburn', type=int, help='Number of desired burn-in samples.', default=1000)
    parser.add_argument('--seed', metavar='seed', type=int, help='Desired seed for the rng.', default=None)
    parser.add_argument('--Nthread', metavar='Nthread', type=int, help='Number of desired cores for multiprocessing.', default=1)
    # execute parser
    args = parser.parse_args()


    paramfile = open(args.rundir + '/config.pickle', 'rb')
    ## things are loaded from the pickle file in the same order they are put in
    params = pickle.load(paramfile)
    inj = pickle.load(paramfile)
    parameters = pickle.load(paramfile)
    ## initualize the postprocessing class
    postprocessor = postprocess(args.rundir,params,inj,parameters)
    ## run the sampler
    sampler = postprocessor.hierarchical_sampler(model=args.model,Nwalkers=args.Nwalkers,Nsamples=args.Nsamples,Nburn=args.Nburn,rng=args.seed,Nthread=args.Nthread)
    ## plot
    chain = sampler.flatchain
    ## model use cases
    knowTrue = False
    if args.model=='breivik2020':
        npar=2
        post_parameters = ['$r_h$','$z_h$']
        ## deal with older config files and assign true values if known
        if 'fg_type' in inj.keys():
            if inj['fg_type'] == 'breivik2020':
                knowTrue = True
                truevals = [inj['rh'],inj['zh']]
    else:
        raise TypeError("Unknown model. Currently supported models: 'breivik2020'.")
    cc = ChainConsumer()
    cc.add_chain(chain, parameters=post_parameters)
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
        sum_ax = sum_data[post_parameters[ii]]
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

        label =  post_parameters[ii][:-1] + ' = ' + mean_form + '^{+' + err[0] + '}_{-' + err[1] + '}$'

        ax.set_title(label, {'fontsize':18}, loc='left')

    ## save
    if args.outdir is None:
        plt.savefig(args.rundir  + '/postproc_corners.png', dpi=150)
        print("Posteriors plots printed in " + args.rundir + "/postproc_corners.png")
        plt.close()
        np.savetxt(args.rundir+'/postprocessing_samples.txt',chain)
    else:
        plt.savefig(args.outdir  + '/postproc_corners.png', dpi=150)
        print("Posteriors plots printed in " + args.outdir + "/postproc_corners.png")
        plt.close()
        np.savetxt(args.outdir+'/postprocessing_samples.txt',chain)
    







