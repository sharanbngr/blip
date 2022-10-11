import numpy as np
import matplotlib
import astropy.coordinates as cc
import astropy.units as u
#matplotlib.use('Agg')
from .makeLISAdata import LISAdata
from .clebschGordan import clebschGordan
#from ..src.makeLISAdata import LISAdata
#from ..src.clebschGordon import clebschGordon
#import ..src.makeLISAdata
#import ..src.clebschGordan
#import ..src.makeLISAdata.generate_galactic_foreground
#import ..src.makeLISAdata.generate_galactic_foreground as generate_galactic_foreground
#import matplotlib.pyplot as plt
#from chainconsumer import ChainConsumer
import healpy as hp
from healpy import Alm
#import pickle, argparse
#import logging
from scipy.stats import multivariate_normal
import emcee
import time
from multiprocessing import Pool
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

class postprocess(LISAdata):
    def __init__(self,rundir,params,inj,parameters):
        self.rundir = rundir
        self.params = params
        self.inj = inj
        self.parameters = parameters
        clebschGordan.__init__(self)
        
    def samples2alm(self,post):
        '''
        Function to convert BLIP blm amplitude+phase posterior samples to the alm basis
        
        Arguments:
            post (array)            : blm posterior samples
    
        Returns:
            alm_samples (array)     : alm posterior samples
        '''

        for ii in range(post.shape[0]):
    
            sample = post[ii, :]
    
            ## get amplitude + phase for each blm
            blms = np.append([1], sample[4:])
            ## convert to blms
            blm_vals = self.blm_params_2_blms(blms)
            ## convert to alms
            alm_vals = self.blm_2_alm(blm_vals)
            
            ## create array on first iteration
            if ii==0:
                alm_samples = np.zeros((post.shape[0],alm_vals.size),dtype='complex')
            alm_samples[ii,:] = alm_vals
            
        return alm_samples
    
    def samples2blm(self,post):
        '''
        Function to convert BLIP blm amplitude+phase posterior samples to the alm basis
        
        Arguments:
            post (array)            : blm posterior samples
    
        Returns:
            alm_samples (array)     : alm posterior samples
        '''

        for ii in range(post.shape[0]):
    
            sample = post[ii, :]
    
            ## get amplitude + phase for each blm
            blms = np.append([1], sample[4:])
            ## convert to blms
            blm_vals = self.blm_params_2_blms(blms)
#            ## convert to alms
#            alm_vals = self.blm_2_alm(blm_vals)
            
            ## create array on first iteration
            if ii==0:
                blm_samples = np.zeros((post.shape[0],blm_vals.size),dtype='complex')
            blm_samples[ii,:] = blm_vals
            
        return blm_samples
#    def post2dist_old(self,post):
#        '''
#        Function to generate a scipy multivariate normal distribution from BLIP blm posteriors.
#        NOTE: Assumes blm posteriors (as transformed to alms) are normally distributed and only uses their moments to determine the distribution.
#        Assumes alm covariance is diagonal.
#        
#        Arguments:
#            post (array)            : blm posterior samples
#    
#        Returns:
#            dist (scipy mv norm)    : Scipy multivariate normal created from the blm posteriors
#        '''
#        ## convert posterior samples to alm basis
#        alm_samples = self.samples2alm(post)
#        ## iterate over blms to get means
#        
#        for lval in range(1, self.params['lmax'] + 1):
#            for mval in range(lval + 1):
#
##                idx = Alm.getidx(self.params['lmax'], lval, mval)
#
##                if mval == 0:
##                    alm_means.append()
##                    truevals.append(np.real(inj['blms'][idx]))
##                else:
##                    truevals.append(np.abs(inj['blms'][idx]))
##                    truevals.append(np.angle(inj['blms'][idx]))
#        ## get means for each alm
#        alm_means = np.mean(alm_samples,axis=0)
#        ## get variance for each alm
#        alm_var = np.var(alm_samples,axis=0)
#        ## create multivariate normal
#        mv_dist = multivariate_normal(mean=alm_means,cov=alm_var)
#        
#        return mv_dist
        
    def blm_decompose(self,blm_samples):
        '''
        Function to decompose complex blms into real-valued components in the proper order.
        
        Arguments:
            blm_samples (array) : complex blms
            
        Returns:
            blms (array)        : real-valued components
        '''
        blms = []
        for lval in range(1, self.params['lmax'] + 1):
            for mval in range(lval + 1):

                idx = Alm.getidx(self.params['lmax'], lval, mval)

                if mval == 0:
                    blms.append(np.real(blm_samples[idx]))
                else:
                    blms.append(np.abs(blm_samples[idx]))
                    blms.append(np.angle(blm_samples[idx]))
        return blms
    
    def post2dist(self,post):
        '''
        Function to generate a scipy multivariate normal distribution from BLIP blm posteriors.
        NOTE: Assumes blm posteriors (as transformed to alms) are normally distributed and only uses their moments to determine the distribution.
        Assumes alm covariance is diagonal.
        
        Arguments:
            post (array)            : blm posterior samples
    
        Returns:
            dist (scipy mv norm)    : Scipy multivariate normal created from the blm posteriors
        '''
        ## convert posterior samples to blm basis
        blm_samples = self.samples2blm(post)
        ## iterate over blms to get means and variances
        blm_means = []
        blm_vars = []
        for lval in range(1, self.params['lmax'] + 1):
            for mval in range(lval + 1):

                idx = Alm.getidx(self.params['lmax'], lval, mval)

                if mval == 0:
                    blm_means.append(np.mean(np.real(blm_samples[idx,:])))
                    blm_vars.append(np.var(np.real(blm_samples[idx,:])))
                else:
                    blm_means.append(np.mean(np.abs(blm_samples[idx,:])))
                    blm_means.append(np.mean(np.angle(blm_samples[idx,:])))
                    blm_vars.append(np.var(np.abs(blm_samples[idx,:])))
                    blm_vars.append(np.var(np.angle(blm_samples[idx,:])))

        ## create multivariate normal
        mv_dist = multivariate_normal(mean=blm_means,cov=blm_vars)
        
        return mv_dist

    def init_breivik2020_grid(self,grid_spec='interval',grid_res=0.33,gal_rad=16,gal_height=8):
        '''
        Function to initialize a grid on which to generate simple parameterized density models of the galactic DWD distribution.
        
        Arguments:
            grid_spec (str)     :   Determines the nature of grid_res (below). Can be 'interval' or 'npoints'. 
                                    If 'interval', grid_res is the dx=dy=dz grid interval in kpc.
                                    If 'npoints', grid_res is the number of number of points along x and y.
                                    (Note that the number of points along z will be scaled to keep dx=dy=dz if gal_rad and gal_height are different.)
            grid_res (float)    :   Grid resolution as defined above. If grid_spec='npoints', type must be int.
            gal_rad (float)     :   Max galactic radius of the grid in kpc. Grid will be definded on -gal_rad <= x,y <= +gal_rad.
            gal_height (float)  :   Max galactic height of the grid in kpc. Grid will be definded on -gal_height <= z <= +gal_height.
            
        '''
        ## create grid *in cartesian coordinates*
        ## size of density grid gives enough padding around the galactic plane without becoming needlessly large
        ## set to 4x max default radial/vertical scale height, respectively (corresponds to "edge" density ~1/10 of central density)
        ## distances in kpc
        if grid_spec=='interval':
            resolution = grid_res
            print("Generating grid with dx = dy = dz = {:0.2f} kpc".format(resolution))
            xs = np.arange(-gal_rad,gal_rad,resolution)
            ys = np.arange(-gal_rad,gal_rad,resolution)
            zs = np.arange(-gal_height,gal_height,resolution)
        elif grid_spec=='npoints':
            if type(grid_res) is not int:
                raise TypeError("If grid_spec is 'npoints', grid_res must be an integer.")
            resolution = gal_rad*2 / grid_res
            print("Generating grid with dx = dy = dz = {:0.2f} kpc".format(resolution))
            xs = np.linspace(-gal_rad,gal_rad,grid_res)
            ys = np.linspace(-gal_rad,gal_rad,grid_res)
            zs = np.arange(-gal_height,gal_height,resolution)
        
        ## generate meshgrid
        x, y, z = np.meshgrid(xs,ys,zs)
        self.z = z
        self.r = np.sqrt(x**2 + y**2)
        ## Use astropy.coordinates to transform from galactocentric frame to galactic (solar system barycenter) frame.
        gc = cc.SkyCoord(x=x*u.kpc,y=y*u.kpc,z=z*u.kpc, frame='galactocentric')
        SSBc = gc.transform_to(cc.Galactic)
        ## 1/D^2 with filtering to avoid nearby, presumeably resolved, DWDs
        self.dist_adj = (np.array(SSBc.distance)>2)*(np.array(SSBc.distance))**-2
        ## make pixel grid
        self.pixels = hp.ang2pix(self.params['nside'],np.array(SSBc.l),np.array(SSBc.b),lonlat=True).flatten()
        self.rGE = hp.rotator.Rotator(coord=['G','E'])
        
        return
        
        

    def breivik2020_mapmaker(self,rh,zh):
        '''
        This is a streamlined version of makeLISAdata.generate_galactic_foreground(), sacrificing compactness for speed.
        Requires initialization via init_breivik2020_grid(), above.
        
        Generate a galactic white dwarf binary foreground modeled after Breivik et al. (2020), consisting of a bulge + disk.
        rh is the radial scale height in kpc, zh is the vertical scale height in kpc. 
        The distribution is azimuthally symmetric in the galactocentric frame.
        
        Returns
        ---------
        DWD_FG_map : float
            Healpy GW power skymap of the DWD galactic foreground.
        
        '''
        ## Calculate density distribution
        r = self.r
        z = self.z
        rho_c = 1 # some fiducial central density
        r_cut = 2.1 #kpc
        r0 = 0.075 #kpc
        alpha = 1.8
        q = 0.5
        disk_density = rho_c*np.exp(-r/rh)*np.exp(-np.abs(z)/zh) 
        bulge_density = rho_c*(np.exp(-(r/r_cut)**2)/(1+np.sqrt(r**2 + (z/q)**2)/r0)**alpha)
        DWD_density = disk_density + bulge_density
        ## use stored grid to convert density to power and filter nearby resolved DWDs
        DWD_unresolved_powers = DWD_density*self.dist_adj
        ## Bin
        DWD_FG_mapG = np.bincount(self.pixels,weights=DWD_unresolved_powers.flatten(),minlength=hp.nside2npix(2*self.params['nside']))
        ## Transform into the ecliptic
        DWD_FG_map = self.rGE.rotate_map_pixel(DWD_FG_mapG)
        
        return DWD_FG_map
        
    
    def breivik2020_log_prior(self,theta,bounds=np.array([[2,4],[0.05,2]])):
        '''
        Prior for the breivik2020 model. Uniform on user-specified bounds in kpc.
        Default bounds are reasonable for the Milky Way.
        
        Arguments:
            theta (array)       : [rh,zh], Breivik+2020 radial and vertical scale height
            bounds (array)      : [[rhmin,rhmax],[zhmin,zhmax]]
    
        Returns:
            logprior (float)              : log prior of theta = {rh,zh}
        '''
        ## unpack theta
        rh,zh = theta
        if rh < bounds[0,0] or rh > bounds[0,1]:
            return -np.inf
        elif zh < bounds[1,0] or zh > bounds[1,1]:
            return -np.inf
        else:
            return 0
    
    def breivik2020_log_likelihood(self,theta,post_dist):
        '''
        Likelihood of theta = {rh,zh} given set of BLIP alm posteriors.
        NOTE: Assumes blm posteriors (as transformed to alms) are normal and only uses their moments to determine the likelihood.
        Assumes blm covariance is diagonal.
        
        Arguments:
            rh (float)                   : Breivik+2020 radial scale height
            zh (float)                   : Breivik+2020 vertical scale height
            post_dist (scipy mv norm)    : Scipy multivariate normal created from the blm posteriors
    
        Returns:
            loglike (float)              : log likelihood of theta = {rh,zh}
        '''
        ## unpack theta
        rh,zh = theta
        ## generate skymaps given rh and zh
#        start = time.time()
        theta_map = self.breivik2020_mapmaker(rh,zh)
#        dur = time.time() - start
#        print('New elapse map gen time is {:0.2f} s.'.format(dur))
#        theta_map, log_theta_map = self.generate_galactic_foreground(rh,zh)
        ## get corresponding blm values
        theta_sph = self.sph_galactic_foreground(theta_map)
        theta_blm = self.blm_decompose(theta_sph)
        ## determine log likelihood
        loglike = post_dist.logpdf(theta_blm)
        
        return loglike
    
    def breivik2020_log_prob(self,theta,post_dist,bounds=np.array([[2,4],[0,2]])):
        '''
        Log probability for the Breivik+2020 model. 
        Prior is uniform on user-specified bounds in kpc; default bounds are reasonable for the Milky Way.
        Likelihood of theta = {rh,zh} given set of BLIP alm posteriors.
        NOTE: Assumes blm posteriors (as transformed to alms) are normal and only uses their moments to determine the likelihood.
        Assumes blm covariance is diagonal.
        
        Arguments:
            theta (array)       : [rh,zh], Breivik+2020 radial and vertical scale height
            post_dist (scipy mv norm)    : Scipy multivariate normal created from the blm posteriors
            bounds (array)      : [[rhmin,rhmax],[zhmin,zhmax]]
    
        Returns:
            logp (float)              : log posterior probability of theta = {rh,zh}
        '''
        ## check prior
        logprior = self.breivik2020_log_prior(theta,bounds)
        if not np.isfinite(logprior):
            return -np.inf
        ## get likelihood
        loglike = self.breivik2020_log_likelihood(theta,post_dist)
        
        return logprior+loglike
    
    def hierarchical_sampler(self,model='breivik2020',Nwalkers=50,Nsamples=10000,Nburn=1000,rng=None,Nthread=1):
        '''
        Function to perform hierarchical sampling to constrain a spatial model of choice.
        '''
        ## set up rng
        if rng is None:
            rng = np.random.default_rng()
        elif type(rng) is int:
            rng = np.random.default_rng(rng)
        elif type(rng) is not np.random._generator.Generator:
            raise TypeError("Invalid specification of the RNG. Can by a numpy default_rng() object, a seed, or None")
        
        ## for now, only option will be the Breivik+2020 model
        if model == 'breivik2020':
            print("Post-processing with spatial model: Breivik+ (2020). Loading posterior samples and parameterizing...")
            ## load posterior samples and process
            post = np.loadtxt(self.rundir + "/post_samples.txt")
            post_dist = self.post2dist(post)
            ## Ndim is 2 {rh,zh}
            Ndim = 2
            ## intialize grid
            self.init_breivik2020_grid()
            ## set up uniform priors on [[rmin,rmax],[zmin,zmax]]
            bounds=np.array([[2,4],[0,2]])
            theta0 = np.array([rng.uniform(low=bounds[0,0],high=bounds[0,1],size=Nwalkers),rng.uniform(low=bounds[1,0],high=bounds[1,1],size=Nwalkers)]).T
            logprob = self.breivik2020_log_prob
            additional_args = (post_dist,bounds)
        else:
            raise TypeError("Unknown model. Currently supported models: 'breivik2020'.")
        
        ## create sampler
        print("Generating Emcee sampler...")
        if Nthread > 1:
            with Pool(Nthread) as pool:
                sampler = emcee.EnsembleSampler(Nwalkers,Ndim,logprob,args=additional_args,pool=pool)
                ## burn-in
                print("Performing {} samples of burn-in...".format(Nburn))
                start = time.time()
                pos, prob, state = sampler.run_mcmc(theta0,Nburn)
                dur = time.time() - start
                print('Time elapsed for burn: {:0.2f} s.'.format(dur))
                ## run
                print("Performing hierarchical sampling...")
                sampler.reset()
                start = time.time()
                pos, prob, state = sampler.run_mcmc(pos,Nsamples)
                dur = time.time() - start
                print('Time elapsed for sampling: {:0.2f} s.'.format(dur))
                pool.close()
                pool.join()
        else:
            sampler = emcee.EnsembleSampler(Nwalkers,Ndim,logprob,args=additional_args)
            ## burn-in
            print("Performing {} samples of burn-in...".format(Nburn))
            start = time.time()
            pos, prob, state = sampler.run_mcmc(theta0,Nburn)
            dur = time.time() - start
            print('Time elapsed for burn: {:0.2f} s.'.format(dur))
            ## run
            print("Performing hierarchical sampling...")
            sampler.reset()
            start = time.time()
            pos, prob, state = sampler.run_mcmc(pos,Nsamples)
            dur = time.time() - start
            print('Time elapsed for sampling: {:0.2f} s.'.format(dur))
            
        return sampler
#
#if __name__ == '__main__':
#
#    # Create parser
#    parser = argparse.ArgumentParser(prog='postproc', usage='%(prog)s [options] rundir', description='run hierarchical postprocessing')
#
#    # Add arguments
#    parser.add_argument('rundir', metavar='rundir', type=str, help='The path to the run directory.')
#    parser.add_argument('--outdir', metavar='outdir', type=str, help='The path to the output directory Defaults to rundir.',default=None)
#    parser.add_argument('--model', metavar='model', type=str, help='Parameterized spatial model to use.', default='breivik2020')
#    parser.add_argument('--Nwalkers', metavar='Nwalkers', type=int, help='Number of walkers.', default=50)
#    parser.add_argument('--Nsamples', metavar='Nsamples', type=int, help='Number of desired samples.', default=10000)
#    parser.add_argument('--Nburn', metavar='Nburn', type=int, help='Number of desired burn-in samples.', default=1000)
#    parser.add_argument('--seed', metavar='seed', type=int, help='Desired seed for the rng.', default=None)
#    # execute parser
#    args = parser.parse_args()
#
#
#    paramfile = open(args.rundir + '/config.pickle', 'rb')
#    ## things are loaded from the pickle file in the same order they are put in
#    params = pickle.load(paramfile)
#    inj = pickle.load(paramfile)
#    parameters = pickle.load(paramfile)
#    ## initualize the postprocessing class
#    postprocess.__init__(params,inj,parameters)
#    ## run the sampler
#    sampler = postprocess.hierarchical_sampler(model=args.model,Nwalkers=args.Nwalkers,Nsamples=args.Nwalkers,Nburn=args.Nburn,rng=args.seed)
#    ## plot
#    chain = sampler.flatchain
#    ## model use cases
#    knowTrue = False
#    if args.model=='breivik2020':
#        npar=2
#        post_parameters = ['$r_h$','$z_h$']
#        if inj['fg_type'] == 'breivik2020':
#            knowTrue = True
#            truevals = [inj['rh'],inj['zh']]
#    else:
#        raise TypeError("Unknown model. Currently supported models: 'breivik2020'.")
#    cc = ChainConsumer()
#    cc.add_chain(chain, parameters=post_parameters)
#    cc.configure(smooth=False, kde=False, max_ticks=2, sigmas=np.array([1, 2]), label_font_size=18, tick_font_size=18, \
#            summary=False, statistics="max_central", spacing=2, summary_area=0.95, cloud=False, bins=1.2)
#    cc.configure_truth(color='g', ls='--', alpha=0.7)
#
#    if knowTrue:
#        fig = cc.plotter.plot(figsize=(16, 16), truth=truevals)
#    else:
#        fig = cc.plotter.plot(figsize=(16, 16))
#
#    ## make axis labels to be parameter summaries
#    sum_data = cc.analysis.get_summary()
#    axes = np.array(fig.axes).reshape((npar, npar))
#
#    # Adjust axis labels
#    for ii in range(npar):
#        ax = axes[ii, ii]
#
#        # get the right summary for the parameter ii
#        sum_ax = sum_data[post_parameters[ii]]
#        err =  [sum_ax[2] - sum_ax[1], sum_ax[1]- sum_ax[0]]
#
#        if np.abs(sum_ax[1]) <= 1e-3:
#            mean_def = '{0:.3e}'.format(sum_ax[1])
#            eidx = mean_def.find('e')
#            base = float(mean_def[0:eidx])
#            exponent = int(mean_def[eidx+1:])
#            mean_form = str(base) + ' \\times ' + '10^{' + str(exponent) + '} '
#        else:
#            mean_form = '{0:.3f}'.format(sum_ax[1])
#
#        if np.abs(err[0]) <= 1e-2:
#            err[0] = '{0:.4f}'.format(err[0])
#        else:
#            err[0] = '{0:.2f}'.format(err[0])
#
#        if np.abs(err[1]) <= 1e-2:
#            err[1] = '{0:.4f}'.format(err[1])
#        else:
#            err[1] = '{0:.2f}'.format(err[1])
#
#        label =  post_parameters[ii][:-1] + ' = ' + mean_form + '^{+' + err[0] + '}_{-' + err[1] + '}$'
#
#        ax.set_title(label, {'fontsize':18}, loc='left')
#
#    ## save
#    if args.outdir is None:
#        plt.savefig(args.rundir  + '/postproc_corners.png', dpi=150)
#        print("Posteriors plots printed in " + args.rundir + "/postproc_corners.png")
#        plt.close()
#        np.savetxt(args.rundir+'/postprocessing_samples.txt')
#    else:
#        plt.savefig(args.outdir  + '/postproc_corners.png', dpi=150)
#        print("Posteriors plots printed in " + args.outdir + "/postproc_corners.png")
#        plt.close()
#        np.savetxt(args.outdir+'/postprocessing_samples.txt')
#    
#
#
#
#
#
#
#
