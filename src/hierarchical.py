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
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
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
        Function to linear-interpolated KDE approximations of BLIP blm posteriors.
        Assumes alm covariance is diagonal and only accounts for the 1-D KDEs on each blm (as KDEs are not reliable for spaces w/ high dimensionality)
        
        Arguments:
            post (array)            : blm posterior samples
    
        Returns:
            dist    : Object with method to compute the combined log probability of (b_lm_proposed | b_lm_samples) over all b_lms
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
        DWD_FG_mapG = np.bincount(self.pixels,weights=DWD_unresolved_powers.flatten(),minlength=hp.nside2npix(self.params['nside']))
        ## Transform into the ecliptic
        DWD_FG_map = self.rGE.rotate_map_pixel(DWD_FG_mapG)
        
        return DWD_FG_map
        
    
    def breivik2020_log_prior(self,theta,bounds=np.array([[2,4],[0.05,2],[0,1]])):
        '''
        Prior for the breivik2020 model. Uniform on user-specified bounds in kpc.
        Default bounds are reasonable for the Milky Way.
        For now, we set a uniform prior on the variance, with default bounds of (0,1].
        
        Arguments:
            theta (array)       : [rh,zh], Breivik+2020 radial and vertical scale height
            bounds (array)      : [[rhmin,rhmax],[zhmin,zhmax]]
    
        Returns:
            logprior (float)              : log prior of theta = {rh,zh}
        '''
        ## unpack theta
        rh,zh,s2 = theta
        if rh < bounds[0,0] or rh > bounds[0,1]:
            return -np.inf
        elif zh < bounds[1,0] or zh > bounds[1,1]:
            return -np.inf
        elif s2 <= bounds[2,0] or s2 > bounds[2,1]:
            return -np.inf
        else:
            return 0
    
    def breivik2020_bayestack_log_prior(self,theta,bounds=np.array([[2,4],[0.05,2]])):
        '''
        Prior for the breivik2020 model. Uniform on user-specified bounds in kpc.
        Default bounds are reasonable for the Milky Way.
        For now, we set a uniform prior on the variance, with default bounds of (0,1].
        
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
    
    def breivik2020_log_likelihood(self,theta,blm_samples):
        '''
        Likelihood of theta = {rh,zh} given set of BLIP alm posteriors.
        NOTE: Assumes blm posteriors (as transformed to alms) are normal and only uses their moments to determine the likelihood.
        Assumes blm covariance is diagonal.
        
        Arguments:
            theta, 1x3 array containing
            rh (float)                   : Breivik+2020 radial scale height
            zh (float)                   : Breivik+2020 vertical scale height
            s2 (float)                   : (assumed identical) variance for the blm components
            
            blm_samples (array)          : blm posterior sample array of shape (N_samples, num_blm)
    
        Returns:
            loglike (float)              : log likelihood of theta = {rh,zh,s2}
        '''
        ## unpack theta
        rh,zh, s2 = theta
        ## generate skymaps given rh and zh
#        start = time.time()
        theta_map = self.breivik2020_mapmaker(rh,zh)
#        dur = time.time() - start
#        print('New elapse map gen time is {:0.2f} s.'.format(dur))
#        theta_map, log_theta_map = self.generate_galactic_foreground(rh,zh)
        ## get corresponding blm values
        theta_sph = self.skymap_pix2sph(theta_map,self.params['lmax'])
        theta_blm = self.blm_decompose(theta_sph)
        
        K = blm_samples.shape[0]
        difflmk = blm_samples - np.array(theta_blm).reshape(-1,1).T
        loglike = -np.log(K) + logsumexp((-0.5/s2)*np.einsum('ij,ik->i',difflmk,difflmk))
        ## determine log likelihood
        
        return loglike
    
    def breivik2020_bayestack_log_likelihood(self,theta,post_dist):
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
        theta_sph = self.skymap_pix2sph(theta_map,self.params['lmax'])
        theta_blm = self.blm_decompose(theta_sph)
        ## determine log likelihood
        loglike = post_dist.logpdf(theta_blm)
        
        return loglike  
    
    def breivik2020_log_prob(self,theta,post,bounds=np.array([[2,4],[0,2]])):
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
        loglike = self.breivik2020_log_likelihood(theta,post)
        
        return logprior+loglike
    
    def breivik2020_bayestack_log_prob(self,theta,post_dist,bounds=np.array([[2,4],[0,2]])):
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
        logprior = self.breivik2020_bayestack_log_prior(theta,bounds)
        if not np.isfinite(logprior):
            return -np.inf
        ## get likelihood
        loglike = self.breivik2020_bayestack_log_likelihood(theta,post_dist)
        
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
            post_full = np.loadtxt(self.rundir + "/post_samples.txt")
            N_spectral_params = len(self.parameters['noise'] + self.parameters['signal'])
            post = post_full[:,N_spectral_params:]
            
#            post_dist = self.post2dist(post)
            ## Ndim is 2 {rh,zh}
            Ndim = 3
            ## intialize grid
            self.init_breivik2020_grid()
            ## set up uniform priors on [[rmin,rmax],[zmin,zmax]]
            bounds=np.array([[2,4],[0,2],[0,1]])
            theta0 = np.array([rng.uniform(low=bounds[0,0],high=bounds[0,1],size=Nwalkers),rng.uniform(low=bounds[1,0],high=bounds[1,1],size=Nwalkers),rng.uniform(low=bounds[2,0],high=bounds[2,1],size=Nwalkers)]).T
            logprob = self.breivik2020_log_prob
            additional_args = (post,bounds)
        elif model == 'breivik2020_bayestack':
            print("Post-processing with spatial model: Breivik+ (2020). Loading posterior samples and parameterizing...")
            print("Using BAYESTACK-style posterior estimation...")
            ## load posterior samples and process
            post_full = np.loadtxt(self.rundir + "/post_samples.txt")
            N_spectral_params = len(self.parameters['noise'] + self.parameters['signal'])
            post = post_full[:,N_spectral_params:]
            
            post_dist = posterior_approx(post,lmax=self.params['lmax'])
            ## Ndim is 2 {rh,zh}
            Ndim = 2
            ## intialize grid
            self.init_breivik2020_grid()
            ## set up uniform priors on [[rmin,rmax],[zmin,zmax]]
            bounds=np.array([[2,4],[0,2]])
            theta0 = np.array([rng.uniform(low=bounds[0,0],high=bounds[0,1],size=Nwalkers),rng.uniform(low=bounds[1,0],high=bounds[1,1],size=Nwalkers)]).T
            logprob = self.breivik2020_bayestack_log_prob
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

class posterior_approx():
    def __init__(self,blm_samples,lmax):
        
        ## this sets up the fast linear interpolators for rapid posterior probability evaluation
        ## the interpolators will be defined on the prior bounds
        ## and return 0 probability outside
        cnt = 0
        amp_range = np.linspace(-3,3,10000)
        phase_range = np.linspace(np.pi,2*np.pi,10000)
        logdists = []
        
        for lval in range(1, lmax + 1):
            for mval in range(lval + 1):

                if mval == 0:
                    kde_lm = gaussian_kde(blm_samples[cnt,:])
                    kde_lm_eval = kde_lm(amp_range)
                    logdist_lm = interp1d(amp_range,np.log(kde_lm_eval),fill_value=-np.inf,bounds_error=False)
                    logdists.append(logdist_lm)
                    cnt = cnt + 1
                else:
                    ## prior on amplitude, phase
                    kde_lm_amp = gaussian_kde(blm_samples[cnt,:])
                    kde_lm_amp_eval = kde_lm_amp(amp_range)
                    logdist_lm_amp = interp1d(amp_range,np.log(kde_lm_amp_eval),fill_value=-np.inf,bounds_error=False)
                    logdists.append(logdist_lm_amp)
                    kde_lm_phase = gaussian_kde(blm_samples[cnt+1,:])
                    kde_lm_phase_eval = kde_lm_phase(phase_range)
                    logdist_lm_phase = interp1d(amp_range,np.log(kde_lm_phase_eval),fill_value=-np.inf,bounds_error=False)
                    logdists.append(logdist_lm_phase)
                    cnt = cnt + 2
        
        self.logdists = logdists
        self.lmax = lmax        
    
    
    
    def logpdf(self,theta_blm):
        
        logp_blm = np.zeros(len(theta_blm))
        
        for idx in range(len(theta_blm)):
            logp_blm[idx] = self.logdists[idx](theta_blm[idx])
        
        return np.sum(logp_blm)


class hierarchy(LISAdata):
    '''
    Class to handle hierarchical analysis components for dynamic usage.
    '''
    def __init__(self,params):
        self.params = params
#        self.init_breivik2020_grid()
        
    
#    @classmethod
#    def define_hierarchy(cls,lisaobj,params):
#    def init_breivik2020_grid(self,grid_spec='interval',grid_res=0.33,gal_rad=16,gal_height=8):
#        '''
#        Function to initialize a grid on which to generate simple parameterized density models of the galactic DWD distribution.
#        
#        Arguments:
#            grid_spec (str)     :   Determines the nature of grid_res (below). Can be 'interval' or 'npoints'. 
#                                    If 'interval', grid_res is the dx=dy=dz grid interval in kpc.
#                                    If 'npoints', grid_res is the number of number of points along x and y.
#                                    (Note that the number of points along z will be scaled to keep dx=dy=dz if gal_rad and gal_height are different.)
#            grid_res (float)    :   Grid resolution as defined above. If grid_spec='npoints', type must be int.
#            gal_rad (float)     :   Max galactic radius of the grid in kpc. Grid will be definded on -gal_rad <= x,y <= +gal_rad.
#            gal_height (float)  :   Max galactic height of the grid in kpc. Grid will be definded on -gal_height <= z <= +gal_height.
#            
#        '''
#        ## create grid *in cartesian coordinates*
#        ## size of density grid gives enough padding around the galactic plane without becoming needlessly large
#        ## set to 4x max default radial/vertical scale height, respectively (corresponds to "edge" density ~1/10 of central density)
#        ## distances in kpc
#        if grid_spec=='interval':
#            resolution = grid_res
#            print("Generating grid with dx = dy = dz = {:0.2f} kpc".format(resolution))
#            xs = np.arange(-gal_rad,gal_rad,resolution)
#            ys = np.arange(-gal_rad,gal_rad,resolution)
#            zs = np.arange(-gal_height,gal_height,resolution)
#        elif grid_spec=='npoints':
#            if type(grid_res) is not int:
#                raise TypeError("If grid_spec is 'npoints', grid_res must be an integer.")
#            resolution = gal_rad*2 / grid_res
#            print("Generating grid with dx = dy = dz = {:0.2f} kpc".format(resolution))
#            xs = np.linspace(-gal_rad,gal_rad,grid_res)
#            ys = np.linspace(-gal_rad,gal_rad,grid_res)
#            zs = np.arange(-gal_height,gal_height,resolution)
#        
#        ## generate meshgrid
#        x, y, z = np.meshgrid(xs,ys,zs)
#        self.z = z
#        self.r = np.sqrt(x**2 + y**2)
#        ## Use astropy.coordinates to transform from galactocentric frame to galactic (solar system barycenter) frame.
#        gc = cc.SkyCoord(x=x*u.kpc,y=y*u.kpc,z=z*u.kpc, frame='galactocentric')
#        SSBc = gc.transform_to(cc.Galactic)
#        ## 1/D^2 with filtering to avoid nearby, presumeably resolved, DWDs
#        self.dist_adj = (np.array(SSBc.distance)>2)*(np.array(SSBc.distance))**-2
#        ## make pixel grid
#        self.pixels = hp.ang2pix(self.params['nside'],np.array(SSBc.l),np.array(SSBc.b),lonlat=True).flatten()
#        self.rGE = hp.rotator.Rotator(coord=['G','E'])
#        
#        return    
    
    def init_breivik2020_grid(self):
        '''
        Function to amortize all possible calculations for a hierarchical galactic foreground spatial prior based on the Breivik2020 galaxy model.
        
        Calculates all quantities analytically transformed into the SSB frame in spherical coordinates, and integrates over the (rh,zh)-independent bulge density.
        
        Resolution is the analysis resolution on the sky and 0.33 kpc in Rprime.
        '''
        ## sky resolution
        npix = hp.nside2npix(self.params['nside'])
        ## grid in R' with 0.33 kpc resolution
        gridspec = 120
        self.Rprime = np.linspace(2,40,gridspec)[:,None]
        self.Rprime2 = self.Rprime**2
        self.delta_Rprime = self.Rprime[1] - self.Rprime[0]
        ## skymap & angles
        pix_idx = np.arange(npix)
        theta, phi = hp.pix2ang(self.params['nside'],pix_idx)
    #     print(np.min(theta),np.max(theta))
    #     print(np.min(phi),np.max(phi))
        # theta = -(theta - np.pi/2)
        # phi = phi - np.pi
        theta = theta[None,:]
        phi = phi[None,:]
        ## amortize angular computations
        stheta = np.sin(theta)
        ctheta = np.cos(theta)
        s2theta = stheta**2
        cphi = np.cos(phi)
        ## set constants
        ## for astropy conventions see https://astropy-cjhang.readthedocs.io/en/latest/api/astropy.coordinates.Galactocentric.html#astropy.coordinates.Galactocentric
        r_s = 8.3 ## distance from sun to galactic center in kpc, following astropy convention
        z_s = 0.027 ## distance of sun above galactic midplane, following astropy convention
        r_cut = 2.1 #kpc
        r0 = 0.075 #kpc
        alpha = 1.8
        q = 0.5
        ## one-time bulge computation
        bulge_dense = np.exp(-np.abs(self.Rprime2*s2theta + r_s**2 - 2*self.Rprime*r_s*cphi*stheta)/r_cut**2) \
                    / (1 + (1/r0)*(np.abs(self.Rprime2*s2theta + r_s**2 - 2*self.Rprime*r_s*cphi*stheta) + (1/q**2)*(self.Rprime*ctheta + z_s)**2))**alpha
        self.bulge_map = self.delta_Rprime * np.sum(bulge_dense / self.Rprime2,axis=0)
        ## non-rh/zh dependent quantities in the disk distribution can also be computed just once
        self.rarg = np.sqrt(self.Rprime2*s2theta + r_s**2 - 2*self.Rprime*r_s*cphi*stheta)
        self.zarg = np.abs(self.Rprime*ctheta + z_s)
        ## get the galactic -> ecliptic rotator
        self.rGE = hp.rotator.Rotator(coord=['G','E'])
    
    def breivik2020_mapmaker(self,rh,zh):
        '''
        This is a streamlined version of makeLISAdata.generate_galactic_foreground(), sacrificing compactness for speed.
        Requires initialization via init_breivik2020_grid(), above.
        
        v2: Instead of performing all calculations on a cartesian grid, transforming, and binning, we now calculate the model as analytically transformed into spherical coordinates in the SSB frame.
        All non rh/zh-dependent quantities are calculated once & amortized in init_breivik2020_grid(), so as to do as few calculations at runtime as possible.
        This version is much less transparent to a reader, but gives a ~30x speedup at similar grid resolutions.
        
        Generate a galactic white dwarf binary foreground modeled after Breivik et al. (2020), consisting of a bulge + disk.
        rh is the radial scale height in kpc, zh is the vertical scale height in kpc. 
        The distribution is azimuthally symmetric in the galactocentric frame.
        
        Arguments
        -----------
        rh, zh : float
            Model radial/vertical scale height parameters, respectively.
        
        Returns
        -----------
        DWD_FG_map : float
            Healpy GW power skymap of the DWD galactic foreground.
        
        '''
        ## Calculate disk density in SSB frame with amortized arguments
        disk_dense = np.exp(-self.rarg/rh) * np.exp(-self.zarg/zh)
        disk_map = self.delta_Rprime * np.sum(disk_dense / self.Rprime2,axis=0)
        galactic_map = disk_map + self.bulge_map
        ## transform to ecliptic and return
        ecliptic_map = self.rGE.rotate_map_pixel(galactic_map)
        
        return ecliptic_map
    
    
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
    
    @staticmethod
    def skymap_pix2sph(skymap, blmax):
        '''
        Transform the foreground produced in generate_galactic_foreground() into
        b_lm spherical harmonic basis
        
        Returns
        ---------
        astro_blms : float
            Spherical harmonic healpy expansion of the galactic foreground
        '''
        ## Take square root of powers
        sqrt_map = np.sqrt(skymap)
        ## Generate blms of power (alms of sqrt(power))
        astro_blms = hp.sphtfunc.map2alm(sqrt_map, lmax=blmax)

        # Normalize such that b00 = 1    
        astro_blms = astro_blms/(astro_blms[0])

        return astro_blms
    
    
    
    
## previous (assumption-heavy) version of the postprocessing analysis    

#class postprocess_old(LISAdata):
#    def __init__(self,rundir,params,inj,parameters):
#        self.rundir = rundir
#        self.params = params
#        self.inj = inj
#        self.parameters = parameters
#        clebschGordan.__init__(self)
#        
#    def samples2alm(self,post):
#        '''
#        Function to convert BLIP blm amplitude+phase posterior samples to the alm basis
#        
#        Arguments:
#            post (array)            : blm posterior samples
#    
#        Returns:
#            alm_samples (array)     : alm posterior samples
#        '''
#
#        for ii in range(post.shape[0]):
#    
#            sample = post[ii, :]
#    
#            ## get amplitude + phase for each blm
#            blms = np.append([1], sample[4:])
#            ## convert to blms
#            blm_vals = self.blm_params_2_blms(blms)
#            ## convert to alms
#            alm_vals = self.blm_2_alm(blm_vals)
#            
#            ## create array on first iteration
#            if ii==0:
#                alm_samples = np.zeros((post.shape[0],alm_vals.size),dtype='complex')
#            alm_samples[ii,:] = alm_vals
#            
#        return alm_samples
#    
#    def samples2blm(self,post):
#        '''
#        Function to convert BLIP blm amplitude+phase posterior samples to the alm basis
#        
#        Arguments:
#            post (array)            : blm posterior samples
#    
#        Returns:
#            alm_samples (array)     : alm posterior samples
#        '''
#
#        for ii in range(post.shape[0]):
#    
#            sample = post[ii, :]
#    
#            ## get amplitude + phase for each blm
#            blms = np.append([1], sample[4:])
#            ## convert to blms
#            blm_vals = self.blm_params_2_blms(blms)
##            ## convert to alms
##            alm_vals = self.blm_2_alm(blm_vals)
#            
#            ## create array on first iteration
#            if ii==0:
#                blm_samples = np.zeros((post.shape[0],blm_vals.size),dtype='complex')
#            blm_samples[ii,:] = blm_vals
#            
#        return blm_samples
##    def post2dist_old(self,post):
##        '''
##        Function to generate a scipy multivariate normal distribution from BLIP blm posteriors.
##        NOTE: Assumes blm posteriors (as transformed to alms) are normally distributed and only uses their moments to determine the distribution.
##        Assumes alm covariance is diagonal.
##        
##        Arguments:
##            post (array)            : blm posterior samples
##    
##        Returns:
##            dist (scipy mv norm)    : Scipy multivariate normal created from the blm posteriors
##        '''
##        ## convert posterior samples to alm basis
##        alm_samples = self.samples2alm(post)
##        ## iterate over blms to get means
##        
##        for lval in range(1, self.params['lmax'] + 1):
##            for mval in range(lval + 1):
##
###                idx = Alm.getidx(self.params['lmax'], lval, mval)
##
###                if mval == 0:
###                    alm_means.append()
###                    truevals.append(np.real(inj['blms'][idx]))
###                else:
###                    truevals.append(np.abs(inj['blms'][idx]))
###                    truevals.append(np.angle(inj['blms'][idx]))
##        ## get means for each alm
##        alm_means = np.mean(alm_samples,axis=0)
##        ## get variance for each alm
##        alm_var = np.var(alm_samples,axis=0)
##        ## create multivariate normal
##        mv_dist = multivariate_normal(mean=alm_means,cov=alm_var)
##        
##        return mv_dist
#        
#    def blm_decompose(self,blm_samples):
#        '''
#        Function to decompose complex blms into real-valued components in the proper order.
#        
#        Arguments:
#            blm_samples (array) : complex blms
#            
#        Returns:
#            blms (array)        : real-valued components
#        '''
#        blms = []
#        for lval in range(1, self.params['lmax'] + 1):
#            for mval in range(lval + 1):
#
#                idx = Alm.getidx(self.params['lmax'], lval, mval)
#
#                if mval == 0:
#                    blms.append(np.real(blm_samples[idx]))
#                else:
#                    blms.append(np.abs(blm_samples[idx]))
#                    blms.append(np.angle(blm_samples[idx]))
#        return blms
#    
#    def post2dist(self,post):
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
#        ## convert posterior samples to blm basis
#        blm_samples = self.samples2blm(post)
#        ## iterate over blms to get means and variances
#        blm_means = []
#        blm_vars = []
#        for lval in range(1, self.params['lmax'] + 1):
#            for mval in range(lval + 1):
#
#                idx = Alm.getidx(self.params['lmax'], lval, mval)
#
#                if mval == 0:
#                    blm_means.append(np.mean(np.real(blm_samples[idx,:])))
#                    blm_vars.append(np.var(np.real(blm_samples[idx,:])))
#                else:
#                    blm_means.append(np.mean(np.abs(blm_samples[idx,:])))
#                    blm_means.append(np.mean(np.angle(blm_samples[idx,:])))
#                    blm_vars.append(np.var(np.abs(blm_samples[idx,:])))
#                    blm_vars.append(np.var(np.angle(blm_samples[idx,:])))
#
#        ## create multivariate normal
#        mv_dist = multivariate_normal(mean=blm_means,cov=blm_vars)
#        
#        return mv_dist
#
#    def init_breivik2020_grid(self,grid_spec='interval',grid_res=0.33,gal_rad=16,gal_height=8):
#        '''
#        Function to initialize a grid on which to generate simple parameterized density models of the galactic DWD distribution.
#        
#        Arguments:
#            grid_spec (str)     :   Determines the nature of grid_res (below). Can be 'interval' or 'npoints'. 
#                                    If 'interval', grid_res is the dx=dy=dz grid interval in kpc.
#                                    If 'npoints', grid_res is the number of number of points along x and y.
#                                    (Note that the number of points along z will be scaled to keep dx=dy=dz if gal_rad and gal_height are different.)
#            grid_res (float)    :   Grid resolution as defined above. If grid_spec='npoints', type must be int.
#            gal_rad (float)     :   Max galactic radius of the grid in kpc. Grid will be definded on -gal_rad <= x,y <= +gal_rad.
#            gal_height (float)  :   Max galactic height of the grid in kpc. Grid will be definded on -gal_height <= z <= +gal_height.
#            
#        '''
#        ## create grid *in cartesian coordinates*
#        ## size of density grid gives enough padding around the galactic plane without becoming needlessly large
#        ## set to 4x max default radial/vertical scale height, respectively (corresponds to "edge" density ~1/10 of central density)
#        ## distances in kpc
#        if grid_spec=='interval':
#            resolution = grid_res
#            print("Generating grid with dx = dy = dz = {:0.2f} kpc".format(resolution))
#            xs = np.arange(-gal_rad,gal_rad,resolution)
#            ys = np.arange(-gal_rad,gal_rad,resolution)
#            zs = np.arange(-gal_height,gal_height,resolution)
#        elif grid_spec=='npoints':
#            if type(grid_res) is not int:
#                raise TypeError("If grid_spec is 'npoints', grid_res must be an integer.")
#            resolution = gal_rad*2 / grid_res
#            print("Generating grid with dx = dy = dz = {:0.2f} kpc".format(resolution))
#            xs = np.linspace(-gal_rad,gal_rad,grid_res)
#            ys = np.linspace(-gal_rad,gal_rad,grid_res)
#            zs = np.arange(-gal_height,gal_height,resolution)
#        
#        ## generate meshgrid
#        x, y, z = np.meshgrid(xs,ys,zs)
#        self.z = z
#        self.r = np.sqrt(x**2 + y**2)
#        ## Use astropy.coordinates to transform from galactocentric frame to galactic (solar system barycenter) frame.
#        gc = cc.SkyCoord(x=x*u.kpc,y=y*u.kpc,z=z*u.kpc, frame='galactocentric')
#        SSBc = gc.transform_to(cc.Galactic)
#        ## 1/D^2 with filtering to avoid nearby, presumeably resolved, DWDs
#        self.dist_adj = (np.array(SSBc.distance)>2)*(np.array(SSBc.distance))**-2
#        ## make pixel grid
#        self.pixels = hp.ang2pix(self.params['nside'],np.array(SSBc.l),np.array(SSBc.b),lonlat=True).flatten()
#        self.rGE = hp.rotator.Rotator(coord=['G','E'])
#        
#        return
#        
#        
#
#    def breivik2020_mapmaker(self,rh,zh):
#        '''
#        This is a streamlined version of makeLISAdata.generate_galactic_foreground(), sacrificing compactness for speed.
#        Requires initialization via init_breivik2020_grid(), above.
#        
#        Generate a galactic white dwarf binary foreground modeled after Breivik et al. (2020), consisting of a bulge + disk.
#        rh is the radial scale height in kpc, zh is the vertical scale height in kpc. 
#        The distribution is azimuthally symmetric in the galactocentric frame.
#        
#        Returns
#        ---------
#        DWD_FG_map : float
#            Healpy GW power skymap of the DWD galactic foreground.
#        
#        '''
#        ## Calculate density distribution
#        r = self.r
#        z = self.z
#        rho_c = 1 # some fiducial central density
#        r_cut = 2.1 #kpc
#        r0 = 0.075 #kpc
#        alpha = 1.8
#        q = 0.5
#        disk_density = rho_c*np.exp(-r/rh)*np.exp(-np.abs(z)/zh) 
#        bulge_density = rho_c*(np.exp(-(r/r_cut)**2)/(1+np.sqrt(r**2 + (z/q)**2)/r0)**alpha)
#        DWD_density = disk_density + bulge_density
#        ## use stored grid to convert density to power and filter nearby resolved DWDs
#        DWD_unresolved_powers = DWD_density*self.dist_adj
#        ## Bin
#        DWD_FG_mapG = np.bincount(self.pixels,weights=DWD_unresolved_powers.flatten(),minlength=hp.nside2npix(2*self.params['nside']))
#        ## Transform into the ecliptic
#        DWD_FG_map = self.rGE.rotate_map_pixel(DWD_FG_mapG)
#        
#        return DWD_FG_map
#        
#    
#    def breivik2020_log_prior(self,theta,bounds=np.array([[2,4],[0.05,2]])):
#        '''
#        Prior for the breivik2020 model. Uniform on user-specified bounds in kpc.
#        Default bounds are reasonable for the Milky Way.
#        
#        Arguments:
#            theta (array)       : [rh,zh], Breivik+2020 radial and vertical scale height
#            bounds (array)      : [[rhmin,rhmax],[zhmin,zhmax]]
#    
#        Returns:
#            logprior (float)              : log prior of theta = {rh,zh}
#        '''
#        ## unpack theta
#        rh,zh = theta
#        if rh < bounds[0,0] or rh > bounds[0,1]:
#            return -np.inf
#        elif zh < bounds[1,0] or zh > bounds[1,1]:
#            return -np.inf
#        else:
#            return 0
#    
#    def breivik2020_log_likelihood(self,theta,post_dist):
#        '''
#        Likelihood of theta = {rh,zh} given set of BLIP alm posteriors.
#        NOTE: Assumes blm posteriors (as transformed to alms) are normal and only uses their moments to determine the likelihood.
#        Assumes blm covariance is diagonal.
#        
#        Arguments:
#            rh (float)                   : Breivik+2020 radial scale height
#            zh (float)                   : Breivik+2020 vertical scale height
#            post_dist (scipy mv norm)    : Scipy multivariate normal created from the blm posteriors
#    
#        Returns:
#            loglike (float)              : log likelihood of theta = {rh,zh}
#        '''
#        ## unpack theta
#        rh,zh = theta
#        ## generate skymaps given rh and zh
##        start = time.time()
#        theta_map = self.breivik2020_mapmaker(rh,zh)
##        dur = time.time() - start
##        print('New elapse map gen time is {:0.2f} s.'.format(dur))
##        theta_map, log_theta_map = self.generate_galactic_foreground(rh,zh)
#        ## get corresponding blm values
#        theta_sph = self.skymap_pix2sph(theta_map)
#        theta_blm = self.blm_decompose(theta_sph)
#        ## determine log likelihood
#        loglike = post_dist.logpdf(theta_blm)
#        
#        return loglike
#    
#    def breivik2020_log_prob(self,theta,post_dist,bounds=np.array([[2,4],[0,2]])):
#        '''
#        Log probability for the Breivik+2020 model. 
#        Prior is uniform on user-specified bounds in kpc; default bounds are reasonable for the Milky Way.
#        Likelihood of theta = {rh,zh} given set of BLIP alm posteriors.
#        NOTE: Assumes blm posteriors (as transformed to alms) are normal and only uses their moments to determine the likelihood.
#        Assumes blm covariance is diagonal.
#        
#        Arguments:
#            theta (array)       : [rh,zh], Breivik+2020 radial and vertical scale height
#            post_dist (scipy mv norm)    : Scipy multivariate normal created from the blm posteriors
#            bounds (array)      : [[rhmin,rhmax],[zhmin,zhmax]]
#    
#        Returns:
#            logp (float)              : log posterior probability of theta = {rh,zh}
#        '''
#        ## check prior
#        logprior = self.breivik2020_log_prior(theta,bounds)
#        if not np.isfinite(logprior):
#            return -np.inf
#        ## get likelihood
#        loglike = self.breivik2020_log_likelihood(theta,post_dist)
#        
#        return logprior+loglike
#    
#    def hierarchical_sampler(self,model='breivik2020',Nwalkers=50,Nsamples=10000,Nburn=1000,rng=None,Nthread=1):
#        '''
#        Function to perform hierarchical sampling to constrain a spatial model of choice.
#        '''
#        ## set up rng
#        if rng is None:
#            rng = np.random.default_rng()
#        elif type(rng) is int:
#            rng = np.random.default_rng(rng)
#        elif type(rng) is not np.random._generator.Generator:
#            raise TypeError("Invalid specification of the RNG. Can by a numpy default_rng() object, a seed, or None")
#        
#        ## for now, only option will be the Breivik+2020 model
#        if model == 'breivik2020':
#            print("Post-processing with spatial model: Breivik+ (2020). Loading posterior samples and parameterizing...")
#            ## load posterior samples and process
#            post = np.loadtxt(self.rundir + "/post_samples.txt")
#            post_dist = self.post2dist(post)
#            ## Ndim is 2 {rh,zh}
#            Ndim = 2
#            ## intialize grid
#            self.init_breivik2020_grid()
#            ## set up uniform priors on [[rmin,rmax],[zmin,zmax]]
#            bounds=np.array([[2,4],[0,2]])
#            theta0 = np.array([rng.uniform(low=bounds[0,0],high=bounds[0,1],size=Nwalkers),rng.uniform(low=bounds[1,0],high=bounds[1,1],size=Nwalkers)]).T
#            logprob = self.breivik2020_log_prob
#            additional_args = (post_dist,bounds)
#        else:
#            raise TypeError("Unknown model. Currently supported models: 'breivik2020'.")
#        
#        ## create sampler
#        print("Generating Emcee sampler...")
#        if Nthread > 1:
#            with Pool(Nthread) as pool:
#                sampler = emcee.EnsembleSampler(Nwalkers,Ndim,logprob,args=additional_args,pool=pool)
#                ## burn-in
#                print("Performing {} samples of burn-in...".format(Nburn))
#                start = time.time()
#                pos, prob, state = sampler.run_mcmc(theta0,Nburn)
#                dur = time.time() - start
#                print('Time elapsed for burn: {:0.2f} s.'.format(dur))
#                ## run
#                print("Performing hierarchical sampling...")
#                sampler.reset()
#                start = time.time()
#                pos, prob, state = sampler.run_mcmc(pos,Nsamples)
#                dur = time.time() - start
#                print('Time elapsed for sampling: {:0.2f} s.'.format(dur))
#                pool.close()
#                pool.join()
#        else:
#            sampler = emcee.EnsembleSampler(Nwalkers,Ndim,logprob,args=additional_args)
#            ## burn-in
#            print("Performing {} samples of burn-in...".format(Nburn))
#            start = time.time()
#            pos, prob, state = sampler.run_mcmc(theta0,Nburn)
#            dur = time.time() - start
#            print('Time elapsed for burn: {:0.2f} s.'.format(dur))
#            ## run
#            print("Performing hierarchical sampling...")
#            sampler.reset()
#            start = time.time()
#            pos, prob, state = sampler.run_mcmc(pos,Nsamples)
#            dur = time.time() - start
#            print('Time elapsed for sampling: {:0.2f} s.'.format(dur))
#            
#        return sampler
    
    
    
    
    
    
    
    
    