import numpy as np
from dynesty import NestedSampler
from dynesty.utils import resample_equal


class dynesty_engine():

    '''
    Class for interfacing with dynesty sampler. This method also contains the
    priors definition for all models written to work with the dynesty sampler.
    '''


    @classmethod
    def define_engine(cls, lisaobj, params, nlive, randst):


        # create the nested sampler objects
        if params['modeltype']=='isgwb':

            print("Doing an isotropic stochastic analysis...")
            parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$', r'$\alpha$', r'$\log_{10} (\Omega_0)$']
            npar = len(parameters)

            engine = NestedSampler(lisaobj.isgwb_log_likelihood, cls.isgwb_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, rstate = randst)

        elif params['modeltype']=='sph_sgwb':

            print("Doing a spherical harmonic stochastic analysis ...")

            # add the basic parameters first
            parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$', r'$\alpha$', r'$\log_{10} (\Omega_0)$']

            # list for imposing periodic boundary conditions on phase variables
            periodic_bc = []        

            # add the blms
            for lval in range(1, params['lmax'] + 1):
                for mval in range(lval + 1):

                    if mval == 0:
                        parameters.append(r'$b_{' + str(lval) + str(mval) + '}$' )
                    else:
                        parameters.append(r'$|b_{' + str(lval) + str(mval) + '}|$' )
                        parameters.append(r'$\phi_{' + str(lval) + str(mval) + '}$' )
                        
                        # keep track of phase variable positions
                        periodic_bc.append(len(parameters) - 1)

            ## RM is line later.
            # parameters.append(r'$|b_{' + str(1) + str(1) + '}|$' )
            # parameters.append(r'$\phi_{' + str(1) + str(1) + '}$' )
            npar = len(parameters)

            engine = NestedSampler(lisaobj.sph_log_likelihood, cls.sph_prior, \
                    npar, bound='multi', sample='rwalk', nlive=nlive, rstate = randst, periodic=periodic_bc)

        elif params['modeltype']=='noise_only':

            print("Doing an instrumental noise only analysis ...")
            parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$']
            npar = len(parameters)

            engine = NestedSampler(lisaobj.instr_log_likelihood,  cls.instr_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, rstate = randst)

        elif params['modeltype'] =='isgwb_only':

            print("Doing an isgwb signal only analysis ...")
            parameters = [r'$\alpha$', r'$\log_{10} (\Omega_0)$']
            npar = len(parameters)

            engine = NestedSampler(lisaobj.isgwb_only_log_likelihood, cls.isgwb_only_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, rstate = randst)

        else:
            raise ValueError('Unknown recovery model selected')

        # print npar
        print("npar = " + str(npar))

        return engine, parameters

    @staticmethod
    def run_engine(engine):


       # -------------------- Run nested sampler ---------------------------
        engine.run_nested(dlogz=0.5,print_progress=True )

        # re-scale weights to have a maximum of one
        res = engine.results
        weights = np.exp(res['logwt'] - res['logz'][-1])
        weights[-1] = 1 - np.sum(weights[0:-1])

        post_samples = resample_equal(res.samples, weights)

        # Pull the evidence and the evidence error
        logz = res['logz']
        logzerr = res['logzerr']


        return post_samples, logz, logzerr

    @staticmethod
    def instr_prior(theta):


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

        return (log_Np, log_Na)

    @staticmethod
    def isgwb_prior(theta):


        '''
        Prior function for an isotropic stochastic backgound analysis.

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
        log_Np, log_Na, alpha, log_omega0 = theta

        # Transform to actual priors
        alpha       =  10*alpha-5
        log_omega0  = -10*log_omega0 - 4
        log_Np      = -5*log_Np - 39
        log_Na      = -5*log_Na - 46

        return (log_Np, log_Na, alpha, log_omega0)

    @staticmethod
    def sph_prior(theta):

        '''
        Prior for a power spectra based spherical harmonic anisotropic analysis

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element is always alpha and the last two are always Np and Na
        '''

        # The first two are the priors on the position and acc noise terms.
        log_Np = -4*theta[0] - 39
        log_Na = -4*theta[1] - 46

        # Prior on alpha, and omega_0
        alpha = 8*theta[2] - 4
        log_omega0  = -6*theta[3] - 5

        # Calculate lmax from the size of theta blm arrays. The shape is
        # given by size = (lmax + 1)**2 - 1. The '-1' is because b00 is
        # an independent parameter
        lmax = np.sqrt( theta[4:].size + 1 ) - 1

        if lmax.is_integer():
            lmax = int(lmax)
        else:
            raise ValueError('Illegitimate theta size passed to the spherical harmonic prior')

        # The rest of the priors define the blm parameter space
        blm_theta = []

        ## counter for the rest of theta
        cnt = 4

        for lval in range(1, lmax + 1):
            for mval in range(lval + 1):

                if mval == 0:
                    blm_theta.append(6*theta[cnt] - 3)
                    cnt = cnt + 1
                else:

                    # prior on real and imaginary parts
                    # blm_theta.append(6*theta[cnt] - 3)
                    # blm_theta.append(6*theta[cnt + 1] - 3)

                    ## prior on amplitude, phase
                    blm_theta.append(3* theta[cnt])
                    blm_theta.append(2*np.pi*theta[cnt+1] - np.pi)
                    cnt = cnt + 2

        # rm these three lines later.
        # blm_theta.append(theta[4])
        # blm_theta.append(2*np.pi*theta[5] - np.pi)

        theta = [log_Np, log_Na, alpha, log_omega0] + blm_theta

        return theta

    @staticmethod
    def isgwb_only_prior(self, theta):


        '''
        Prior function for an isotropic stochastic backgound analysis.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref

        '''

        # Unpack: Theta is defined in the unit cube
        alpha, log_omega0  = theta

        # Transform to actual priors
        alpha       = 10*alpha-5
        log_omega0  = -10*log_omega0 - 4

        return (alpha, log_omega0)
















