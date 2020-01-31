import numpy as np

@profile
def isgwb_mich_strain_response(self, f0):

        '''
        Calculate the detector transfer function functions to an isotropic SGWB non-polarized using basic michelson
        channels. Note that since this is the response to an isotropic background, the response function is integrated
        over sky direction and averaged over polarozation. The angular integral is a linear and rectangular in the
        cos(theta) and phi space.  Note also that f0 is (pi*L*f)/c and is input as an array. The response function is given
        for the strain of the signal rather than the power

        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

    

        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization.
        '''

        pi_val = np.pi
        tt = np.linspace(-1, 1, 200)
        ppi = np.linspace(0, 2*pi_val, 200, endpoint=False)

        [ct, phi] = np.meshgrid(tt,ppi)
        ct = ct.astype('float16')
        phi = phi.astype('float16')
        dct = ct[0, 1] - ct[0,0]
        dphi = phi[1,0] - phi[0,0]

        npix = ct.size
        norm = np.sqrt(0.5/npix)

        ## udir is just u.r, where r is the directional vector
        udir = (np.sqrt(1-ct**2) * np.sin(phi + pi_val/6)).astype('float16')
        vdir = (np.sqrt(1-ct**2) * np.sin(phi - pi_val/6)).astype('float16')
        wdir = vdir - udir

        # Initlize arrays for the detector reponse
        R1 = np.zeros((f0.size, 2), dtype='c16')
        R2 = np.zeros((f0.size, 2), dtype='c16')
        R3 = np.zeros((f0.size, 2), dtype='c16')


        ## response function u x u : eplus
        ##  Fplus_u = (u x u):eplus

        Fplus_u   = (1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - \
                        np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2)  + \
                            0.5*((np.cos(phi))**2 - ct**2))
        
        Fplus_v   = (1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 + \
                        np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2) + \
                            0.5*((np.cos(phi))**2 - ct**2))

        Fplus_w   = (1 - (1+ct**2)*(np.cos(phi))**2)

        ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross
        Fcross_u  = - ct * (np.sin(2*phi + pi_val/3))
        Fcross_v  = - ct * (np.sin(2*phi - pi_val/3))
        Fcross_w   = ct*np.sin(2*phi)
        
        uminus_1 = np.multiply.outer(f0, (1-udir)/pi_val).astype('float16')
        vminus_1 = np.multiply.outer(f0, (1-vdir)/pi_val).astype('float16')
        wminus_1 = np.multiply.outer(f0, (1-wdir)/pi_val).astype('float16')

        uplus_1 = np.multiply.outer(f0, (1+udir)/pi_val).astype('float16')
        vplus_1 = np.multiply.outer(f0, (1+vdir)/pi_val).astype('float16')
        wplus_1 = np.multiply.outer(f0, (1+wdir)/pi_val).astype('float16')


        uminus_3 = np.multiply.outer(f0, (3-udir)).astype('float16')
        vminus_3 = np.multiply.outer(f0, (3-vdir)).astype('float16')
        wminus_3 = np.multiply.outer(f0, (3-wdir)).astype('float16')

        uplus_3 = np.multiply.outer(f0, (3+udir)).astype('float16')
        vplus_3 = np.multiply.outer(f0, (3+vdir)).astype('float16')
        wplus_3 = np.multiply.outer(f0, (3+wdir)).astype('float16')

        gammaU_plus = 1/2 * ( np.sinc(uminus_1)*np.exp(-1j*uplus_3) + np.sinc(uplus_1)*np.exp(-1j*pi_val*uplus_1) ).astype('c16')
        gammaV_plus = 1/2 * ( np.sinc(vminus_1)*np.exp(-1j*vplus_3) + np.sinc(vplus_1)*np.exp(-1j*pi_val*vplus_1) ).astype('c16')
        gammaW_plus = 1/2 * ( np.sinc(wminus_1)*np.exp(-1j*wplus_3) + np.sinc(wplus_1)*np.exp(-1j*pi_val*wplus_1) ).astype('c16')

        gammaU_minus = 1/2 * ( np.sinc(uplus_1)*np.exp(-1j*uminus_3) + np.sinc(uminus_1)*np.exp(-1j*pi_val*uminus_1) ).astype('c16')
        gammaV_minus = 1/2 * ( np.sinc(vplus_1)*np.exp(-1j*vminus_3) + np.sinc(vminus_1)*np.exp(-1j*pi_val*vminus_1) ).astype('c16')
        gammaW_minus = 1/2 * ( np.sinc(wplus_1)*np.exp(-1j*wminus_3) + np.sinc(wminus_1)*np.exp(-1j*pi_val*wminus_1) ).astype('c16')


        ## Michelson antenna patterns: Calculate Fplus
        Fplus1 = 0.5*(Fplus_u[None,:,:] * gammaU_plus  - Fplus_v[None,:,:] * gammaV_plus)
        Fplus2 = 0.5*(Fplus_w[None,:,:] * gammaW_plus  - Fplus_u[None,:,:] * gammaU_minus) * np.exp(2j * np.multiply.outer(f0, udir).astype('c16'))
        Fplus3 = 0.5*(Fplus_v[None,:,:] * gammaV_minus - Fplus_w[None,:,:] * gammaW_minus)*np.exp(2j * np.multiply.outer(f0, vdir).astype('c16'))

        ## Michelson antenna patterns: Calculate Fcross
        Fcross1 = 0.5*(Fcross_u[None,:,:] * gammaU_plus  - Fcross_v[None,:,:] * gammaV_plus).astype('c16')
        Fcross2 = 0.5*(Fcross_w[None,:,:] * gammaW_plus  - Fcross_u[None,:,:] * gammaU_minus) * np.exp(2j * np.multiply.outer(f0, udir).astype('c16'))
        Fcross3 = 0.5*(Fcross_v[None,:,:] * gammaV_minus - Fcross_w[None,:,:] * gammaW_minus) * np.exp(2j * np.multiply.outer(f0, vdir).astype('c16'))

        del gammaU_plus, gammaV_plus, gammaW_plus, gammaU_minus, gammaV_minus, gammaW_minus

        rand_plus = (np.random.normal(size=Fplus1.shape) + 1j* np.random.normal(size=Fplus1.shape)).astype('c16')
        rand_cross = (np.random.normal(size=Fplus1.shape) + 1j* np.random.normal(size=Fplus1.shape)).astype('c16')

        ## Detector response summed over polarization and integrated over sky direction
        R1[:, 0], R1[:, 1] = norm*(Fplus1*rand_plus).sum(axis=(1, 2)), norm*(Fcross1*rand_cross).sum(axis=(1, 2)) 
        R2[:, 0], R2[:, 1] = norm*(Fplus2*rand_plus).sum(axis=(1, 2)), norm*(Fcross2*rand_cross).sum(axis=(1, 2)) 
        R3[:, 0], R3[:, 1] = norm*(Fplus3*rand_plus).sum(axis=(1, 2)), norm*(Fcross3*rand_cross).sum(axis=(1, 2))

        return R1, R2, R3