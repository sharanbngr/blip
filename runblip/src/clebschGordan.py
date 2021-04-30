import numpy as np
from healpy import Alm
from sympy.physics.quantum.cg import CG
from collections import OrderedDict


class clebschGordan():

    '''
    Class with methods for manipulating clebsch-gordon coeffcients.
    '''

    def __init__(self):
        self.blmax = self.params['lmax']
        self.almax = 2*self.blmax

        ## size of arrays: for blms its only non-negative m values but for alms it is all of them
        self.alm_size = (self.almax + 1)**2
        self.blm_size = Alm.getsize(self.blmax)

        ## calculate and store beta
        self.calc_beta()

        ## calculate and store the output of the idxtoalm method for blmax.
        ## This will be used many times for the spherical harmonic likelihood

        ## Array of blm values for both +ve and -ve indices
        self.bl_idx = np.zeros(2*self.blm_size - self.blmax - 1, dtype='int')
        self.bm_idx = np.zeros(2*self.blm_size - self.blmax - 1, dtype='int')


        for ii in range(self.bl_idx.size):

            #lval, mval = Alm.getlm(blmax, jj)
            self.bl_idx[ii], self.bm_idx[ii] = self.idxtoalm(self.blmax, ii)


    def idxtoalm(self, lmax, ii):

        '''
        index --> (l, m) function which works for negetive indices too
        '''

        alm_size = Alm.getsize(lmax)

        if ii >= (2*alm_size - lmax - 1):
            raise ValueError('Index larger than acceptable')
        elif ii < alm_size:
            l, m = Alm.getlm(lmax, ii)
        else:
            l, m = Alm.getlm(lmax, ii - alm_size + lmax + 1)

            if m ==0:
                raise ValueError('Something wrong with ind -> (l, m) conversion')
            else:
                m = -m

        return l, m


    def calc_beta(self):

        '''
        Method to calculate beta array to convert from blm to alm
        '''

        ## initialize beta array
        beta_vals = np.zeros((self.alm_size, 2*self.blm_size - self.blmax - 1, 2*self.blm_size - self.blmax - 1))

        for ii in range(beta_vals.shape[0]):
            for jj in range(beta_vals.shape[1]):
                for kk in range(beta_vals.shape[2]):

                    l1, m1 = self.idxtoalm(self.blmax, jj)
                    l2, m2 = self.idxtoalm(self.blmax, kk)
                    L, M = self.idxtoalm(self.almax, ii)

                    ## clebs gordon coeffcients
                    cg0 = (CG(l1, 0, l2, 0, L, 0).doit()).evalf()
                    cg1 = (CG(l1, m1, l2, m2, L, M).doit()).evalf()

                    beta_vals[ii, jj, kk] =  np.sqrt( (2*l1 + 1) * (2*l2 + 1) / ((4*np.pi) * (2*L + 1) )) * cg0 * cg1


        self.beta_vals = beta_vals

    def calc_blm_full(self, blms_in):

        '''
        Convert samples in blm space to blm complex values including negetive m vals

        Input:  blms ordered dictionary
        Ouput:  blms_full, list including blms with negative m vals

        '''

        ## Array of blm values for both +ve and -ve indices
        blms_full = np.zeros(2*self.blm_size - self.blmax - 1, dtype='complex')


        for jj in range(blms_full.size):

            lval, mval = self.bl_idx[jj], self.bm_idx[jj]

            if mval >= 0:
                blms_full[jj] = blms_in[Alm.getidx(self.blmax, lval, mval)]

            elif mval < 0:
                mval = -mval
                blms_full[jj] = (-1)**mval *  np.conj(blms_in[Alm.getidx(self.blmax, lval, mval)])

        return blms_full

    def blm_2_alm(self, blms_in):

        '''
        Convert complex blm values to alm complex values. This will contain both -ve m values too in the standard order
        '''

        if blms_in.size != self.blm_size:
            raise ValueError('The size of the input blm array does not match the size defined by lmax ')

        ## convert blm array into a full blm array with -m values too
        blm_full = self.calc_blm_full(blms_in)

        alm_vals = np.einsum('ijk,j,k', self.beta_vals, blm_full, blm_full)

        return alm_vals


    def blm_params_2_blms(self, blm_params):

        '''
        convert blm parameter values where amplitudes and phases are seperate to complex
        blm values.
        '''

        ## initialize blm_vals array
        blm_vals = np.zeros(self.blm_size, dtype='complex')

        ## this is b00, alsways set to 1
        blm_vals[0] = 1

        ## counter for blm_vals
        cnt = 0

        for lval in range(1, self.params['lmax'] + 1):
            for mval in range(lval + 1):

                idx = Alm.getidx(self.blmax, lval, mval)

                if mval == 0:
                    blm_vals[idx] = blm_params[cnt]
                    cnt = cnt + 1
                else:
                    #blm_vals[idx] = blm_params[cnt] + 1j * blm_params[cnt+1]
                    ## prior on amplitude, phase
                    blm_vals[idx] = blm_params[cnt] * np.exp(1j * blm_params[cnt+1])
                    cnt = cnt + 2

        return blm_vals



