import numpy as np
from scipy.special import lpmn, sph_harm
import numpy.linalg as LA
import types
import healpy as hp
from healpy import Alm
from src.clebschGordan import clebschGordan

class sph_geometry(clebschGordan):

    def __init__(self):
        clebschGordan.__init__(self)


    def asgwb_mich_response(self, f0, tsegmid):

        '''
        Calculate the Antenna pattern/ detector transfer function functions to acSGWB using michelson channels,
        and using a spherical harmonic decomposition. Note that the response function to power is integrated over
        sky direction with the appropriate spherical harmonics, and averaged over polarozation. The angular
        integral is numerically done by divvying up the sky into a healpix grid.

        Note that f0 is (pi*L*f)/c and is input as an array

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged
            over polarization. The arrays are 2-d, one direction corresponds to frequency and the other to the l coeffcient.
        '''

        print('calculating the anisotropic responses')

        ## array size of almax
        alm_size = (self.almax + 1)**2

        npix = hp.nside2npix(self.params['nside'])

        # Array of pixel indices
        pix_idx  = np.arange(npix)

        #Angular coordinates of pixel indcides
        theta, phi = hp.pix2ang(self.params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)

        # Area of each pixel in sq.radians
        dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])

        # Create 2D array of (x,y,z) unit vectors for every sky direction.
        omegahat = np.array([np.sqrt(1-ctheta**2)*np.cos(phi),np.sqrt(1-ctheta**2)*np.sin(phi),ctheta])

        # Call lisa_orbits to compute satellite positions at the midpoint of each time segment
        rs1, rs2, rs3 = self.lisa_orbits(tsegmid)

        ## Calculate directional unit vector dot products
        ## Dimensions of udir is time-segs x sky-pixels
        udir = np.einsum('ij,ik',(rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :],omegahat)
        vdir = np.einsum('ij,ik',(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :],omegahat)
        wdir = np.einsum('ij,ik',(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :],omegahat)

        ## NB --    An attempt to directly adapt e.g. (u o u):e+ as implicit tensor calculations
        ##             as opposed to the explicit forms we've previously used. '''
        mhat = np.array([np.sin(phi),-np.cos(phi),np.zeros(len(phi))])
        nhat = np.array([np.cos(phi)*ctheta,np.sin(phi)*ctheta,-np.sqrt(1-ctheta**2)])

        # 1/2 u x u : eplus. These depend only on geometry so they only have a time and directionality dependence and not of frequency
        Fplus_u = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :], (rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) - np.einsum("ik,jk -> ijk",nhat,nhat))

        Fplus_v = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :],(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) - np.einsum("ik,jk -> ijk",nhat,nhat))

        Fplus_w = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :],(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) - np.einsum("ik,jk -> ijk",nhat,nhat))

        # 1/2 u x u : ecross
        Fcross_u = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :],(rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) + np.einsum("ik,jk -> ijk",nhat,nhat))

        Fcross_v = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :],(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) + np.einsum("ik,jk -> ijk",nhat,nhat))

        Fcross_w = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :],(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) + np.einsum("ik,jk -> ijk",nhat,nhat))


        # Initlize arrays for the detector reponse
        R1 = np.zeros((f0.size, tsegmid.size, alm_size), dtype='complex')
        R2 = np.zeros((f0.size, tsegmid.size, alm_size), dtype='complex')
        R3 = np.zeros((f0.size, tsegmid.size, alm_size), dtype='complex')
        R12 = np.zeros((f0.size, tsegmid.size, alm_size), dtype='complex')
        R13 = np.zeros((f0.size, tsegmid.size, alm_size), dtype='complex')
        R23 = np.zeros((f0.size, tsegmid.size, alm_size), dtype='complex')
        R21 = np.zeros((f0.size, tsegmid.size, alm_size), dtype='complex')
        R31 = np.zeros((f0.size, tsegmid.size, alm_size), dtype='complex')
        R32 = np.zeros((f0.size, tsegmid.size, alm_size), dtype='complex')


        ## initalize array for Ylms
        Ylms = np.zeros((npix, alm_size ), dtype='complex')

        ## Get the spherical harmonics
        for ii in range(alm_size):
            lval, mval = self.idxtoalm(self.almax, ii)
            Ylms[:, ii] = sph_harm(mval, lval, phi, theta)


        # Calculate the detector response for each frequency
        for ii in range(0, f0.size):

            # Calculate GW transfer function for the michelson channels
            gammaU_plus    =    1/2 * (np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(3+udir)) + \
                             np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(1+udir)))

            gammaV_plus    =    1/2 * (np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(3+vdir)) + \
                             np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(1+vdir)))

            gammaW_plus    =    1/2 * (np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(3+wdir)) + \
                             np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(1+wdir)))


            # Calculate GW transfer function for the michelson channels
            gammaU_minus    =    1/2 * (np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(3 - udir)) + \
                             np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(1 - udir)))

            gammaV_minus    =    1/2 * (np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - vdir)) + \
                             np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - vdir)))

            gammaW_minus    =    1/2 * (np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - wdir)) + \
                             np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - wdir)))

            ## Calculate Fplus
            Fplus1 = 0.5*(Fplus_u*gammaU_plus - Fplus_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3))
            Fplus2 = 0.5*(Fplus_w*gammaW_plus - Fplus_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3))
            Fplus3 = 0.5*(Fplus_v*gammaV_minus - Fplus_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3))

            ## Calculate Fcross
            Fcross1 = 0.5*(Fcross_u*gammaU_plus  - Fcross_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3))
            Fcross2 = 0.5*(Fcross_w*gammaW_plus  - Fcross_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3))
            Fcross3 = 0.5*(Fcross_v*gammaV_minus - Fcross_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3))


            ## Detector response for the TDI Channels, summed over polarization
            ## and integrated over sky direction
            F1 = (np.absolute(Fplus1))**2 + (np.absolute(Fcross1))**2
            F2 = (np.absolute(Fplus2))**2 + (np.absolute(Fcross2))**2
            F3 = (np.absolute(Fplus3))**2 + (np.absolute(Fcross3))**2
            F12 = np.conj(Fplus1)*Fplus2 + np.conj(Fcross1)*Fcross2
            F13 = np.conj(Fplus1)*Fplus3 + np.conj(Fcross1)*Fcross3
            F23 = np.conj(Fplus2)*Fplus3 + np.conj(Fcross2)*Fcross3

            R1[ii, :, :] = dOmega/(8*np.pi)*np.einsum('ij, jk', F1, Ylms)
            R2[ii, :, :] = dOmega/(8*np.pi)*np.einsum('ij, jk', F2, Ylms)
            R3[ii, :, :] = dOmega/(8*np.pi)*np.einsum('ij, jk', F3, Ylms)
            R12[ii, :, :] = dOmega/(8*np.pi)*np.einsum('ij, jk', F12, Ylms)
            R13[ii, :, :] = dOmega/(8*np.pi)*np.einsum('ij, jk', F13, Ylms)
            R23[ii, :, :] = dOmega/(8*np.pi)*np.einsum('ij, jk', F23, Ylms)
            R21[ii, :, :] = dOmega/(8*np.pi)*np.einsum('ij, jk', np.conj(F12), Ylms)
            R31[ii, :, :] = dOmega/(8*np.pi)*np.einsum('ij, jk', np.conj(F13), Ylms)
            R32[ii, :, :] = dOmega/(8*np.pi)*np.einsum('ij, jk', np.conj(F23), Ylms)


        response_mat = np.array([ [R1, R12, R13] , [R21, R2, R23], [R31, R32, R3] ])

        return response_mat


    def asgwb_xyz_response(self, f0, tsegmid):

        '''
        Calculate the Antenna pattern/ detector transfer function functions to acSGWB using X,Y,Z TDI channels,
        and using a spherical harmonic decomposition. Note that the response function to power is integrated over
        sky direction with the appropriate spherical harmonics, and averaged over polarozation. The angular
        integral is numerically done by divvying up the sky into a healpix grid.

        Note that f0 is (pi*L*f)/c and is input as an array

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)



        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged
            over polarization. The arrays are 2-d, one direction corresponds to frequency and the other to the l coeffcient.
        '''

        mich_response_mat = self.asgwb_mich_response(f0, tsegmid)
        xyz_response_mat = 4 * mich_response_mat * (np.sin(2*f0[None, None, :, None, None]))**2

        return xyz_response_mat


    def asgwb_aet_response(self, f0, tsegmid):

        '''
        Calculate the Antenna pattern/ detector transfer function functions to acSGWB using X,Y,Z TDI channels,
        and using a spherical harmonic decomposition. Note that the response function to power is integrated over
        sky direction with the appropriate spherical harmonics, and averaged over polarozation. The angular
        integral is numerically done by divvying up the sky into a healpix grid.

        Note that f0 is (pi*L*f)/c and is input as an array



        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)



        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged
            over polarization. The arrays are 2-d, one direction corresponds to frequency and the other to the l coeffcient.
        '''

        xyz_response_mat = self.asgwb_xyz_response(f0, tsegmid)

        ## Upnack xyz matrix to make assembling the aet matrix easier
        RXX, RYY, RZZ = xyz_response_mat[0, 0], xyz_response_mat[1, 1], xyz_response_mat[2, 2]
        RXY, RXZ, RYZ = xyz_response_mat[0, 1], xyz_response_mat[0, 2], xyz_response_mat[1, 2]

        ## construct AET matrix elements
        RAA = (1/9) * (4*RXX + RYY + RZZ - 2*RXY - 2*np.conj(RXY) - 2*RXZ - 2*np.conj(RXZ) + \
                        RYZ  + np.conj(RYZ))

        REE = (1/3) * (RZZ + RYY - RYZ - np.conj(RYZ))

        RTT = (1/9) * (RXX + RYY + RZZ + RXY + np.conj(RXY) + RXZ + np.conj(RXZ) + RYZ + np.conj(RYZ))

        RAE = (1/(3*np.sqrt(3))) * (RYY - RZZ - RYZ + np.conj(RYZ) + 2*RXZ - 2*RXY)

        RAT = (1/9) * (2*RXX - RYY - RZZ + 2*RXY - np.conj(RXY) + 2*RXZ - np.conj(RXZ) - RYZ - np.conj(RYZ))

        RET = (1/(3*np.sqrt(3))) * (RZZ - RYY - RYZ + np.conj(RYZ) + np.conj(RXZ) - np.conj(RXY))

        aet_response_mat = np.array([ [RAA, RAE, RAT] , \
                                    [np.conj(RAE), REE, RET], \
                                    [np.conj(RAT), np.conj(RET), RTT] ])

        return aet_response_mat


