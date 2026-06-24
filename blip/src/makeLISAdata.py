import os

import numpy as np
import jax
import chex
import jax.numpy as jnp
from jax import config as jaxconfig

import h5py
from tqdm import tqdm

from blip.src.models import Injection

jaxconfig.update("jax_enable_x64", True)


def get_simulation_tf_grid(dur, tstart, fs, tsplice):
    """Compute time-frequency grid parameters used for simulation.

    Parameters
    ----------
    dur : float
        total data duration in seconds, as set by the :confval:`duration` parameter.
    tstart : float
        starting time in seconds.
    fs : float
        data sampling rate in Hz. Inverse of dt.
    tsplice : float
        Splice segment length in seconds.

    Returns
    -------
    int
        nsplice: number of splices
    array (nsplice,)
        tsegmid: array of mid-splice times
    int
        Npersplice: number of time-domain points in splice segment

    See also
    --------
    :ref:`devguide-timefreq`
    """
    # the segments to be splices are half-overlapping
    nsplice = 2 * int(dur / tsplice) + 1
    tsegmid = tstart + (tsplice / 2.0) * np.arange(nsplice) + (tsplice / 2.0)
    Npersplice = int(fs * tsplice)

    return tsplice, nsplice, tsegmid, Npersplice


def get_simulation_length(dur, tstart, fs, tsplice):
    """Compute simulation array length.

    This is the size of the array produced by :meth:`LISAdata.add_sgwb_data`.

    Parameters
    ----------
    dur : float
        total data duration in seconds, as set by the :confval:`duration` parameter.
    tstart : float
        starting time in seconds.
    fs : float
        data sampling rate in Hz. Inverse of dt.
    tsplice : float
        Splice segment length in seconds.

    Returns
    -------
    int
        length of simulation arrays.
    """
    _, nsplice, _, Npersplice = get_simulation_tf_grid(dur, tstart, fs, tsplice)
    halfNpersplice = int(0.5 * Npersplice)
    return Npersplice + (nsplice - 3) * halfNpersplice


def get_data_tf_grid(dur, seglen, fs):
    """Compute time-frequency grid parameters used for data analysis.

    Parameters
    ----------
    dur : float
        total data duration in seconds, as set by the :confval:`duration` parameter.
    seglen : float
        segment length set by :confval:`seglen`.
    fs : float
        data sampling rate in Hz. Inverse of dt.

    Returns
    -------
    int
        nsegs: number of (overlapping) segments
    int
        Nperseg: number of time-domain points in each segment

    See also
    --------
    :ref:`devguide-timefreq`
    """
    nsegs = int(np.floor(dur / seglen)) - 1
    Nperseg = int(fs * seglen)
    return nsegs, Nperseg


def get_data_length(dur, seglen, fs):
    """Compute data array length.

    The data is required to have this length, aligned with segment sizes, for use in
    :meth:`LISAdata.tser2fser`.

    Parameters
    ----------
    dur : float
        total data duration in seconds, as set by the :confval:`duration` parameter.
    seglen : float
        segment length set by :confval:`seglen`.
    fs : float
        data sampling rate in Hz. Inverse of dt.

    Returns
    -------
    int
        length of analysis data array.
    """
    nsegs, Nperseg = get_data_tf_grid(dur, seglen, fs)
    return nsegs * Nperseg


class LISAdata():
    '''
    Class for lisa data. Includes methods for generation of gaussian instrumental noise, and generation
    of isotropic stochastic background. Signal models should be added as methods here.
    '''

    def __init__(self, params, inj):
        self.params = params
        self.inj = inj
        self.armlength = 2.5e9 ## armlength in meters

        ## numpy/jax.numpy switch
        global xp
        backend = jax.default_backend()
        if backend == 'gpu':
            print("GPU detected; performing data simulation on GPU...")
            self.gpu = True
            xp = jnp
        elif backend == 'cpu':
            print("No GPU detected; performing data simulation on CPU...")
            self.gpu = False
            xp = np
        else:
            print("Warning: something fishy is afoot! JAX backend is neither CPU nor GPU. Defaulting to CPU; if you are trying to run BLIP on a TPU, don't!")
            self.gpu = False
            xp = np

        # the injection must exist before add_sgwb_data() is called.
        self.injection: Injection | None = None

    ## Method for reading frequency domain spectral data if given in an npz file
    def read_spectrum(self):

        '''
        Read an input frequency domain data file. Returns the fourier transform of the data from the three channels and an array of reference frequencyes

        Returns
        ---------

        rA, rE, rT, fdata   :   float

        '''

        if os.path.isfile(self.params['input_spectrum']) and not self.params['doPreProc']:
            print("loading freq domain data from input file")

            data = np.load(self.params['out_dir'] + '/' +self.params['input_spectrum'])
            r1    = data['r1']
            r2    = data['r2']
            r3    = data['r3']
            fdata = data['fdata']

            return r1, r2, r3, fdata

    def add_sgwb_data(self, injmodel, key, tbreak = 0.0):

        '''
        Function to simulate the SGWB time series, given a spectrum.
        
        Arguments
        --------------------
        injmodel (Injection component) : The injection component submodel.
        key (int) : Key for the numpy (or JAX) rng
        
        '''

        assert self.injection is not None
        assert isinstance(self.injection, Injection)

        N = self.injection.Npersplice
        halfN = int(0.5*N)

        ## compute the astrophysical spectrum
        injmodel_args = [injmodel.truevals[parameter] for parameter in injmodel.spectral_parameters]

        Sgw = injmodel.compute_Sgw(self.injection.frange,injmodel_args)

        injmodel.frozen_spectra = Sgw

        ## the spectrum of the frequecy domain gaussian for ifft
        norms = xp.sqrt(self.params['fs']*Sgw*N)/2

        ## index array for one segment
        t_arr = xp.arange(N)

        ## the window for splicing
        splice_win = xp.sin(xp.pi * t_arr/N)

        ## create rng, etc.
        if self.gpu:
            jax_key = jax.random.key(key)
        else:
            numpy_rng = np.random.default_rng(key)

        ## Loop over splice segments
        print("Simulating time-domain data for component '{}'...".format(injmodel.name))
        for ii in tqdm(range(self.injection.nsplice), desc=injmodel.name, unit="splice"):
            # Do Cholesky decomposition assuming certain symmetries
            L_cholesky = cholesky_symm(injmodel.response_mat[:, :, :, ii])
            L_cholesky = xp.moveaxis(L_cholesky, -1, 0)
            # multiply by ASD
            L_cholesky = norms[:, xp.newaxis, xp.newaxis] * L_cholesky

            ## generate standard normal complex data first
            if self.gpu:
                # Only the subkeys are used in the PRNG, and deleted to prevent reuse.
                # The main key is only split.
                jax_key, subkey1, subkey2 = jax.random.split(jax_key, 3)
                z_norm = jax.random.normal(
                    subkey1, shape=(self.injection.frange.size, 3)
                ) + 1j * jax.random.normal(
                    subkey2, shape=(self.injection.frange.size, 3)
                )
                del subkey1, subkey2
            else:
                z_norm = numpy_rng.normal(size=(self.injection.frange.size, 3)) + 1j * numpy_rng.normal(size=(self.injection.frange.size, 3))

            ## The data in z_norm is rescaled into z_scale using L_cholesky
            z_scale = xp.einsum('ijk, ikl -> ijl', L_cholesky, z_norm[:, :, None])[:, :, 0]

            ## The three channels : concatenate with norm at f = 0 to be zero
            zero_arr = xp.zeros(1)
            htilda1  = xp.concatenate([ zero_arr, z_scale[:, 0]])
            htilda2  = xp.concatenate([ zero_arr, z_scale[:, 1]])
            htilda3  = xp.concatenate([ zero_arr, z_scale[:, 2]])

            if ii == 0:
                # Take inverse fft to get time series data
                h1 = splice_win * xp.fft.irfft(htilda1, N)
                h2 = splice_win * xp.fft.irfft(htilda2, N)
                h3 = splice_win * xp.fft.irfft(htilda3, N)

            else:

                ## First append half-splice worth of zeros
                h1 = xp.append(h1, xp.zeros(halfN))
                h2 = xp.append(h2, xp.zeros(halfN))
                h3 = xp.append(h3, xp.zeros(halfN))

                ## Then add the new splice segment
                if self.gpu:
                    h1 = h1.at[-N:].set(h1[-N:] + splice_win * xp.fft.irfft(htilda1, N))
                    h2  = h2.at[-N:].set(h2[-N:] + splice_win * xp.fft.irfft(htilda2, N))
                    h3 = h3.at[-N:].set(h3[-N:] + splice_win * xp.fft.irfft(htilda3, N))
                else:
                    h1[-N:] = h1[-N:] + splice_win * xp.fft.irfft(htilda1, N)
                    h2[-N:] = h2[-N:] + splice_win * xp.fft.irfft(htilda2, N)
                    h3[-N:] = h3[-N:] + splice_win * xp.fft.irfft(htilda3, N)

        ## remove the first half and the last half splice.
        h1, h2, h3 = h1[halfN:-halfN], h2[halfN:-halfN], h3[halfN:-halfN]

        tarr = self.params['tstart'] + tbreak +  xp.arange(0, self.params['dur'], 1.0/self.params['fs'])

        return h1, h2, h3, tarr

    def process_external_data(self):
        """
        Read external data from params.datafile assuming it is in the
        file format specified by params.datafileformat.
        """

        if self.params["datafileformat"] == "mldc":
            self._process_mldc_data()
        elif self.params["datafileformat"] == "ldc":
            self._process_ldc_data()
        else:
            assert False, "Unreachable"

    def _process_ldc_data(self):
        # NOTE this needs to behave similarly to LISA.makedata() wrt side effects.
        # This method defines the attributes h1,h2,h3, r1,r2,r3, f0, timearray, fdata,
        # tsegstart, and tsegmid. It relies on self.tser2fser() to avoid duplicating
        # some of the logic, but it still needs to be manually kept in sync with
        # makedata().

        if self.params["tdi_lev"] != "xyz":
            # TODO
            raise NotImplementedError("cannot use AET or Michelson variables for now")
        if self.params["datadomain"] != "freq":
            # TODO when implementing time-domain, change _validate_ldc_data accordingly
            raise NotImplementedError("cannot yet deal with time-domain external data")
        if self.params["datatype"] != "doppler":
            print("WARNING: External data with datatype=strain: probably a mistake")

        filepath = os.path.abspath(self.params["datafile"])
        assert os.path.isfile(filepath), f"Not a file: {filepath}"
        tdi, attrs = _ldc_load_array(filepath, name=self.params["ldc_dataset"])

        # Make sure the numbers make sense
        tdi, dt, N = self._validate_ldc_data(tdi, attrs)

        # Convert Doppler to strain units if necessary
        cspeed = 3e8
        fstar = cspeed/(2*np.pi*self.armlength)
        strain_to_doppler = -2j*np.fft.rfftfreq(2*(tdi.shape[1]-1), dt)/fstar
        doppler_to_strain = np.divide(1, strain_to_doppler, where=(strain_to_doppler!=0))
        doppler_to_strain[0] = 0.0  # remove uninitialized value
        if self.params["datatype"] == "doppler":
            tdi *= doppler_to_strain[np.newaxis, :]

        # Change to time domain.
        # The 1/dt factor is necessary because LDC frequency series are discrete time
        # Fourier transforms (DTFTs): every rfft needs to be accompanied by a "*dt",
        # and every irfft by a "/dt".
        tdi = np.fft.irfft(tdi, axis=1) / dt

        # cut to required length
        tdi = tdi[:, :N]

        self.timearray = np.arange(N) * dt
        self.h1, self.h2, self.h3 = tdi[0, :], tdi[1, :], tdi[2, :]

        self.r1, self.r2, self.r3, self.fdata, self.tsegstart, self.tsegmid = (
            self.tser2fser(self.h1, self.h2, self.h3, self.timearray)
        )

        self.f0 = self.fdata/(2*fstar)

    def _validate_ldc_data(self, tdi, attrs):
        # We will accept the shape (3, nfreqs) only for now.
        if len(tdi.shape) != 2:
            raise ValueError(f"Expected TDI array with rank 2, got: {len(tdi.shape) = }")
        if not tdi.shape[0]==3:
            raise ValueError(f"Expected three TDI channels, got {tdi.shape = }")

        if self.params["datadomain"] == "freq":
            df = attrs.get("df")
            if df is None:
                raise ValueError(f"The array {self.params['ldc_dataset']} contains"
                    f"no 'df' attribute.\nIs it really in frequency domain? If not,"
                    f"change the parameter `datadomain` to 'time'."
            )
            if df <= 0:
                raise ValueError("df must be positive")
            Tobs = 1 / df
            N = 2 * (tdi.shape[1] - 1)  # same size assumed by np.fft.irfft
            dt = Tobs / N

        else:
            assert False, "unreachable, only freq domain accepted"

        if not np.isclose(1 / dt, self.params["fs"]):
            raise ValueError(
                f"The `fs` option was set to {self.params['fs']}, incompatible with input data where fs={1/dt}"
            )

        # find required size
        Nreq = int(self.params["dur"] / dt)
        if Nreq > N:
            raise ValueError(
                f"Requested duration {self.params['dur']} s, but the input data only has {Tobs} s"
            )

        return tdi, dt, Nreq

    def _process_mldc_data(self):
        h1, h2, h3, self.timearray = self._read_mldc_data()

        # Calculate other tdi combinations if necessary.
        if self.params['tdi_lev'] == 'aet':
            h1 = (1.0/3.0)*(2*h1 - h2 - h3)
            h2 = (1.0/np.sqrt(3.0))*(h3 - h2)
            h3 = (1.0/3.0)*(h1 + h2 + h3)

        # Generate lisa freq domain data from time domain data
        self.r1, self.r2, self.r3, self.fdata, self.tsegstart, self.tsegmid = self.tser2fser(h1, h2, h3, self.timearray)

        # Charactersitic frequency. Define f0
        cspeed = 3e8
        fstar = cspeed/(2*np.pi*self.armlength)
        self.f0 = self.fdata/(2*fstar)

        # Convert doppler data to strain if readfile datatype is doppler.
        if self.params['datatype'] == 'doppler':
            # This is needed to convert from doppler data to strain data.
            self.r1, self.r2, self.r3 = self.r1/(4*self.f0.reshape(self.f0.size, 1)), self.r2/(4*self.f0.reshape(self.f0.size, 1)), self.r3/(4*self.f0.reshape(self.f0.size, 1))

        elif self.params['datatype'] == 'strain':
            pass

    def tser2fser(self, h1, h2, h3, timearray):

        '''
        Convert time domain data to fourier domain and return ffts. The convention is that the
        the ffts are divided by the sampling frequency and corrected for windowing. A hann window
        is applied by default when moving to the fourier domain. The ffts are also normalized so that
        thier square gives the PSD.

        Parameters
        -----------
        h1, h2, h3 : float
            time series data for the three input channels

        timearray : float
            times corresponding to data in h1, h2, h3

        Returns
        ---------

        r1, r2, r3 : float
            frequency series data for the three input channels

        fdata : float
            Reference frequency series

        tsegstart : float
            Segmented time array giving segment start points

        tsegmid : float
            Segmented time array giving segment midpoints


        '''

        print ("Calculating fourier spectra... ")

        # precomputed during configuration parsing
        nsegs, Nperseg = self.params["nsegs"], self.params["Nperseg"]

        '''
        # Apply a cascading low pass filter
        b, a = sg.butter(2, 0.4*self.params['fs']/(self.params['fs']/2),\
                btype='lowpass', output='ba')
        #sos = sg.zpk2sos(zz, pp, kk)

        for ii in range(8):
            print('low pass filtering ...')
            h1 = sg.filtfilt(b, a, h1)
            h2 = sg.filtfilt(b, a, h2)
            h3 = sg.filtfilt(b, a, h3)
        '''

        fftfreqs = xp.fft.rfftfreq(Nperseg, 1.0/self.params['fs'])

        # Map of spectrum
        r1 = xp.zeros((fftfreqs.size, nsegs), dtype='complex')
        r2 = xp.zeros((fftfreqs.size, nsegs), dtype='complex')
        r3 = xp.zeros((fftfreqs.size, nsegs), dtype='complex')

        # Hann Window
        hwin = xp.hanning(Nperseg)
        win_fact = xp.mean(hwin**2)

        #        zpad = np.zeros(Nperseg)

        ## Initiate time segment arrays
        tsegstart = xp.zeros(nsegs)
        tsegmid = xp.zeros(nsegs)

        # We will use 50% overlapping segments
        for ii in range(0, nsegs):

            idxmin = int(ii*Nperseg)
            idxmax = idxmin + Nperseg
            idxmid = idxmin + int(Nperseg/2)
            if hwin.size != h1[idxmin:idxmax].size:
                import pdb; pdb.set_trace()

            if self.gpu:
                r1 = r1.at[:, ii].set(xp.fft.rfft(hwin*h1[idxmin:idxmax], axis=0))
                r2 = r2.at[:, ii].set(xp.fft.rfft(hwin*h2[idxmin:idxmax], axis=0))
                r3 = r3.at[:, ii].set(xp.fft.rfft(hwin*h3[idxmin:idxmax], axis=0))
                tsegstart = tsegstart.at[ii].set(timearray[idxmin])
                tsegmid = tsegmid.at[ii].set(timearray[idxmid])
            else:
                r1[:, ii] =   xp.fft.rfft(hwin*h1[idxmin:idxmax], axis=0)
                r2[:, ii] =   xp.fft.rfft(hwin*h2[idxmin:idxmax], axis=0)
                r3[:, ii] =   xp.fft.rfft(hwin*h3[idxmin:idxmax], axis=0)

                ## There's probably a more pythonic way of doing this, but it'll work for now.
                tsegstart[ii] = timearray[idxmin]
                tsegmid[ii] = timearray[idxmid]

        # "Cut" to desired frequencies
        idx = xp.logical_and(fftfreqs >=  self.params['fmin'] , fftfreqs <=  self.params['fmax'])

        # Output arrays
        fdata = fftfreqs[idx]

        # Get desired frequencies only
        # We want to normalize ffts so thier square give the psd
        # win_fact is to adjust for hann windowing, sqrt(2) for single sided
        r1 = xp.sqrt(2/win_fact)*r1[idx, :]/(self.params['fs']*xp.sqrt(self.params['seglen']))
        r2 = xp.sqrt(2/win_fact)*r2[idx, :]/(self.params['fs']*xp.sqrt(self.params['seglen']))
        r3 = xp.sqrt(2/win_fact)*r3[idx, :]/(self.params['fs']*xp.sqrt(self.params['seglen']))

        np.savez(self.params['out_dir'] + '/' +self.params['input_spectrum'], r1=r1, r2=r2, r3=r3, fdata=fdata)

        return np.array(r1), np.array(r2), np.array(r3), np.array(fdata), np.array(tsegstart), np.array(tsegmid)

    def _read_mldc_data(self):

        '''
        Read mldc or other external domain data from an ascii txt file. Since this was used primarily for
        the MLDC, it assumes that the data is given in X,Y and Z channels.
        Returns
        ---------

        h1, h2, h3 : float
            Time series data for the three TDI channels


        '''

        hoft = np.loadtxt(self.params['datafile'])

        fs_default = 1.0/(hoft[1, 0] - hoft[0, 0])

        ## Read in the duration seconds of data + one segment of buffer
        end_idx = int((self.params['dur'] + self.params['seglen'])*fs_default)

        ## the mldc data is X,Y,Z tdi
        times, h1, h2, h3 = hoft[0:end_idx, 0], hoft[0:end_idx, 1], hoft[0:end_idx, 2], hoft[0:end_idx, 3]

        delt = times[1] - times[0]

        ## Check if the requested sampel rate is consistant
        if self.params['fs'] != 1.0/delt:
            self.params['fs'] = 1.0/delt

        return h1, h2, h3, times


def cholesky_symm(m):
    r"""Cholesky decomposition of response matrix.

    Parameters
    ----------
    m : complex array (3, 3, ...)
        Response matrix. Assumed to have the form

        .. math::
            \begin{pmatrix}A & B & B \\ B^* & A & B \\ B^* & B^* & A\end{pmatrix}

    Returns
    -------
    complex array (3, 3, ...)
        Lower-triangular matrix L such that m = L L^H.
    """
    # manual cholesky decomposition assuming response mat of the form
    #     [ A  B  B ]
    # M = [ B* A  B ]. We take the elements of M from averaging the diagonal and
    #     [ B* B* A ]
    # off-diagonal components of the response_mat.
    # We do this custom decomposition to avoid NaNs because the SGWB response is rank-2.
    chex.assert_shape(m, (3, 3, ...))
    a = jnp.mean(jnp.array([m[0, 0], m[1, 1], m[2, 2]]), axis=0)
    b = jnp.mean(jnp.array([m[0, 1], m[0, 2], m[1, 2]]), axis=0)

    # a must be positive
    a = jnp.abs(a)

    mu = jnp.sqrt(a)
    lam = b.conj() / mu
    rho = lam
    sigma2 = jnp.maximum(0.0, a - jnp.abs(lam) ** 2)
    sigma = jnp.sqrt(sigma2)
    kappa = (b.conj() - jnp.abs(lam) ** 2) / sigma
    beta2 = jnp.maximum(0.0, a - jnp.abs(kappa) ** 2 - jnp.abs(lam) ** 2)
    beta = jnp.sqrt(beta2)

    _z = 0.0 * mu
    res = jnp.array([[mu, _z, _z], [lam, sigma, _z], [rho, kappa, beta]])

    chex.assert_equal_shape([m, res])
    return res


# Functions prefixed with _ldc were imported from the LDC hdf5 I/O module

def _ldc_str_decode(value):
    """Decode value if string

    >>> str_decode(b'hello')
    'hello'
    """
    if isinstance(value, (bytes)):
        return value.decode()
    return value

def _ldc_decode_utype(array):
    """Replace btype column in numpy array by unicode format.

    >>> decode_utype(np.rec.fromarrays([[b"a", b"b", b"c"], [1, 2, 3]], names=["name", "val"]))
    rec.array([('a', 1), ('b', 2), ('c', 3)],
              dtype=[('name', '<U1'), ('val', '<i8')])
    """
    sizeof_numpy_unicode_char = np.dtype("S1").itemsize

    if array.dtype.fields:
        new_dtype = [
            (
                (n, dt[0])
                if dt[0].kind != "S"
                else (
                    n,
                    np.dtype("<U%d" % (dt[0].itemsize // sizeof_numpy_unicode_char)),
                )
            )
            for n, dt in array.dtype.fields.items()
        ]
        array = array.astype(new_dtype)
    return array

def _ldc_load_array(filename, name="", full_output=True, sl=None):
    """Return array and its attributes from hdf5 file.

    if full_output is True, return array and meta data as dict.
    Otherwise, return array only.

    one can select a subset of data by giving a slice object as sl
    argument.

    """
    with h5py.File(filename, "r") as fid:
        names = [name] if name else list(fid.keys())
        attr = {}
        arrs = []
        for name in names:
            dset = fid.get(name)
            for k, v in dset.attrs.items():
                attr[k] = _ldc_str_decode(v)
            if sl is not None:
                dset = dset[sl]
            arrs.append(_ldc_decode_utype(np.array(dset)).squeeze())
        if len(names) > 1:
            try:  # make a rec array if all arrays share same size
                arr = np.rec.fromarrays(arrs, names=names)
            except ValueError:  # make a dict otherwise
                arr = dict(zip(names, arrs))
        else:
            arr = arrs[0]
        if full_output:
            return arr, attr
        return arr
