import sys, os
sys.path.append(os.getcwd()) ## this lets python find src
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from chainconsumer import ChainConsumer
import healpy as hp
from healpy import Alm
from astropy import units as u
import pickle, argparse
import logging
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import h5py
from scipy.stats import norm
import pickle, argparse


#############################
#############################
#change according to each storage
#############################
directory = 'run_directory_97'
fmin=4e-4
fmax=1e-3 
seglen = 1e5

#change the prior
lnrmax = 40
lnrmin= -70
taumax = 10
taumin = 0.5
sigmamax = 20
sigmamin = 0.01
############################
############################



# interval for bounds
interval = 0.9

alpha = 0.667
fref= 25
samp_size=500
path = '/home/hqn7750/lisa/Storage/'+directory+'/nessai_output/result.hdf5'
#############################
#############################

l=2

#############################
#############################
#read the posterior samples
f = h5py.File(path, 'r')
dset = f['posterior_samples']
parameters = dset.dtype.names
data = np.array([np.array(dset[name]) for name in parameters])

tau, sigma, lnr = dset['$\\tau$'], dset['$\\sigma$'], dset['$\\ln r$']

if l==2:
	n_ang = 8
ns_name = ['$n_{' + str(i) + '}$' for i in range(len(parameters)-(8+n_ang))]
ns = np.array([np.array(dset[name]) for name in ns_name]).transpose()
fs = np.linspace(fmin,fmax,len(ns[0]))

#transform/scale from [0,1] to their actural distributions
lnr =  np.log((np.exp(lnrmin+lnrmax)-np.exp(lnrmin))*lnr +np.exp(lnrmin))## for lnr with log prior
tau = (taumax*np.log(fmax/fmin)*tau+taumin*np.log(1+(1/seglen)/fmax)) ## for tau
sigma  = (sigmamax*sigma + sigmamin) ## for sigma
ns = norm.ppf(ns) # for ns
qt = [0.5,(1-interval)/2,1-((1-interval)/2)]  #median, lower bound, and upper boundn

ns_t = ns.transpose()
f_ind = int(len(ns_t))
med_ind = int(len(ns_t)/2)
ni = ns_t[0]
nmed = ns_t[med_ind]
nf = ns_t[-1]
#############################
#############################



#############################
#############################
#corner plot
post = [lnr,tau,sigma,ni,nmed,nf]
all_parameters = ['$\\ln r$','$\\tau$','$\\sigma$','$n_0$','$n_{' + str(med_ind) + '}$','$n_{' + str(f_ind) + '}$']
npar = len(all_parameters)
cc = ChainConsumer()
cc.add_chain(post, parameters=all_parameters)
cc.configure(smooth=False, kde=False, max_ticks=2, sigmas=np.array([1, 2]), label_font_size=18, tick_font_size=18, \
            summary=False, statistics="max_central", spacing=2, summary_area=0.95, cloud=False, bins=1.2)
fig = cc.plotter.plot(figsize=(16, 16))
sum_data = cc.analysis.get_summary()
axes = np.array(fig.axes).reshape((npar, npar))
# Adjust axis labels
for ii in range(npar):
    ax = axes[ii, ii]

    # get the right summary for the parameter ii
    sum_ax = sum_data[all_parameters[ii]]
    err =  [sum_ax[2] - sum_ax[1], sum_ax[1]- sum_ax[0]]

    if np.abs(sum_ax[1]) <= 1e-3:
        mean_def = '{0:.3e}'.format(sum_ax[1])
        eidx = mean_def.find('e')
        base = float(mean_def[0:eidx])
        exponent = int(mean_def[eidx+1:])
        mean_form = str(base)
        exp_form = ' \\times ' + '10^{' + str(exponent) + '}'
    else:
        mean_form = '{0:.3f}'.format(sum_ax[1])
        exp_form = ''

    if np.abs(err[0]) <= 1e-2:
        err[0] = '{0:.4f}'.format(err[0])
    else:
        err[0] = '{0:.2f}'.format(err[0])

    if np.abs(err[1]) <= 1e-2:
        err[1] = '{0:.4f}'.format(err[1])
    else:
        err[1] = '{0:.2f}'.format(err[1])

    label =  all_parameters[ii][:-1] + ' = ' + mean_form + '^{+' + err[0] + '}_{-' + err[1] + '}'+exp_form+'$'

    ax.set_title(label, {'fontsize':18}, loc='left')
plt.savefig('/home/hqn7750/lisa/Storage/'+directory+'/AR_corner.png', dpi=200)
#############################
#############################

#############################
#############################
#AR process plot
h = 2.2e-18 
#hubble constant
sgw_l = []
for i in range(len(lnr)):
    mean = np.full(len(fs), lnr[i])
    cov = sigma[i]**2 * (np.exp(-np.abs((np.log(fs)[:, np.newaxis] - np.log(fs)))/tau[i]))
    A = np.linalg.cholesky(cov)
    logP = np.dot(A, ns[i]) + mean
    P = np.exp(logP)
    sgw = (3 * (h ** 2) * (P)) / (4 * (fs ** 3) * np.pi ** 2)
    sgw_l.append(sgw)
sgw_l = np.array(sgw_l)
sgw_t = sgw_l.transpose()
sgw_stats= [[np.quantile(sgw_t[i], q) for i in range(len(sgw_t))] for q in qt]
np.save('/home/hqn7750/lisa/Storage/'+directory+'/sgw_l.npy', sgw_l)

ind = np.random.choice(len(lnr),samp_size)
# plt.figure(figsize=(12,9))
plt.figure()

for i in range(samp_size):
	plt.plot(fs, sgw_l[ind][i], color = 'lightblue', alpha = 0.4, linewidth = 0.15)
# Omega_power = 10**(log_omega0)*(fs/fref)**alpha
# Omega_sgw =(3 * (h ** 2) * (Omega_power)) / (4 * (fs ** 3) * np.pi ** 2)
# plt.plot(fs, Omega_sgw,color ='orange',alpha =0.8,linewidth = 0.7)

#############################
#############################
#injected population plot
path = '/home/hqn7750/lisa/Storage/'+directory
with open(path + '/injection.pickle', 'rb') as injectionfile:
    Injection = pickle.load(injectionfile)
with open(path + '/model.pickle', 'rb') as modelfile:
    Model = pickle.load(modelfile)
component_name = 'population'
frange = Model.fs
ffilt = np.logical_and(frange >= fmin, frange <= fmax)
fs = frange[ffilt]
fs = fs.reshape(-1,1)
Injection.plot_injected_spectra(component_name,fs_new=fs,legend=False, color = 'orange',linewidth = 0.7,alpha = 1, label = 'Injection')
#############################
#############################

plt.plot(fs, sgw_stats[0], color = 'blue', alpha = 0.8, linestyle = 'dashed',linewidth = 0.6,label = 'Recovery')
plt.plot(fs, sgw_stats[1], color = 'blue', alpha = 0.8, linestyle = 'dotted',linewidth = 0.6)
plt.plot(fs, sgw_stats[2], color = 'blue', alpha = 0.8, linestyle = 'dotted',linewidth = 0.6)
	

plt.legend(loc='upper right',fontsize = 10)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [1/Hz]')
#############################
#############################

plt.savefig('/home/hqn7750/lisa/Storage/'+directory+'/AR_plot_bounds.png')




