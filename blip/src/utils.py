import logging
from contextlib import contextmanager
import numpy as np

'''

utils.py contains a collection of ragtag miscellaneous utility functions. If you don't know where to put something, it's always welcome here <3

'''


## Some helper functions for Models, Injections, and submodels.
def catch_duplicates(names):
    '''
    Function to catch duplicate names so we don't overwrite keys while building a Model or Injection
    
    Arguments
    ---------------
    names (list of str) : model or injection submodel names
    
    Returns
    ---------------
    names (list of str) : model or injection submodel names, with duplicates numbered
    '''
    original_names = names.copy()
    duplicate_check = {name:names.count(name) for name in names}
    for key in duplicate_check.keys():
        if duplicate_check[key] > 1:
            cnt = 1
            for i, original_name in enumerate(original_names):
                if original_name == key:
                    names[i] = original_name + '-' + str(cnt)
    
    return names

def gen_suffixes(names):
    '''
    Function to generate appropriate parameter suffixes so repeated parameters are clearly linked to their respective submodel configurations.
    
    Arguments
    ---------------
    names (list of str) : model or injection submodel names
    
    Returns
    ---------------
    suffixes (list of str) : parameter suffixes for each respective model or injection submodel
    '''
    ## grab the spatial designation (or just 'noise' for the noise case)
    end_lst = [name.split('-')[0].split('_')[-1] for name in names]
    ## if we just have noise and a lone signal, we don't need to do this.
    if ('noise' in end_lst) and len(end_lst)==2:
        suffixes = ['','']
        return suffixes
    ## set up our building blocks and model counts for iterative numbering
    shorthand = {'noise':{'abbrv':r'','count':1},
                 'fixednoise':{'abbrv':r'','count':1},
                 'isgwb':{'abbrv':r'I','count':1},
                 'sph':{'abbrv':r'A','count':1},
                 'population':{'abbrv':r'P','count':1},
                 'hierarchical':{'abbrv':r'H','count':1},
                 'lmc':{'abbrv':r'LMC','count':1},
                 'fixedlmc':{'abbrv':r'LMC','count':1},
                 'sdg':{'abbrv':r'DG','count':1},
                 '1parametermw':{'abbrv':r'1MW','count':1},
                 '2parametermw':{'abbrv':r'2MW','count':1},
                 'galaxy':{'abbrv':r'G','count':1},
                 'fixedgalaxy':{'abbrv':r'G','count':1},
                 'popmap':{'abbrv':r'PM','count':1}} ## having galaxy and fixedgalaxy both map to 'G' is a skosh hacky but solves a later problem and shouldn't cause other issues
    
    suffixes = [r"[" for i in range(len(names))]
    
    ## find duplicates and count them
    dupc = {end:end_lst.count(end) for end in end_lst}
    
    ## some handling to prevent things from breaking if the above dict hasn't been updated for a given model
    for end in end_lst:
        if end not in shorthand.keys():
            shorthand |= {end:{'abbrv':r'Mod','count':1}}
    
    ## generate the suffixes by assigning the abbreviated notation and numbering as necessary
    for i, (end,suff) in enumerate(zip(end_lst,suffixes)):
        if end == 'noise':
            if dupc[end] > 1:
                raise ValueError("Multiple noise injections/models is not supported.")
            else:
                suffixes[i] = r''
        elif dupc[end] == 1:
            suffixes[i] = suff + shorthand[end]['abbrv'] + r']'
#            suffixes[i] = r'%s' %suff_i
        else:
            suffixes[i] = suff + shorthand[end]['abbrv'] + r'_' + str(shorthand[end]['count']) + r']'
#            suffixes[i] = r'%s' %suff_i
            shorthand[end]['count'] += 1

    return suffixes

def catch_color_duplicates(Object,color_pool=None,sacred_labels=[]):
    '''
    Function to catch duplicate plotting colors and reassign from a default or user-specified pool of matplotlib colors.
    
    Arguments
    ------------
    Object : Model or Injection with attached submodels.
    color_pool : List of matplotlib color namestrings; see https://matplotlib.org/stable/gallery/color/named_colors.html
    sacred_labels : List of submodel names whose colors should be treated as inviolate.
    
    '''
    if color_pool is None:
        ## this is meant to be a decently large pool, all of which are reasonably distinct from one another
        ## we include all the default colors assigned to submodels above, as its rare that all of them will be in use
        color_pool = ['fuchsia','sienna','turquoise','deeppink','goldenrod',
                      'darkmagenta','midnightblue','gold','crimson','mediumorchid','darkorange','maroon','forestgreen','teal']
        
    
    ## handle Model vs. Injection differences
    if hasattr(Object,"component_names"):
        labels = Object.component_names
        items = Object.components
    elif hasattr(Object,"submodel_names"):
        labels = Object.submodel_names
        items = Object.submodels
    else:
        raise TypeError("Provided Object is not a properly-constructed Model or Injection.")
    
    ## remove in-use colors from the pool
    for idx, color in enumerate(color_pool):
        if color in [items[label].color for label in labels]:
            del color_pool[idx]

    ## step through the submodels and re-assign any duplicated colors
    color_list = [items[label].color for label in sacred_labels]
    for label in labels:
        if (items[label].color in color_list) and (label not in sacred_labels):
            items[label].color = color_pool.pop(0)
        color_list.append(items[label].color)
    
    return

def ensure_color_matching(Model,Injection):
    '''
    Function to ensure linked Model and Injection models share a color in the final posterior fitmaker plot.
    
    (i.e., pairwise matching between submodels and injection components that share a name.)
    
    Arguments
    -----------
    Model       : Model object
    Injection   : Injection object
    
    '''
    
    ## find matches
    matching_keys = [key for key in Injection.component_names if key in Model.submodel_names]
    
    ## ensure color matching
    for key in matching_keys:
        if Injection.components[key].color != Model.submodels[key].color:
            Injection.components[key].color = Model.submodels[key].color
    
    ## reassign unmatched color duplicates as needed
    catch_color_duplicates(Injection,sacred_labels=matching_keys)
    
    return

def get_robson19_shape_pars_from_tobs(T_obs,shape_pars=[r"$\\alpha_{\\rm shape}$", r"$\\beta$", r"$\\kappa$", r"$\\gamma$", r"$f_{\\rm knee}$"]):
    '''
    Function to return the time-dependent foreground shape parameters of the Robson+19 analytic foreground spectrum,
    given an observing time in years. The function will round the given duration to the closest observing time 
    in Table 1 of https://arxiv.org/abs/1803.01944.
    
    Arguments
    -------------
    T_obs (float)    :  Observing duration **in years**. Will be rounded to one of (6mo, 1yr, 2yr, 4yr).
    shape_pars (list of str) : Names of the desired shape parameters (default all) from 
                                r"$\\alpha_{\\rm shape}$", r"$\\beta$", r"$\\kappa$", r"$\\gamma$", r"$f_{\\rm knee}$'
    
    Returns
    -------------
    shapevals (dict) : Dictionary of 'parameter_name':val for the shape parameters of the Robson 19 analytic foreground model.
    
    '''
    ## values from Table 1 of Robson+19
    r19_table1 = {'6mo':{r"$\\alpha_{\\rm shape}$":0.133, r"$\\beta$":243, r"$\\kappa$":482, r"$\\gamma$":917, r"$f_{\\rm knee}$":0.00258},
                  '1yr':{r"$\\alpha_{\\rm shape}$":0.171, r"$\\beta$":292, r"$\\kappa$":1020, r"$\\gamma$":1680, r"$f_{\\rm knee}$":0.00215},
                  '2yr':{r"$\\alpha_{\\rm shape}$":0.165, r"$\\beta$":299, r"$\\kappa$":611, r"$\\gamma$":1340, r"$f_{\\rm knee}$":0.00173},
                  '4yr':{r"$\\alpha_{\\rm shape}$":0.138, r"$\\beta$":-221, r"$\\kappa$":521, r"$\\gamma$":1680, r"$f_{\\rm knee}$":0.00113}
                  }
    ## catch if T_obs is given in s
    if type(T_obs) is not str:
        if (T_obs > 11) and (T_obs >= 1e5):
            print("Warning, likely trying to reference Robson+19 Table 1 with T_obs given in seconds. Converting to years...")
        elif T_obs > 4.1:
            print("Warning: Robson+19 Table 1 shape parameter values only available for up to a 4yr observing duration. Defaulting to the 4yr values...")
    
        ## convert T_obs to closest value and get corresponding key
        table_times = np.array([0.5,1,2,4])
        table_time_strings = ['6mo','1yr','2yr','4yr']
        t_idx = np.argmin(np.abs(table_times - T_obs))
        r19_key = table_time_strings[t_idx]
    else:
        r19_key = T_obs
    
    ## build shapeval dict
    shapevals = {key:r19_table1[r19_key][key] for key in shape_pars}
    
    return shapevals
    
    
    
    

## function for telling healpy to hush up
@contextmanager
def log_manager(level):
    '''
    Context manager to clean up bits of the code where we want e.g., healpy to be quieter.
    Adapted from code by Martin Heinz (https://martinheinz.dev/blog/34)
    
    Arguments
    -----------
    level: logging level (DEBUG, INFO, WARNING, ERROR)

    '''
    logger = logging.getLogger()
    current_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(current_level)