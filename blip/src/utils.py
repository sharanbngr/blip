import logging
from contextlib import contextmanager


'''

utils.py contains a collection of ragtag miscellaneous utility functions. If you don't know where to put something, it's always welcome here <3

'''


def parse_submodel_name(name):
    '''
    Parse a BLIP submodel/component name into spectral and spatial pieces.

    The base syntax is ``spectral_spatial``. We also support extended spatial
    labels like ``sph_l2`` for an ``ell``-collapsed anisotropic SGWB component.
    Duplicate markers appended as ``-N`` are ignored for the parsed metadata.
    '''

    base_name = name.split('-')[0]

    if base_name == 'noise':
        return {
            'base_name': base_name,
            'spectral_name': 'noise',
            'spatial_name': 'noise',
            'spatial_kind': 'noise',
            'ell': None,
        }

    if '_' in base_name:
        spectral_name, spatial_name = base_name.split('_', 1)
    else:
        spectral_name = spatial_name = base_name

    spatial_kind = spatial_name
    ell = None

    if spatial_name.startswith('sph_l'):
        ell_text = spatial_name[len('sph_l'):]
        if (ell_text == '') or (not ell_text.isdigit()):
            raise ValueError(
                "Invalid ell-collapsed anisotropic spatial model '{}'. "
                "Use the form 'sph_lN', e.g. 'sph_l2'.".format(spatial_name)
            )
        spatial_kind = 'sph_l'
        ell = int(ell_text)

    return {
        'base_name': base_name,
        'spectral_name': spectral_name,
        'spatial_name': spatial_name,
        'spatial_kind': spatial_kind,
        'ell': ell,
    }


def spatial_suffix_label(name):
    '''
    Generate a short suffix label for a model/component based on its spatial
    specification.
    '''

    parsed = parse_submodel_name(name)
    spatial_kind = parsed['spatial_kind']

    shorthand = {
        'noise': '',
        'isgwb': 'I',
        'sph': 'A',
        'population': 'P',
        'hierarchical': 'H',
        'galaxy': 'G',
        'dwarfgalaxy': 'DG',
        'lmc': 'LMC',
        'pointsource': '1P',
        'twopoints': '2P',
    }

    if spatial_kind == 'sph_l':
        return 'L{}'.format(parsed['ell'])

    return shorthand.get(spatial_kind, parsed['spatial_name'])


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
    parsed_names = [parse_submodel_name(name) for name in names]
    end_lst = []
    label_lst = []
    for parsed, name in zip(parsed_names, names):
        if parsed['spatial_kind'] == 'sph_l':
            end_lst.append('sph_l{}'.format(parsed['ell']))
        else:
            end_lst.append(parsed['spatial_kind'])
        label_lst.append(spatial_suffix_label(name))

    ## if we just have noise and a lone signal, we don't need to do this.
    if ('noise' in end_lst) and len(end_lst)==2:
        suffixes = ['','']
        return suffixes

    count_map = {end:1 for end in end_lst}
    
    suffixes = ['  $\mathrm{[' for i in range(len(names))]
    
    ## find duplicates and count them
    dupc = {end:end_lst.count(end) for end in end_lst}
    
    ## generate the suffixes by assigning the abbreviated notation and numbering as necessary
    for i, (end, label, suff) in enumerate(zip(end_lst,label_lst,suffixes)):
        if end == 'noise':
            if dupc[end] > 1:
                raise ValueError("Multiple noise injections/models is not supported.")
            else:
                suffixes[i] = ''
        elif dupc[end] == 1:
            suffixes[i] = suff + label + ']}$'
        else:
            suffixes[i] = suff + label + '_' + str(count_map[end]) + ']}$'
            count_map[end] += 1

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
