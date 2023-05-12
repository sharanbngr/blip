import logging
from contextlib import contextmanager
import numpy as np

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