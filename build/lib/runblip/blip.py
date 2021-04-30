#!/usr/bin/env python3

import argparse
import os, shutil
import configparser

def main():
    parser = argparse.ArgumentParser(prog='blip', description='Bayesian pipeline for detecting stochastic backgrounds with LISA')

    parser.add_argument('-i', '--init', action='store_const', const=True, help='Initialize params.ini file in your current working directory', dest='init')
    
    parser.add_argument('-x', '--execute', action='store_const', const=True, help='run_blip with the params file in the current working directory', dest='execute')
    
    args = parser.parse_args()
    
    if args.init:
        cwd = os.getcwd()
        cwd_path = os.path.join(cwd, 'params.ini')
        
        orig_path = os.path.dirname(os.path.realpath(__file__))
        orig_path = os.path.join(orig_path, 'params.ini')
        
        print('Initializing default params.ini...')
        
        shutil.copyfile(orig_path, cwd_path)
        
        print('Initialized default params at', cwd+'/')
    
    if args.execute:
        print('Running run_blip...')
        from runblip.main import run_blip
        
        cwd = os.getcwd()
        cwd_path = os.path.join(cwd, 'params.ini')
        
        config = configparser.ConfigParser()
        config.read(cwd_path)
        
        out_dir = str(config.get("run_params", "out_dir"))
        
        new_dir = os.path.join(cwd, out_dir)
        
        if out_dir != new_dir:
            config.set('run_params', 'out_dir', new_dir)
            
            with open(cwd_path, 'w') as configfile:
                config.write(configfile)

        run_blip.blip(cwd_path)

# if __name__ == "__main__":
#     main()