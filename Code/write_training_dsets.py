#!/usr/bin/env python3

""" Writes data set for deep learning model to predict dives using acceleration
data. """

__author__ = 'Luke Swaby (lds20@ic.ac.uk)'
__version__ = '0.0.1'

## Imports ##
import pandas as pd

import core
import os
import sys
import glob
import re
import time
import argparse
import dask.dataframe as dd
import dask.array as da

## Functions ##

def parse_arguments():
    """Function to parse args from command line
    """
    parser = argparse.ArgumentParser(
    description="Script for generating large data set for deep learning "
                "from rolling window of acceleration values..")

    parser.add_argument('-i', dest='indir', type=str,
                    default='../Data/GLS Data 2019 Jan DG RFB Short-term/matched/',
                    help='Path to directory containing the raw data')
    parser.add_argument('-d', dest='dtype', type=str, choices=['ACC', 'IMM'], required=True,
                        help='Data sets to analyse (ACC/IMM).')
    parser.add_argument('-o', dest='out', type=str,
                    default='../Data/Reduced/ACC_reduced_all_dives.csv',
                    help='File/directory path for out file(s).')
    parser.add_argument('-w', dest='window', type=int, default=2,
                    help='Rolling window width.')
    parser.add_argument('-t', dest='threshold', type=float, default=0.03,
                    help='Depth threshold for identifying dives.')
    parser.add_argument('-r', dest='res', type=int, default=25,
                    help='Data resolution (Hz for ACC data and seconds between each record for IMM data).')
    parser.add_argument('--reduce', dest='reduce', action='store_true',
                        help='Reduce dset to csv? (Will create a balanced dataset including all dive rows and similar '
                             'number of non-dive rows. Omission will write full data set to npy stack using dask by '
                             'default.)')
    parser.set_defaults(reduce=False)

    # indir, dtype, outpth, wdw, threshold, res, reduce = ('../Data/BIOT_DGBP/', 'ACC', '../Data/poo.csv', 8, 0.03, 25, True)
    # indir, dtype, outpth, wdw, threshold, res, reduce = ('../Data/GLS Data 2019 Jan DG RFB Short-term/matched/', 'IMM', '../Data/poo.csv', 120, 0.03, 6, True)

    args = parser.parse_args()

    print(f'\nPARAMS USED:\n'
          f'indir:\t{args.indir}\n'
          f'dtype:\t{args.dtype}\n'
          f'out:\t{args.out}\n'
          f'window:\t{args.window}s\n'
          f'thold:\t{args.threshold}\n'
          f'res:\t{args.res}Hz\n'
          f'reduce:\t{args.reduce}\n')

    return args.indir, args.dtype, args.out, args.window, args.threshold, args.res, args.reduce


def main(indir, dtype, outpth, wdw=10, threshold=0.03, res=25, reduce=True):
    """
    Creates numpy dataframe by taking a rolling window of 250 rows of the input
    arr and horizontally arranging x, y, and z values followed by a binary
    value indicating whether or not a dive has occured in that window.
    """
    assert indir.endswith('/'), "indir arg must end with a '/'"
    if reduce:
        assert not outpth.endswith('/'), "Out path should be to file, not dir"
        if os.path.exists(outpth):
            # check if user wishes to remove file at outpth to continue
            ans = input(f"The file '{outpth}' already exists and must be removed to continue. Rm file? (y/n) ").lower()
            while ans not in ['y', 'n']:
                ans = input(f"Type 'y' to remove the file '{outpth}' and continue or 'n' to exit: ").lower()
            if ans == 'n':
                sys.exit('\nGoodbye :)')
            else:
                print(f'{outpth} removed.\n')
                os.remove(outpth)  # remove file if already exists, otherwise out file will be appended

    # Grab list of relevant file paths
    files = glob.glob(f'{indir}{dtype}*.csv')

    # f = glob.glob(f'{indir}GLS_x_Depth*.csv')[0]
    # f = glob.glob(f'{indir}{dtype}*.csv')[2]

    t_count = 0
    rows = 0
    dives = 0
    non_dives = 0

    for no, f in enumerate(files, 1):

        tik = time.time()

        print(f"PROCESSING FILE {no}: '{f}'...")

        # Create dset from rolling window
        print('\r\tCreating rolling window dset...')

        if dtype == 'ACC':
            arr = dd.read_csv(f).to_dask_array(lengths=True)
            train_data = core.rolling_acceleration_window(arr, wdw, threshold, res)
        else:
            df = dd.read_csv(f, usecols=['ix', 'wet/dry', 'Depth_mod'], dtype={'Depth_mod': 'float64'})
            conv_nums = {'wet/dry': {'wet': 1, 'dry': 0}}
            arr = df[['ix', 'wet/dry', 'Depth_mod']].replace(conv_nums).to_dask_array(lengths=True)
            train_data = core.rolling_immersion_window(arr, wdw, threshold, res)

        bird = re.search(fr"{dtype}_(\w+).csv", f).group(1)  # bird ID from path

        if reduce:
            # Reduce dataset
            print('\r\tReducing dset...')
            out_dset = core.reduce_dset(train_data)
            out_dset['BirdID'] = bird
            out_dset = out_dset[['BirdID', *out_dset.columns[:-1]]]  # reorder cols
            print('\r\tWriting to csv...')
            out_dset.to_csv(outpth, mode='a', header=True if no == 1 else False, index=False)

            """
            ## Dask version#############
            for i in range(out_dset.npartitions):
                print(f'\nPartition {i}...\n')
                out_i = out_dset.get_partition(i).compute()
                out_i = out_i.sample(frac=1)  # shuffle
                out_i.to_csv(outpth, mode='a', header=True if (no == 1 and i == 1) else False, index=False)

            #########################
            """

        else:
            out_dset = train_data
            outdir = outpth if outpth.endswith('/') else outpth + '/'
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
                print("\tCreated folder: ", outdir)
            print('\tSaving to npy stack...')
            da.to_npy_stack(f'{outdir}{bird}', out_dset, axis=0)

        tok = time.time() - tik

        # Informative output
        d = (out_dset['Dive'] == 1).sum() if reduce else (out_dset[:, -1] == 1).sum()
        nd = out_dset.shape[0] - d

        print(f"\r\tDone!\n"
              f"\tShape: {out_dset.shape}\n"
              f"\tDives: {d}\n"
              f"\tNon-dives: {nd}\n"
              f"\tTime elapsed: %.2fs" % tok)

        t_count += tok
        rows += out_dset.shape[0]
        dives += d
        non_dives += nd

    # Informative output
    # TODO: correct these outputs for IMM data
    shp = (rows, wdw*res*3+2) if reduce else (rows, wdw*res*3+1)  # no id col if npy
    sys.exit(f"\nDone\n\nTotal time elapsed: %.2fs\n\n"
             f"Outfile stats:\n"
             f"\tShape: {shp}\n"
             f"\tDives: {dives}\n"
             f"\tNon-dives: {non_dives}" % t_count)


if __name__ == '__main__':
    main(*parse_arguments())