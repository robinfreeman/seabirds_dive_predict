#!/usr/bin/env python3

""" Writes data set for deep learning model to predict dives using acceleration
data. """

__author__ = 'Luke Swaby (lds20@ic.ac.uk)'
__version__ = '0.0.1'

## Imports ##

import os
import sys
import glob
import re
import time
import random
import argparse
import numpy as np
import pandas
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
                        default='../Data/BIOT_DGBP/BIOT_DGBP/',
                        help='Path to directory containing the raw data')
    parser.add_argument('-o', dest='out', type=str,
                        default='../Data/Acc_npy/',
                        help='File/directory path for out file(s).')
    parser.add_argument('-w', dest='window', type=int,
                        default=250,
                        help='Rolling window width.')
    parser.add_argument('-t', dest='threshold', type=float,
                        default=0.03,
                        help='Depth threshold for identifying dives.')
    parser.add_argument('-r', dest='reduce', type=bool,
                        default=True,
                        help='Boolean (True/False) indicating whether or not '
                             'you want larger dset written to a npy stack '
                             '(False) or a reduced dset written to csv (True).')

    args = parser.parse_args()

    print(f'PARAMS USED:\n'
          f'indir:\t{args.indir}\n'
          f'out:\t{args.out}\n'
          f'window:\t{args.window}\n'
          f'threshold:\t{args.threshold}\n'
          f'reduce:\t{args.reduce}\n')

    return args.indir, args.out, args.window, args.threshold, args.reduce


def check_for_dive(arr, thrshold=0.03):
    """
    Determines whether dive occured in given window of depth values.

    :param arr: array of depth values
    :param thrshold: depth value past which a dive is defined
    :return: boolean indicating whether dive has occured
    """
    return int((arr[~da.isnan(arr)] > thrshold).any())


def rolling_acceleration_window(arr, wdw, threshold):
    """
    Create huge dset
    :return:
    """
    x = da.lib.stride_tricks.sliding_window_view(arr[:, 0], wdw)
    y = da.lib.stride_tricks.sliding_window_view(arr[:, 1], wdw)
    z = da.lib.stride_tricks.sliding_window_view(arr[:, 2], wdw)
    depth = da.lib.stride_tricks.sliding_window_view(arr[:, 3], wdw)

    d = da.apply_along_axis(check_for_dive, 1, depth, threshold)
    train_data = da.hstack((x, y, z, d.reshape((d.shape[0], 1))))

    return train_data

def reduce_dset(data, idd):
    """
    Creates reduce dset for each bird
    :param data: huge dask array of acc->dive data
    :param idd: bird id
    :return: reduce dset
    """
    # Sample dives
    dive_ix = da.where(data[:, -1] == 1)[0].compute()
    pos = data[dive_ix]
    n_dive = len(dive_ix)

    # Sample non-dives
    no_dive_ix = np.setdiff1d(np.arange(data.shape[0]), dive_ix)
    n_no_dive = random.randint(n_dive, round(n_dive*1.1))
    neg = data[da.random.choice(no_dive_ix, n_no_dive, replace=False)]

    # Stack, shuffle and compute
    data_add = dd.from_dask_array(da.vstack((pos, neg))).compute()
    data_add = data_add.sample(frac=1)
    data_add.columns = [*data_add.columns[:-1], 'Dive']  # rename last col
    data_add['Dive'] = data_add['Dive'].astype(int)
    data_add['BirdID'] = idd
    data_add = data_add[['BirdID'] + list(data_add.columns[:-1])]  # reorder cols

    return data_add


def main(indir, outpth, wdw=250, threshold=0.03, reduce=True):
    """
    Creates numpy dataframe by taking a rolling window of 250 rows of the input
    arr and horizontally arranging x, y, and z values followed by a binary
    value indicating whether or not a dive has occured in that window.
    """
    assert indir.endswith('/'), "indir arg must end with a '/'"
    if reduce:
        assert not outpth.endswith('/'), "Out path should be to file, not dir"

    # Grab list of reevant file paths
    files = glob.glob(f'{indir}*1.csv')

    t_count = 0
    rows = 0
    dives = 0
    non_dives = 0

    for no, file in enumerate(files, 1):

        tik = time.time()

        print(f"PROCESSING FILE {no}: '{file}'...")

        # Create dset from rolling window
        print('\r\tCreating rolling window dset...')
        arr = dd.read_csv(file, usecols=['X', 'Y', 'Z', 'Depth_mod']).to_dask_array(
            lengths=True)
        train_data = rolling_acceleration_window(arr, wdw, threshold)
        bird = re.search(r"/(\w+).csv", file).group(1)  # bird ID from path

        if reduce:
            # Reduce dataset
            print('\r\tReducing dset...')
            out_dset = reduce_dset(train_data, bird)
            print('\r\tWriting to csv...')
            out_dset.to_csv(outpth, mode='a', header=True if no == 1 else False,
                            index=False)
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
    shp = (rows, wdw*3+2) if reduce else (rows, wdw*3+1)  # no id col if npy
    sys.exit(f"\nDone\n\nTotal time elapsed: %.2fs\n\n"
             f"Outfile stats:\n"
             f"\tShape: {shp}\n"
             f"\tDives: {dives}\n"
             f"\tNon-dives: {non_dives}" % t_count)


if __name__ == '__main__':
    main(*parse_arguments())