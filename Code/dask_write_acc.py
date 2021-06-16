#!/usr/bin/env python3

""" Writes data set for deep learning model to predict dives using acceleration
data. """

__author__ = 'Luke Swaby (lds20@ic.ac.uk)'
__version__ = '0.0.1'

## Imports ##

import os #
import pickle #
import uuid #
import sys
import glob
import re
import time
import argparse
import numpy as np #
import dask.dataframe as dd
import dask.array as da
import dask.core as core #
from dask.highlevelgraph import HighLevelGraph #
from dask.base import compute_as_if_collection #

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
    parser.add_argument('-o', dest='outdir', type=str,
                        default='../Data/Acc_npy/',
                        help='Path to directory to stack output npy files.')
    parser.add_argument('-w', dest='window', type=int,
                        default=250,
                        help='Rolling window width.')
    parser.add_argument('-t', dest='threshold', type=float,
                        default=0.03,
                        help='Depth threshold for identifying dives.')

    args = parser.parse_args()

    print(f'PARAMS USED:\n'
          f'indir:\t{args.indir}\n'
          f'outdir:\t{args.outdir}\n'
          f'window:\t{args.window}\n'
          f'threshold:\t{args.threshold}\n')

    return args.indir, args.outdir, args.window, args.threshold


def check_for_dive(arr, thrshold=0.03):
    """
    Determines whether dive occured in given window of depth values.

    :param arr: array of depth values
    :param thrshold: depth value past which a dive is defined
    :return: boolean indicating whether dive has occured
    """
    return int((arr[~da.isnan(arr)] > thrshold).any())


def to_npz_stack(dirname, x, axis=0):
    """
    Write dask array to a stack of .npz files
    This partitions the dask.array along one axis and stores each block along
    that axis as a single compressed .npz file in the specified directory
    Modified from the dask function: to_npy_stack()
    :param dirname:
    :param x:
    :param axis:
    :return:
    """

    chunks = tuple((c if i == axis else (sum(c),)) for i, c in enumerate(x.chunks))
    xx = x.rechunk(chunks)

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    #meta = {"chunks": chunks, "dtype": x.dtype, "axis": axis}

    with open(os.path.join(dirname, "info"), "wb") as f:
        pickle.dump(meta, f)

    name = "to-npy-stack-" + str(uuid.uuid1())  # create universally unique ID
    dsk = {
        (name, i): (np.savez_compressed, os.path.join(dirname, "%d.npz" % i), key)
        for i, key in enumerate(core.flatten(xx.__dask_keys__()))
    }

    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[xx])
    compute_as_if_collection(da.Array, graph, list(dsk))

    #return meta


def main(indir, outdir, wdw=250, threshold=0.03):
    """
    Creates numpy dataframe by taking a rolling window of 250 rows of the input
    arr and horizontally arranging x, y, and z values followed by a binary
    value indicating whether or not a dive has occured in that window.

    :type birdname: str
    :param birdname: TagID of bird currently being processed
    :param arr: numpy array containing 4 columns: X, Y, Z, and Depth
    :return:
    """
    assert indir.endswith('/'), "indir arg must end with a '/'"
    assert outdir.endswith('/'), "outdir arg must end with a '/'"

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        print("Created folder: ", outdir)

    # Grab list of reevant file paths
    files = glob.glob(f'{indir}*1.csv')

    count = 0

    for no, file in enumerate(files, 1):

        tik = time.time()

        print(f"PROCESSING FILE {no}: '{file}'...")
        arr = dd.read_csv(file, usecols=['X', 'Y', 'Z', 'Depth_mod']).to_dask_array(
            lengths=True)

        # Grab bird ID from filepath
        bird = re.search(r"/(\w+).csv", file).group(1)

        x = da.lib.stride_tricks.sliding_window_view(arr[:, 0], wdw)
        y = da.lib.stride_tricks.sliding_window_view(arr[:, 1], wdw)
        z = da.lib.stride_tricks.sliding_window_view(arr[:, 2], wdw)
        depth = da.lib.stride_tricks.sliding_window_view(arr[:, 3], wdw)

        d = da.apply_along_axis(check_for_dive, 1, depth, threshold)
        train_data = da.hstack((x, y, z, d.reshape((d.shape[0], 1))))
        #shuf_ix = da.random.choice(len(train_data), len(train_data), replace=False)
        #train_data = train_data[shuf_ix]
        #train_data = train_data[shuf_ix].rechunk(train_data.chunks)

        da.to_npy_stack(f'{outdir}{bird}', train_data, axis=0)
        #to_npz_stack(f'{outdir}{bird}', train_data, axis=0)

        #########################################################
        # save dive stats
        #dive_no = da.count_nonzero(train_data[:,-1], 1).compute()
        #stats = {'Dives': dive_no,
        #         'Non-dives': train_data.shape[0] - dive_no}
        #with open(f'{outdir}{bird}/dive_stats.pickle', 'wb') as handle:
        #    pickle.dump(stats, handle)
        #########################################################

        # Informative output stats
        meta = {'Shape': train_data.shape, 'Files': len(train_data.chunks[0])}

        # Informative output
        for k, v in meta.items():
            print('\t' + k + ':', v)

        tok = time.time() - tik

        print("\tTime elapsed: %.2fs" % tok)
        count += tok

    sys.exit('\nDone\n\nTotal time elapsed: %.2fs' % count)


if __name__ == '__main__':
    pth, outdir, window, threshold = parse_arguments()
    main(pth, outdir, window, threshold)