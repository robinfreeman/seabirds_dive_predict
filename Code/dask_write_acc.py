#!/usr/bin/env python3

""" Writes data set for deep learning model to predict dives using acceleration
data. """

__author__ = 'Luke Swaby (lds20@ic.ac.uk)'
__version__ = '0.0.1'

## Imports ##

import sys
import glob
import re
import dask.dataframe as dd
import dask.array as da

import time

import os
import pickle
import uuid
import numpy as np
import dask.core as core
from dask.highlevelgraph import HighLevelGraph
from dask.base import compute_as_if_collection

## Functions ##

def check_for_dive(arr, thrshold=0.5):
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

    meta = {"chunks": chunks, "dtype": x.dtype, "axis": axis}

    with open(os.path.join(dirname, "info"), "wb") as f:
        pickle.dump(meta, f)

    # I want it to save files named birdID.npz containing all separate arrays
    # to specified dir
    name = "to-npy-stack-" + str(uuid.uuid1())  # create universally unique ID
    dsk = {
        (name, i): (np.savez_compressed, os.path.join(dirname, "%d.npz" % i), key)
        for i, key in enumerate(core.flatten(xx.__dask_keys__()))
    }

    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[xx])
    compute_as_if_collection(da.Array, graph, list(dsk))

    return meta

def to_npz_stack_other(dirname, x, axis=0):
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

    meta = {"chunks": chunks, "dtype": x.dtype, "axis": axis}

    with open(os.path.join(dirname, "info"), "wb") as f:
        pickle.dump(meta, f)

    # I want it to save files named birdID.npz containing all separate arrays
    # to specified dir
    name = "to-npy-stack-" + str(uuid.uuid1())  # create universally unique ID
    dsk = {
        (name, i): (np.savez_compressed, os.path.join(dirname, "%d.npz" % i), key)
        for i, key in enumerate(core.flatten(xx.__dask_keys__()))
    }

    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[xx])
    compute_as_if_collection(da.Array, graph, list(dsk))

    # dsk1 = {
    #    name : (np.savez_compressed,
    #           os.path.join(dirname, birdie),
    #           tuple(core.flatten(xx.__dask_keys__()))
    #           )
    # }
    #
    #graph1 = HighLevelGraph._from_collection(name, dsk1, collection=xx)
    #compute_as_if_collection(da.Array, graph1, list(dsk1))

    return meta

def main(arr, birdID, wdw=250):
    """
    Creates numpy dataframe by taking a rolling window of 250 rows of the input
    arr and horizontally arranging x, y, and z values followed by a binary
    value indicating whether or not a dive has occured in that window.

    :type birdname: str
    :param birdname: TagID of bird currently being processed
    :param arr: numpy array containing 4 columns: X, Y, Z, and Depth
    :return:
    """
    x = da.lib.stride_tricks.sliding_window_view(arr[:, 0], wdw)
    y = da.lib.stride_tricks.sliding_window_view(arr[:, 1], wdw)
    z = da.lib.stride_tricks.sliding_window_view(arr[:, 2], wdw)
    depth = da.lib.stride_tricks.sliding_window_view(arr[:, 3], wdw)

    d = da.apply_along_axis(check_for_dive, 1, depth)
    train_data = da.hstack((x, y, z, d.reshape((d.shape[0], 1))))

    meta = to_npz_stack(f'../Data/Acc/{birdID}', train_data, axis=0)

    # Informative output stats
    meta_out = {'shape': train_data.shape, 'files': len(train_data.chunks[0])}
    meta_out.update(meta)

    return meta_out

## Main ##

if __name__ == '__main__':
    # check inputs
    if len(sys.argv) != 2:
        pth = '../Data/BIOT_DGBP/BIOT_DGBP/'
        print(f"WARNING: incorrect number of args provided. Please provide "
              f"directory path as a single argument.\nDefault used: '{pth}'\n")
    else:
        pth = sys.argv[1]

    # Grab list of reevant file paths
    files = glob.glob(f'{pth}*1.csv')

    count = 0

    for no, file in enumerate(files, 1):

        tik = time.time()

        print(f"PROCESSING FILE {no}: '{file}'...")
        arr = dd.read_csv(file, usecols=['X', 'Y', 'Z', 'Depth']).to_dask_array(
            lengths=True)

        #arr = arr[:2500000]

        # Grab bird ID from filepath
        bird = re.search(r"/(\w+).csv", file).group(1)
        meta = main(arr, birdID=bird)

        # Informative output
        for k, v in meta.items():
            print('\t' + k + ':', v)

        tok = time.time() - tik

        print("Time elapsed: %.2fs" % tok)
        count += tok

    sys.exit('\nDone\n\nTotal time elapsed: %.2fs' % count)