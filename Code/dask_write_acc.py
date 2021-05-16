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

## Functions ##

def check_for_dive(arr, thrshold=0.5):
    """
    Determines whether dive occured in given window of depth values.

    :param arr: array of depth values
    :param thrshold: depth value past which a dive is defined
    :return: boolean indicating whether dive has occured
    """
    return int((arr[~da.isnan(arr)] > thrshold).any())


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

    # Informative output stats
    shp = train_data.shape
    no_files = len(train_data.chunks[0])
    #for i in range(len(train_data.chunks[0])):
    #    print(f'\t{(train_data.chunks[0][i], sum(train_data.chunks[1]))}')

    da.to_npy_stack(f'../Data/Acc/{birdID}', train_data, axis=0)

    return shp, no_files

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

    for file in files:

        print(f'Processing file {file}...')
        arr = dd.read_csv(file, usecols=['X', 'Y', 'Z', 'Depth']).to_dask_array(
            lengths=True)

        #arr = arr[:2500]

        # Grab bird ID from filepath
        bird = re.search(r"/(\w+).csv", file).group(1)

        shp, no_files = main(arr, birdID=bird)

        print(f' - Saved array of shape {shp} to {no_files} .npy files')

    sys.exit('\nDone')