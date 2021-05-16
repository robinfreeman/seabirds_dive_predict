#!/usr/bin/env python3

""" Writes data set for deep learning model to predict dives using acceleration
data. """

__author__ = 'Luke Swaby (lds20@ic.ac.uk)'
__version__ = '0.0.1'

## Imports ##
import pandas as pd
import numpy as np
from dask import dataframe as dd
import dask.array as da
import time
#import csv
import sys
import os
import pickle


pth = '../Data/BIOT_DGBP/BIOT_DGBP/ch_gps03_S1.csv'
start = time.time()
df = dd.read_csv(pth, usecols=['X', 'Y', 'Z', 'Depth']).to_dask_array(lengths=True)
t3 = time.time() - start
print("Read csv with dask: ", t3, "sec")

del df
t = time.time()
df = pd.read_csv(pth, usecols=['X', 'Y', 'Z', 'Depth']).to_numpy()
print("Read csv with pd: ", time.time()-t, "sec")


## PANDAS ##
# start = time.time()
# wdw = 250
# thrshold = 0.5
# with open('../Data/acceleration_dset.csv', 'a') as f:
#    for ix in range(2500-1):  #range(len(df)-1):
#        X = df.X[ix:(ix + wdw)].tolist()
#        Y = df.Y[ix:(ix + wdw)].tolist()
#        Z = df.Z[ix:(ix + wdw)].tolist()
#        dive = (df.Depth[ix:(ix + wdw)].dropna() > thrshold).any()

#        writer = csv.writer(f)
#        writer.writerow(X + Y + Z + [int(dive)])
# f.write([X, Y, Z, dive])
# t1 = time.time() - start
# print(t1)


## NUMPY ##
# start = time.time()
"""
wdw = 250
thrshold = 0.5
with open('../Data/acceleration_dset.csv', 'a') as f:
    for ix in range(len(df) - 1):
        X = df[ix:(ix + wdw), 0]
        Y = df[ix:(ix + wdw), 1]
        Z = df[ix:(ix + wdw), 2]
        depth = df[ix:(ix + wdw), 3]
        dive = (depth[~np.isnan(depth)] > thrshold).any()
        row = np.append(np.concatenate((X, Y, Z)), int(dive))
        writer = csv.writer(f)
        writer.writerow(row)
"""

# t1 = time.time() - start
# print(t1)


# TODO:
# 1: import df as np array
#


def window(a, w_size=250, step=1, copy=False):
    """
    Writes 2D array from rolling window of 1D array

    Parameters
    ----------

    :param a: 1D array to extract from.
    :param w_size: window size.
    :param step: step size.
    :param copy: boolean indicating whether or not you want to write to the
                 windowed array, as otherwise it is a memory-sharing view.
    :return: 2D array of rolling windows of a.

    Example
    -------
    >> a=np.arange(6)
    >> window(a, w_size=2)
    array([[0, 1],
           [1, 2],
           [2, 3],
           [3, 4],
           [4, 5]])
    """
    shp = (a.size - w_size + 1, w_size)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=shp)[0::step]

    if copy:
        return view.copy()
    else:
        return view


def dive(arr, thrshold=0.5):
    """
    Determines whether dive occured in given window of depth values.

    :param arr: array of depth values
    :param thrshold: depth value past which a dive is defined
    :return: boolean indicating whether dive has occured
    """
    return int((arr[~np.isnan(arr)] > thrshold).any())


def main(arr):
    """
    Creates numpy dataframe by taking a rolling window of 250 rows of the input
    arr and horizontally arranging x, y, and z values followed by a binary
    value indicating whether or not a dive has occured in that window.

    :param arr: numpy array containing 4 columns: X, Y, Z, and Depth
    :return:
    """
    t = time.time()
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    d = np.apply_along_axis(dive, 1, window(arr[:, 3]))
    train_data = np.hstack((window(x), window(y), window(z),
                            d.reshape((d.shape[0], 1))))
    print(f'Time: {time.time()-t}s')

    # np.savetxt('../Data/acc_train_data.csv', train_data, delimiter=',')

    #print('Saving .npz file...')
    #np.savez_compressed('../Data/acc_train_data.npz', train_data)

    #pickle_out = open('../Data/acc_train_data.pickle', "wb")
    #pickle.dump(train_data, pickle_out)
    #pickle_out.close()

    # Note: this will temporarily more memory that np.savetxt as it must
    # convert to df, but is quicker overall
    pd.DataFrame(train_data).to_csv('../Data/acc_train_data.csv.gz',
                                    compression='gzip')

    t = time.time()
    pd.DataFrame(train_data).to_csv('../Data/to_csv.csv.gz',
                                    compression='gzip')
    t1 = time.time()-t
    print(f'pd.to_csv(): {t1}s')

    t = time.time()
    pd.DataFrame(train_data).to_csv('../Data/to_csv.csv', header=False)
    t3 = time.time() - t
    print(f'pd.to_csv(): {t3}s')

    t = time.time()
    np.savetxt('../Data/savetxt.csv', train_data, delimiter=',')
    t2 = time.time() - t
    print(f'np.savetxt(): {t2}s')


    return 0

if __name__ == '__main__':
    # check inputs
    if len(sys.argv) != 2:
        pth = '../Data/BIOT_DGBP/BIOT_DGBP/ch_gps03_S1.csv'
        print(f"WARNING: incorrect number of args provided. Please provide "
              f"file path as a single argument.\nDefault used: '{pth}'")
    else:
        pth = sys.argv[1]

    print('Reading in dataframe...')
    t=time.time()
    df = pd.read_csv(pth, usecols=['X', 'Y', 'Z', 'Depth']).to_numpy()
    print(time.time()-t)

    print('\rCreating new dataset...')
    status = main(df)

    print('Done!')
    sys.exit(status)

#//////////////////// SANBOX ////////////////////////////
import time
import pandas as pd
import os

# subset data
#df = df[:2500000]
df = df[:25000]

"""
## savetxt
file_name = "../Sandbox/time_npsavetxt.csv"
t = time.time()
np.savetxt(file_name, df, delimiter=",")
t1 = time.time()-t
print(f"np.savetxt(): {t1}s")
t = time.time()
data = np.loadtxt(file_name, delimiter=',')
t2 = time.time()-t
print(f"np.loadtxt(): {t2}s")
print(f'TOTAL: {t1+t2}s')
print(f'Filesize: {os.stat(file_name).st_size/1000000} MB\n\n')
"""

# pickle
file_name = "../Sandbox/time_pickle.pickle"
t = time.time()
pickle_out = open(file_name, "wb")
pickle.dump(df, pickle_out)
pickle_out.close()
t1 = time.time()-t
print(f"pickle.dump(): {t1}")
t = time.time()
pickle_in = open(file_name, "rb")
data = pickle.load(pickle_in)
t2 = time.time()-t
print(f"pickle.load(): {t2}s")
print(f'TOTAL: {t1+t2}s')
print(f'Filesize: {os.stat(file_name).st_size/1000000} MB\n\n')


## save
file_name = "../Sandbox/time_npsave.npy"
t = time.time()
np.save(file_name, df)
t1 = time.time()-t
print(f"np.save(): {t1}")
t = time.time()
data = np.load(file_name)
t2 = time.time()-t
print(f"np.load(): {t2}s")
print(f'TOTAL: {t1+t2}s')
print(f'Filesize: {os.stat(file_name).st_size/1000000} MB\n\n')

## savez_compressed
file_name = "../Sandbox/time_npsavez_compressed.npz"
t = time.time()
np.savez_compressed(file_name, df)
t1 = time.time()-t
print(f"np.savez_compressed(): {t1}s")
t = time.time()
data = np.load(file_name)
data = data[data.files[0]]
t2 = time.time()-t
print(f"np.load() (compressed): {t2}s")
print(f'TOTAL: {t1+t2}s')
print(f'Filesize: {os.stat(file_name).st_size/1000000} MB')