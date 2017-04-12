#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 22:20:33 2017

@author: soolr
"""

import multiprocessing as mp
from multiprocessing import Pool, Value, Array, cpu_count
import time
import numpy as np
from itertools import product
import sharedmem
from numba import njit

def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print('Elapsed time is ' + str(time.time() - startTime_for_tictoc) + ' seconds.')
    else:
        print('Toc: start time not set')

@njit
def mc_img(y, x):
    s = 0
    trials = 50
    for i in range(trials):
        j = np.random.rand(arr.shape[2]).sum()
        if j <= arr[y, x, :].sum():
            s += 1
    return s

def main():
    with Pool(processes=cpu_count()) as pool:
        res = pool.starmap(mc_img, [pos for pos in inputs])
    return res
 
if __name__ == '__main__':
    img = np.random.rand(480,640,30)
    h, w = img.shape[:2]
    inputs = product(range(h), range(w))
    inputs2 = product(range(h), range(w))
    
    #unshared_arr = np.random.rand(480, 640)
    arr = sharedmem.empty(img.shape)
    arr[:] = img.copy()
    
    tic()
    i = np.array(main()).reshape(img.shape[:2])
    toc()
    
    tic()
    i2 = np.array([mc_img(loc[0],loc[1]) for loc in inputs2]).reshape(img.shape[:2])
    toc()