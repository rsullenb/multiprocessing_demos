#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 22:20:33 2017

@author: soolr
"""

#from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from multiprocessing import Pool, Value, Array, cpu_count
import time
import numpy as np
from itertools import product
from numba import jit

def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print('Elapsed time is ' + str(time.time() - startTime_for_tictoc) + ' seconds.')
    else:
        print('Toc: start time not set')

@jit(nopython=True)
def mc_img(y, x):
    s = 0
    trials = 100
    for i in range(trials):
        j = np.random.rand(30).sum()
        if j <= arr[y, x, :].sum():
            s += 1
    return s

@jit(nopython=True)
def mc_ry(y, x):
    s = 0
    trials = 1e3
    for i in range(trials):
        j = np.random.uniform(y, x)
        if j <= (y+x)/2:
            s += 1
    return s/trials

def main():
    #with Pool(processes=4) as pool:
    #    res = pool.starmap(processInput, [pos for pos in inputs])
    
    # USE THIS ONE
    #with Pool(processes=cpu_count()) as pool:
    #    res = pool.starmap(mc_ry, [pos for pos in inputs])#, chunksize=500)
    
    with Pool(processes=cpu_count()) as pool:
        res = pool.starmap(mc_ry, [pos for pos in inputs])
    
    #with ProcessPoolExecutor(max_workers=4) as pool:
    #    res = [pool.map(processInput, pos) for pos in inputs]
    #with concurrent.futures.ThreadPoolExecutor(max_workers=13) as executor:
    #    result = [executor.submit(processInput, pos) for pos in inputs]
    return res
    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #    return list(executor.map(processInput, inputs))
 
if __name__ == '__main__':
    img = np.random.rand(480,640)
    h, w = img.shape
    inputs = product(range(h), range(w))
    inputs2 = product(range(h), range(w))
    
    tic()
    i = np.array(main()).reshape(img.shape)
    toc()
    
    tic()
    i2 = np.array([mc_ry(loc[0],loc[1]) for loc in inputs2]).reshape(img.shape)
    toc()