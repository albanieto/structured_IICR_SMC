### VERSION 23/11 A LAS 00

import gyarados as gy
import random as ra
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import multiprocessing


def calc_m(M, N=1000, d=20):
    #M = 2N*m(d-1)
    #m = M/((d-1)*2N)
    m="{:.4E}".format(M/((d-1)*2*N))
    return m

def call_gyarados(d, N, m, M, Nt):
    M=str(M)
    sentence="sbatch gyarados_call.sh -n %d -N %d -m %s -b 100000000 -c 5 -t 1 -p 1 -i 1 -d 1e-8 -s 2 -M %s -P %s" % (d, N, m, M, Nt)
    print(d,N,m,M, Nt)
    print(sentence)
    os.system(sentence)
    return sentence

#N_tot=[4000, 20000, 40000, 70000, 100000]
#N_vector=[5000,4746,4494,4242,3988,3736,34384,3230,29278,2726,2472,2220,1968,1714,1462,1210,956,704,452,200]
N_vector=[2000]
#print(N_tot)
#d_vector=[2,5,10,50,100]
d_vector=[50]
#M_vector=[1. ,2.5 ,5 ,15,50]
#M_vector=[1.0,13.572088082974531,1.544452104946379,20.96144000826768,2.385332304473301,32.37394014347626,3.6840314986403864,49.99999999999999,5.689810202763908,8.7876393444041]
M_vector=[5]
if N_vector:
    for M in M_vector:
        for N in N_vector:
            for d in d_vector:
                m=calc_m(M, N, d)
                print(m)
                Nt=N*d
                call_gyarados(d,N,m,M, Nt)
else:
    for M in M_vector:
        for Nt in N_tot:
            for d in d_vector:
                N=Nt//d
                if N%2!=0:
                    print(N, "no par")
                    N=N+1
                    print("New N is", N)
                m=calc_m(M, N, d)
                print(m)
                call_gyarados(d,N,m,M, Nt)
