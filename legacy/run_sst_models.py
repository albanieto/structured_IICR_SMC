### VERSION 7/03 A LAS 00

import gyarados as gy
import random as ra
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import math
import multiprocessing


def calc_m_sst(M, N=1000):
    #M = 2N*m(d-1)
    #m = M/((d-1)*2N)
    m="{:.4E}".format(M/N)
    return m

def call_gyarados(L, N, m, M, sampled):
    #sentence="sbatch gyarados_call.sh -n %d -N %d -m %s -b 100000000 -c 5 -t 1 -p 1 -i 1 -d 1e-8 -s 20 -M %s -P %s" % (d, N, m, M, Nt)
    sentence="sbatch gyarados_call.sh -L %d -N %d -m %s -b 100000000 -c 5 -t 4 -p %s -i 1 -d 1e-8 -s 20 -M %2f" %(L,N, m, sampled,M)
    print(L,N,m,M, sampled)
    print(sentence)
    os.system(sentence)
    return sentence

def find_deme_size(N, L):
  """
  This function finds the closest integer to N/demes that satisfies the condition of being an even number.

  Args:
      N: Population size
      L: Demes side length (square lattice)

  Returns:
      The closest even integer to N/demes
  """
  demes = L * L
  average_deme_size = N / demes

  # Start with average_deme_size rounded down to the nearest integer
  deme_size = int(average_deme_size)

  # Adjust deme_size to be even (increment if odd)
  if deme_size % 2 != 0:
    deme_size += 1

  return deme_size


Nt_vector=[10000,20000,40000]
L_vector=[5,7,9]
M_vector=[0.5,1,2,5]
sampled_vector=["1,13","1,25","1,41"]
            
for M in M_vector:
    for nt in range(len(Nt_vector)):
        for li in range(len(L_vector)):
            L=L_vector[li]
            N=find_deme_size(Nt_vector[nt], L_vector[li])
            sampled=sampled_vector[li]
            print(sampled, L)
            m=calc_m_sst(M, N)
            print(m)
            call_gyarados(L,N,m,M,sampled)
