# -*- coding: utf-8 -*-

## IMPORTS


######### VERSION 23/11 a las 11 #############

import argparse
import os
import pandas as pd
import mmap
import shutil
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.ticker
#import warnings
# import allel
import gyarados as gy # my module
import sys
import inspect
import json
import math
import itertools
import time
#import warnings
import random as ra
import string
import re
import multiprocessing
import shlex
import ditto as gy_ditto


### add snif path
#sys.path.append("./snif")
#warnings.filterwarnings("ignore")
#from snif import *  # snif 


################# VECTORS ############


def logspace(limit,start, n):
    result = [start]
    n=n+1
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)[1:]

#lin_vector=np.linspace(50,100000, num=5000)
#log_vector=logspace(100000,50, 500)


lin_vector=np.linspace(50,100000, num=2000)
log_vector=gy.logspace(100000+1,50, 400)
log_vector=gy.logspace(5000000+1,50, 200)
log_vector_sst=gy.logspace(100000+1,50, 400)
log_vector_in=gy.logspace(50000+1,50, 200)
log_vector_in=log_vector
log_vector_sst=log_vector
#log_vector=gy.logspace(100000000+1,50, 40000)
#lin_vector=np.linspace(50,450000, num=2000)
#log_vector=gy.logspace(450000+1,20, 600) ## NISHA LOG VECTOR

DEFAULT_GYARADOS_MODE = "iicr,simulate,stats,psmc"
DEFAULT_PSMC_PATTERN = "4+25*2+4+6"
DEFAULT_PSMC_S = 100

MODE_ATOMS = {"iicr", "simulate", "stats", "psmc", "smcpp", "transition_matrix"}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def parse_run_modes(mode=None):
    """Normalize comma-separated run modes into executable pipeline steps."""
    if mode is None or str(mode).strip() == "":
        mode = DEFAULT_GYARADOS_MODE
    requested = [m.strip().lower().replace("-", "_").replace(" ", "_") for m in str(mode).split(",") if m.strip()]
    steps = set()
    for item in requested:
        if item == "none":
            continue
        elif item in MODE_ATOMS:
            steps.add(item)
        else:
            valid = sorted(MODE_ATOMS | {"none"})
            raise ValueError("Unknown mode '%s'. Valid modes: %s" % (item, ", ".join(valid)))
    return steps


def parse_psmc_patterns(psmc_patterns=None):
    """Return one or more PSMC -p time vectors."""
    if psmc_patterns is None or str(psmc_patterns).strip() == "":
        return [DEFAULT_PSMC_PATTERN]
    if isinstance(psmc_patterns, (list, tuple)):
        raw_patterns = psmc_patterns
    else:
        raw_patterns = re.split(r"[;,]", str(psmc_patterns))
    patterns = [str(p).strip().strip('"').strip("'") for p in raw_patterns if str(p).strip()]
    return patterns or [DEFAULT_PSMC_PATTERN]


def psmc_pattern_label(psmc_pattern):
    label = re.sub(r"[^A-Za-z0-9]+", "_", str(psmc_pattern)).strip("_")
    return label or "default"


def psmc_pattern_suffix(psmc_pattern):
    return "_p" + psmc_pattern_label(psmc_pattern)


def psmc_output_prefix(it, s, p, individual, psmc_pattern=DEFAULT_PSMC_PATTERN):
    suffix = psmc_pattern_suffix(psmc_pattern)
    return it.fullname+"_s"+str(s)+"_deme_"+str(p)+"_ind_"+str(individual)+suffix

#####################################################################################################################################################
######################################                      CLASS MODEL                  ############################################################
#####################################################################################################################################################

### MAIN CLASS ----> MODEL

class Model():
    '''
    - mod_type = type of model (str). Ex: eq_sim_n_island
    - mod_name = model name (str)
    - mod_n_islands = number of islands (int)
    - mod_N = haploid individuals in each island (list)
    - mod_nm = initial migration matrix (np.array)
    - mod_sizes = size of each island (list)
    - mod_n_mm = number of migration matrixes (int)
    - mod_n_events = number of events (int)
    - mod_events = demographic events (list of dirs)
    - mod_chr = number of chromosomes per individual (list)
    - mod_gr = growth rate (list or float)
    - mod_mu = mutation rate (float)
    - mod_rho = recombination rate (float)
    '''
    def __init__(self,d=None, params=None, mod_type=None, mod_name=None, mod_n_islands=None, mod_N=None, mod_samples=None, mod_mm=None, mod_sizes=None, mod_n_mm=None, mod_n_events=None,mod_events=None, mod_chr=None, mod_gr=None,mod_mu=1E-8, mod_rho=1E-8, mod_par=None, sam_demes=None, create_folder=True):
        '''
        Init function. Creates the object Model. 
        Each argument is defined inside the specific model subclass
        '''
        if os.path.exists(mod_name+"/"+mod_name+".json"):
            d=gy.get_from_json(mod_name+"/"+mod_name+".json")
                
        if d: # a dictionary has been parsed because the model already exists
            self.mtype=d["mtype"]
            if self.mtype != "panmictic":
                self.M=d["M"]
                self.islands=d["islands"]
                self.mig_matrix=d["mig_matrix"]
                self.n_mm=d["n_mm"]
                self.sampled_demes=d["sampled_demes"]

            self.N=d["N"]
            self.chr=d["chr"]  
            self.events=d["events"]
            self.gr=d["gr"]
            
          
            self.mu=d["mu"]
            self.n_events=d["n_events"]
            
            self.name=d["name"]
            self.par=d["par"]
            self.rho=d["rho"]
            self.samples=d["samples"]
            
            self.sizes=d["sizes"]
            self.iicr_lin_path=d["iicr_lin_path"]
            self.iicr_log_path=d["iicr_log_path"]
            self.id=d["id"]
           
            self.json=mod_name+"/"+mod_name+".json"


        else: # create the model
                self.name = mod_name
                self.mtype = mod_type
                self.islands=mod_n_islands
                self.N=mod_N
                self.samples=mod_samples
                self.mig_matrix=mod_mm
                self.sizes=mod_sizes
                self.n_mm=mod_n_mm
                self.n_events=mod_n_events
                self.events=mod_events
                self.chr=mod_chr
                self.mu = mod_mu
                self.rho = mod_rho
                self.gr = mod_gr
                self.id = ''.join(ra.choices(string.ascii_uppercase + string.digits, k=20))
                ## CONSIDER THAT MAY BE MORE THAN 1 IICR per model
                if self.mtype =="StSi" or self.mtype=="panmictic":
                    self.iicr_log_path = self.name+"/"+self.name+"_IICR/"+self.name+"_log_IICR.tsv.gz"
                    self.iicr_lin_path = self.name+"/"+self.name+"_IICR/"+self.name+"_lin_IICR.tsv.gz"
                elif self.mtype == "SST":
                    if len(self.sampled_demes)>1:
                     ## TWO SAMPLINGS SPECIFIED FOR THE MOMENT
                        self.iicr_log_path = {self.sampled_demes[0]:self.name+"/"+self.name+"_IICR/"+self.name+"_deme_"+str(self.sampled_demes[0])+"_log_IICR.tsv.gz",
                                            self.sampled_demes[1]:self.name+"/"+self.name+"_IICR/"+self.name+"_deme_"+str(self.sampled_demes[1])+"_log_IICR.tsv.gz"}
                        self.iicr_lin_path = {self.sampled_demes[0]:self.name+"/"+self.name+"_IICR/"+self.name+"_deme_"+str(self.sampled_demes[0])+"_lin_IICR.tsv.gz",
                                            self.sampled_demes[1]:self.name+"/"+self.name+"_IICR/"+self.name+"_deme_"+str(self.sampled_demes[1])+"_lin_IICR.tsv.gz"}
                    else: 
                         self.iicr_log_path = {self.sampled_demes[0]:self.name+"/"+self.name+"_IICR/"+self.name+"_deme_"+str(self.sampled_demes[0])+"_log_IICR.tsv.gz"}
                         self.iicr_lin_path = {self.sampled_demes[0]:self.name+"/"+self.name+"_IICR/"+self.name+"_deme_"+str(self.sampled_demes[0])+"_lin_IICR.tsv.gz"}

                elif self.mtype == "Free":
                    self.iicr_lin_path={}
                    self.iicr_log_path={}
                    for i in self.sampled_demes:
                        self.iicr_log_path[i]=self.name+"/"+self.name+"_IICR/"+self.name+"_deme_"+str(i)+"_log_IICR.tsv.gz"
                        self.iicr_lin_path[i]=self.name+"/"+self.name+"_IICR/"+self.name+"_deme_"+str(i)+"_lin_IICR.tsv.gz"

                if mod_par:
                    print ("Par file parsed...")
                    parfile=mod_par
                    if create_folder:
                        print("Creating a folder...")
                        sentence="mkdir "+self.name
                        os.system(sentence)
                        sentence="cp "+parfile+" "+self.name+"/"
                        os.system(sentence)
                        self.par=self.name+"/"+self.name+".par"
                    else:
                        self.par=self.name+".par"

                else:
                    self.par()
                self.params = self.params
                self.json=gy.get_json(self)
                self.sampled_demes = sam_demes
                
    def par(self):
        sentence="mkdir "+self.name
        os.system(sentence)
        parfile=self.name+"/"+self.name+".par"
        if not os.path.exists(parfile):
            with open(parfile,'w') as f:
                f.write("//Number of population samples (demes)\n")
                f.write(str(self.islands)+"\n")
                f.write("//Population effective sizes (number of genes 2*diploids)\n")
                [f.write(str(N)+"\n") for N in self.N]
                f.write("//Sample sizes (number of genes 2*diploids)"+"\n")
                [f.write(str(sample)+"\n") for sample in self.samples]
                f.write("//Grow rates: negative grow rates implies population expansion"+"\n")
                [f.write(str(gr)+"\n") for gr in self.gr]
                f.write("//Number of migration matrixes"+"\n")
                f.write(str(self.n_mm)+"\n")
                for i in range(self.n_mm):
                    this_mm=self.mig_matrix[i]
                    print(this_mm)
                    f.write("//migration matrix"+"\n")
                    for row in range(len(this_mm)):
                        for col in range(len(this_mm)):
                            f.write(str(self.mig_matrix[i][row][col])+" ")
                        f.write("\n")

                f.write("//historical event: time, source, sink, migrants, new deme size, new growth rate, migration matrix index" + "\n")
                f.write(str(self.n_events)+" events\n")
                if self.n_events>=1:
                    [f.write(str(event)+"\n") for event in self.events]  
                
                f.write("//Number of independent loci [chromosome] (Number of sequences of 300 bp per gamete)"+ "\n")
                f.write(str(self.chr)+" 0"+ "\n")
                for c in self.sizes:
                    f.write("//Chromosome structure 1 begins with number of loci\n1\n//per block: data type, number of loci, per generation recombination and mutation rates and optional parameters"+"\n")
                    f.write("DNA "+" "+str(c)+" "+str(self.rho)+" "+str(self.mu)+"\n")
        self.par=parfile
        return parfile

    
    '''
    def in_folder(self):
        sentence="mkdir "+self.name
        os.system(sentence)
    '''
        


## MODEL 1: EQ_SIM_N_ISLAND

class StSi(Model):
    '''
    Equilibrium simetric n-islands model
    - There are not any demographic events (equilibrium)
    - All the islands interchange migrants at the same rate
    - No grow rate
    - Only one migration matrix
    '''
    # Class definition

    def __init__(self,simn_n_islands=None, simn_N=None, simn_samples=None, simn_mig=None, simn_sizes=None,simn_chr=None, simn_mu=1E-8,simn_M=None, simn_Nt=None, simn_rho=1E-8, simn_gr=0, d=None, simn_events=[]):
        '''
        - simn_name: name of the simulation (str)
        - simn_n_islands: number of islands (int)
        - simn_N: haploid individuals in each islands (list). If is the same, provide a single number (int)
        - simn_mig: frequency of migrants (float)
        - simn_sizes: size of each chromosome (list)
        - simn_mu: mutation rate (float)
        - simn_rho: recombination rate (float)
        - simn_gr: growth rate (float)
        '''
        if d: # a dictionary has been parsed because the model  already exists
            self.N=d["N"]
            self.chr=d["chr"]
            self.events=d["events"]
            self.gr=d["gr"]
            self.islands=d["islands"]
            self.mig_matrix=d["mig_matrix"]
            self.mig=d["mig"]
            self.mu=d["mu"]
            self.n_events=d["n_events"]
            self.n_mm=d["n_mm"]
            self.name=d["name"]
            self.par=d["par"]
            self.rho=d["rho"]
            self.samples=d["samples"]
            self.mtype=d["mtype"]
            self.sizes=d["sizes"]
            self.params=d["params"]
            self.M=d["M"]
            self.Nt=d["Nt"]

        else: # create the model
            self.islands = simn_n_islands
            self.mtype="StSi"
            # self.N
            if type(simn_N)==int:
                self.N=[simn_N]*simn_n_islands
            elif len(simn_N.split(","))==simn_n_islands:
                self.N=simn_N.split(",")
            else: 
                raise ValueError("ERROR: Population sizes not matching the number of islands.\nIf all islands are the same, set only one size")
            
            # self.samples
            if type(simn_samples)==int:
                self.samples=[0]*simn_n_islands
                # I only need two islands to calculate the Fst
                self.samples[0]=simn_samples
                #self.samples[1]=simn_samples
            elif len(simn_samples.split(","))==simn_n_islands:
                self.samples=simn_samples.split(",")
            else: 
                raise ValueError("ERROR: Population samples not matching the number of islands.\nIf all islands are the same, set only one size of sample")

            self.mig=[simn_mig]
            self.mig_matrix()
            self.n_mm=1 # no events
            self.chr=simn_chr

            self.sizes=simn_sizes

            self.mu=simn_mu
            self.rho=simn_rho
            self.gr=[simn_gr]*self.islands
            self.events=simn_events
            self.n_events=len(simn_events) # equilibrium: no events
            self.params=["N", "d", "m"]
             # empty list of events
            self.name()  ################################################### mig in log
            if simn_M:
                self.M=simn_M
            else:
                self.M=2*self.N[0]*self.mig[0]*(self.islands-1)
            
            if simn_Nt:
                self.Nt=simn_Nt
            else:
                self.Nt=2*self.N[0]*self.islands

            # In a simmetric island model, the sampled demes to do the SFS are always the same
            # The first two demes

            self.sampled_demes=[1,2]
                
        

        Model.__init__(self,params=self.params, mod_type=self.mtype,mod_name=self.name,mod_n_islands=self.islands,mod_N=self.N,mod_samples=self.samples,mod_n_mm=self.n_mm,mod_mm=self.mig_matrix,mod_sizes=self.sizes, mod_n_events=self.n_events,mod_events=self.events,mod_chr=self.chr,mod_mu=self.mu,mod_rho=self.rho,mod_gr=self.gr,d=d)

    def mig_matrix(self):
        mm=np.full((self.islands,self.islands), self.mig)
        for i in range(self.islands):
            mm[i,i]=0
        self.mig_matrix=[mm]   
        return [mm]
    
    def name(self):
        # N
        quoteN="N"
        values, counts = np.unique(self.N, return_counts=True)
        for s in range(len(values)):
            q=str(counts[s])+"x"+str(values[s])
            quoteN=quoteN+"-"+q

        # Samples

        quoteS="sam"
        values, counts = np.unique(self.samples, return_counts=True)
        for s in range(len(values)):
            q=str(counts[s])+"x"+str(values[s])
            quoteS=quoteS+"-"+q

        # Sizes

        quoteB="size"
        values, counts = np.unique(self.sizes, return_counts=True)
        for s in range(len(values)):
            q=str(counts[s])+"x"+str(values[s])
            quoteB=quoteB+"-"+q


        # events
        quoteE=""
        if self.n_events!=0:
            for e in self.events:
                add_e="_e"+"_".join(e.split(" "))
                quoteE+=add_e
        else:
            quoteE="_no_events"
        print(quoteE)

        n="StSI_"+str(self.islands)+"I"+"_"+quoteN+"_"+quoteS+"_"+"m_"+str(round(math.log10(self.mig[0]),4))+"_"+quoteB+"_mu_"+"{:.1E}".format(self.mu)+"_gr_"+str(self.gr[0])+quoteE
        self.name=n
        return n


## MODEL 2: panmictic POPULATION

class panmictic(Model):
    '''
    Equilibrium simetric n-islands model
    - There are not any demographic events (equilibrium)
    - All the islands interchange migrants at the same rate
    - No grow rate
    - Only one migration matrix
    '''
    # Class definition

    def __init__(self, pan_N=None, pan_samples=None, pan_sizes=None, pan_chr=None, pan_gr=0, pan_mu=1E-8, pan_rho=1E-8, d=None, pan_events=[]):
        '''
        - simn_name: name of the simulation (str)
        - simn_n_islands: number of islands (int)
        - simn_N: haploid individuals in each islands (list). If is the same, provide a single number (int)
        - simn_mig: frequency of migrants (float)
        - simn_sizes: size of each chromosome (list)
        - simn_mu: mutation rate (float)
        - simn_rho: recombination rate (float)
        - simn_gr: growth rate (float)
        '''
        if d: # a dictionary has been parsed because the model  already exists
            self.N=d["N"]
            self.chr=d["chr"]
            self.events=d["events"]
            self.gr=d["gr"]
            self.islands=d["islands"]
            self.mig_matrix=d["mig_matrix"]
            self.mig=d["mig"]
            self.mu=d["mu"]
            self.n_events=d["n_events"]
            self.n_mm=d["n_mm"]
            self.name=d["name"]
            self.par=d["par"]
            self.rho=d["rho"]
            self.samples=d["samples"]
            self.mtype=d["mtype"]
            self.sizes=d["sizes"]
            self.params=d["params"]

        else: # create the model
            self.islands = 1
            self.mtype="panmictic"
            # self.N
            self.N=[pan_N]

            self.samples=[pan_samples]
            self.mig=0
            self.mig_matrix=0
            self.n_mm=0 # no events
            self.chr=int(pan_chr)
            

            self.sizes=pan_sizes

            self.mu=pan_mu
            self.rho=pan_rho
            self.gr=[pan_gr]
            self.n_events=len(pan_events) 
            self.events=pan_events 
            self.params=["N"]
            self.name()
                
        

        Model.__init__(self,params=self.params, mod_type=self.mtype,mod_name=self.name,mod_n_islands=self.islands,mod_N=self.N,mod_samples=self.samples,mod_n_mm=self.n_mm,mod_mm=self.mig_matrix,mod_sizes=self.sizes, mod_n_events=self.n_events,mod_events=self.events,mod_chr=self.chr,mod_mu=self.mu,mod_rho=self.rho,mod_gr=self.gr,d=d)

    
    def name(self):
        # N
        quoteN="N"
        values, counts = np.unique(self.N, return_counts=True)
        for s in range(len(values)):
            q=str(counts[s])+"x"+str(values[s])
            quoteN=quoteN+"-"+q

        # Samples

        quoteS="sam"
        values, counts = np.unique(self.samples, return_counts=True)
        for s in range(len(values)):
            q=str(counts[s])+"x"+str(values[s])
            quoteS=quoteS+"-"+q

        # Sizes

        quoteB="size"
        values, counts = np.unique(self.sizes, return_counts=True)
        for s in range(len(values)):
            q=str(counts[s])+"x"+str(values[s])
            quoteB=quoteB+"-"+q

        # Events
        quoteE=""
        if self.n_events!=0:
            for e in self.events:
                add_e="_e"+"_".join(e.split(" "))
                quoteE+=add_e
        else:
            quoteE="_no_events"


        print(quoteE)
        n="panmictic"+"_"+quoteN+"_"+quoteS+"_"+quoteB+"_mu_"+"{:.1E}".format(self.mu)+"_gr_"+str(self.gr[0])+quoteE
        self.name=n
        self.name=n
        return n
    

## MODEL 3: FREE MODEL

class Free(Model):
    # Use a par file or specify all the arguments separately to define this model
    def __init__(self, parfile=None, d=None, N=None,islands=None, samples=None, mu=None, gr=None, rho=None, n_events=None, chr=None, events=None, sizes=None, name=None, mig_matrix=None, n_mm=None, create_folder=True):
        if d: # a dictionary has been parsed because the model  already exists
            self.N=d["N"]
            self.chr=d["chr"]
            self.events=d["events"]
            self.gr=d["gr"]
            self.islands=d["islands"]
            self.mig_matrix=d["mig_matrix"]
            self.mu=d["mu"]
            self.n_events=d["n_events"]
            self.n_mm=d["n_mm"]
            self.name=d["name"]
            self.par=d["par"]
            self.rho=d["rho"]
            self.samples=d["samples"]
            self.mtype=d["mtype"]
            self.sizes=d["sizes"]
            self.sampled_demes=d["sampled_demes"]
            self.Nt=d["Nt"]
            self.M=d["M"]
            self.params=d["params"]
        elif parfile: # parse it directly from par file
            self.mig="Complex"
            self.M="Complex"
            self.par=parfile
            link_to_par_file=parfile
            self.name=os.path.basename(link_to_par_file).split('.par')[0]
            print("Model name:", self.name)

            with open(link_to_par_file, "r") as p:
                all_file=p.readlines()
                all_file=[line.strip() for line in all_file]

            bool_headers = [line.startswith("//") for line in all_file]
            headers_index= list(filter(lambda i: bool_headers[i], range(len(bool_headers))))
            headers = [all_file[i] for i in headers_index]

            # Get islands
            print("Retrieving islands...")
            str_islands=all_file[1]
            self.islands=int(re.findall("^\d+",str_islands)[0])
            print("There are %d islands" % self.islands)


            # Get population sizes
            print("Retrieving deme sizes...")
            self.N=[]
            line_index=headers_index[1]+1
            while line_index!=headers_index[2]:
                self.N.append(int(all_file[line_index]))
                line_index=line_index+1
            print("Deme sizes are", *self.N)

            # Get sample sizes
            print("Retrieving sample sizes...")
            self.samples=[]
            line_index=headers_index[2]+1
            while line_index!=headers_index[3]:
                self.samples.append(int(all_file[line_index]))
                line_index=line_index+1
            print("Sample sizes are", *self.samples)

            # get grow rates
            print("Retrieving grow rates...")
            self.gr=[]
            line_index=headers_index[3]+1
            while line_index!=headers_index[4]:
                self.gr.append(all_file[line_index])
                line_index=line_index+1

            print("Grow rates are", *self.gr)

            # Get migration matrixes


            self.n_mm=int(all_file[headers_index[4]+1])
            print("There are %d migration matrixes" % self.n_mm)

            if self.n_mm:
                self.mig_matrix=[]
                hi=5
                str_n_events=all_file[headers_index[5]+self.n_mm+1]
                for i in range(self.n_mm): # two times (0 and 1)
                    print("Retrieving matrix number %d" % (i+1))
                    this_mig_matrix=[]
                    line_index=headers_index[hi+i]+1
                    while line_index!=headers_index[hi+i+1]:
                        this_line=[float(x) for x in all_file[line_index].split(" ")]
                        this_mig_matrix.append(this_line)
                        line_index=line_index+1
                    last_index=headers_index[hi+i+1]
                    self.mig_matrix.append(this_mig_matrix)
            else:
                hi=5
                
                str_n_events=all_file[headers_index[5]+1]
                last_index=headers_index[5]
                self.mig_matrix=[]

            # Get events 
            
            
            self.n_events=int(re.findall("^\d+",str_n_events)[0])
            print("Retrieving %d events" % self.n_events)
            self.events=[]
            for i in range(self.n_events):
                self.events.append(all_file[last_index+2+i])
            print(*self.events, sep="\n")


            #  Get chromosome sizes
            print("Retrieving chromosomes...")

            last_header=headers_index[headers_index.index(last_index)+1]
            str_nchr=all_file[headers_index[headers_index.index(last_index)+1]+1]
            self.chr=int(str_nchr.split(" ")[0])
            print("Number of chromosomes is %d" % self.chr)
            type_of_data=int(str_nchr.split(" ")[1])
            index_header=headers_index.index(last_header)+2
            if type_of_data: # chr structure is different (1)
                print("Different chromosome structure. Sizes are:")
                # 2 headers for each chromosome
                self.sizes=[]
                for i in range(self.chr):
                    str_size=all_file[headers_index[index_header]+1]
                    this_chr_size=int(re.findall('\d+',str_size)[0]) # the first digit is the chromosome size
                    self.sizes.append(this_chr_size)
                    print(this_chr_size)
                    index_header+=2
            else: # same structure (0)
                print("Same chromosome structure. Size is")
                str_size=all_file[headers_index[index_header]+1]
                this_chr_size=int(re.findall('\d+',str_size)[0]) # the first digit is the chromosome size
                self.sizes=[this_chr_size]*self.chr
                print(self.sizes[0], "bp")

            # Get mutation and rho
            print("Mutation and rho are assumed to be the same for all chromosomes")
            self.mu=float(str_size.split(" ")[-1])
            self.rho=float(str_size.split(" ")[-2])
            print("mu:", self.mu)
            print("rho:", self.rho)
        
            self.params=["d", "n_mm", "events", "gr", "name"]
            self.mtype="Free"
            self.sampled_demes=list(range(1,self.islands+1))
            self.Nt=sum([self.N[i] for i in range(self.islands)])    

        Model.__init__(self,params=self.params, mod_type=self.mtype,mod_name=self.name,mod_n_islands=self.islands,mod_N=self.N,mod_samples=self.samples,mod_n_mm=self.n_mm,mod_mm=self.mig_matrix,mod_sizes=self.sizes, mod_n_events=self.n_events,mod_events=self.events,mod_chr=self.chr,mod_mu=self.mu,mod_rho=self.rho,mod_gr=self.gr,mod_par=self.par,d=d, create_folder=create_folder)

class SST(Model):
    '''
    2D Stepping stone model
    '''
    # Class definition

    def __init__(self,sst_L=None, sst_N=None, sst_samples=None, sst_sampled_demes = [], sst_mig=None, sst_sizes=None, sst_chr=None, sst_square=True, sst_mu=1E-8, sst_M=None, sst_Nt=None, sst_rho=1E-8, sst_gr=0, d=None, sst_events=[]):
        '''
        - sst_nislands = number of demes
        - sst_N = deme size (its simmetrical)
        - sst_samples = samples
        - sst_mig = migration rate (its simmetrical)
        - sst_sizes = chromosome size
        - sst_square = If the grid is square (True)
        - sst_mu
        - sst_M = scaled migration rate (its simmetrical)
        - sst_Nt = Metapopulation size
        - sst_rho 
        - sst_gr = grow rate
        - sst_events = [an array of events]
        '''
        if d: # a dictionary has been parsed because the model  already exists
            self.N=d["N"]
            self.chr=d["chr"]
            self.events=d["events"]
            self.gr=d["gr"]
            self.L=d["L"]
            self.islands=d["islands"]
            self.mig_matrix=d["mig_matrix"]
            self.mig=d["mig"]
            self.mu=d["mu"]
            self.n_events=d["n_events"]
            self.n_mm=d["n_mm"]
            self.name=d["name"]
            self.par=d["par"]
            self.rho=d["rho"]
            self.samples=d["samples"]
            self.mtype=d["mtype"]
            self.sizes=d["sizes"]
            self.params=d["params"]
            self.M=d["M"]
            self.Nt=d["Nt"]
            self.sampled_demes=d["sampled_demes"]
            self.Nm=d["Nm"]

        else: # create the model
            self.L=sst_L
            sst_nislands=self.L**2
            self.islands = sst_nislands
            self.mtype="SST"
            # self.N
            if type(sst_N)==int:
                self.N=[sst_N]*sst_nislands
            elif len(sst_N.split(","))==sst_nislands:
                self.N=sst_N.split(",")
            else: 
                raise ValueError("ERROR: Population sizes not matching the number of islands.\nIf all islands are the same, set only one size")
            
            self.sampled_demes=sst_sampled_demes
            # FEX: demes samples = [1,13] (from 5x5)
            
            if type(sst_samples)==int:
                self.samples=[0]*sst_nislands
                self.sampled_demes=[self.sampled_demes]
                # Putting samples in the one I need to
                #self.samples[self.sampled_demes[0]-1]=sst_samples
                #self.samples[self.sampled_demes[1]-1]=sst_samples ################################################ THIS
                self.samples[self.sampled_demes[0]-1]=sst_samples

            self.mig=[sst_mig]
            self.mig_matrix()
            self.n_mm=1 # no events
            self.chr=sst_chr

            self.sizes=sst_sizes

            self.mu=sst_mu
            self.rho=sst_rho
            self.gr=[sst_gr]*self.islands
            self.events=sst_events
            self.n_events=len(sst_events) # equilibrium: no events
            self.params=["N", "m", "Nm", "L", "deme"]
             # empty list of events
            self.name()
            if sst_M:
                self.M=sst_M
        
            else:
                self.M=2*self.N[0]*self.mig[0]
            self.Nm=self.M
            self.Nt=self.N[0]*self.islands

            # In a simmetric island model, the sampled demes to do the SFS are always the same
            # The first two demes

            
                
        

        Model.__init__(self,params=self.params, mod_type=self.mtype,mod_name=self.name,mod_n_islands=self.islands,mod_N=self.N,mod_samples=self.samples,mod_n_mm=self.n_mm,mod_mm=self.mig_matrix,mod_sizes=self.sizes, mod_n_events=self.n_events,mod_events=self.events,mod_chr=self.chr,mod_mu=self.mu,mod_rho=self.rho,mod_gr=self.gr, d=d)

    def mig_matrix(self):  
        L=self.L
        nislands=self.islands
        m=self.mig[0]
        arr=np.arange(1,nislands+1)
        grid=arr.reshape(L,L)
        center=math.ceil(nislands/2)
        matrix=np.zeros(shape=(nislands, nislands))
        print(matrix.shape)
        for pos_island in range(0,nislands):
            island=pos_island+1
            # Convert the mask to an array of indices using np.where
            indices = np.where(grid==island)
            row, column = indices
            row=row[0]
            column=column[0]
            surr_positions=[(row,column-1), (row,column+1), (row+1,column), (row-1,column)]
            surr_demes=[]
            for position in surr_positions:
                if position[0]<0 or position[1]<0 or position[0]>L-1 or position[1]>L-1: ## list index out of range
                    continue
                else:
                    # find the surrounding deme
                    surr_deme=grid[position[0]][position[1]]
                    surr_demes.append(surr_deme)
            print("For deme", island, "surrounding demes are:", *surr_demes)
            # Change these values in this line of the matrix
            for deme in surr_demes:
                matrix[pos_island][deme-1]=m
        self.mig_matrix=[matrix]
        return [matrix]
    
    def name(self):
        # N
        quoteN="N"
        values, counts = np.unique(self.N, return_counts=True)
        for s in range(len(values)):
            q=str(counts[s])+"x"+str(values[s])
            quoteN=quoteN+"-"+q

        # Samples

        quoteS="sam"
        values, counts = np.unique(self.samples, return_counts=True)
        for s in range(len(values)):
            q=str(counts[s])+"x"+str(values[s])
            quoteS=quoteS+"-"+q

        # Sizes

        quoteB="size"
        values, counts = np.unique(self.sizes, return_counts=True)
        for s in range(len(values)):
            q=str(counts[s])+"x"+str(values[s])
            quoteB=quoteB+"-"+q


        # events
        quoteE=""
        if self.n_events!=0:
            for e in self.events:
                add_e="_e"+"_".join(e.split(" "))
                quoteE+=add_e
        else:
            quoteE="_no_events"
        print(quoteE)
        print("mig is", self.mig[0])
        print("the type is : ", type(self.mig[0]))

        n="SST_"+str(self.L)+"x"+str(self.L)+"_"+quoteN+"_"+quoteS+"_"+"m_"+str(round(math.log10(float(self.mig[0])),4))+"_"+quoteB+"_mu_"+"{:.1E}".format(float(self.mu))+"_gr_"+str(self.gr[0])+quoteE
        self.name=n
        return n


## REPETITION CLASS
    
class Repetition_model():

    '''
    Each repetition
    '''
    
    def __init__(self,mod,rep,p):
        '''
        ins
        - mod: MK model object
        - rep : int
        - p : str
        '''
        
        self.name = mod.name+"_rep"+str(rep)
        self.fullname = mod.name+"/"+self.name+"/"+self.name
        self.dir = mod.name+"/"+self.name
        self.old_gen_path = self.fullname+"_1_1.gen"
        self.gen_path = self.fullname+"_1_1.gen.gz"
        self.par = mod.name+"/"+self.name+".par"
        self.pop=p
        if mod.mtype!="panmictic":
            self.blueprint=self.fullname+"_pop"+str(p)+".blueprint"
        else:
            self.blueprint=self.fullname+".blueprint"
        self.rep=rep

########################################################################################################################
###########################                      PARSE FUNCTIONS                             ############################
########################################################################################################################



def parse_gy_args(callist):
    '''
    Function that, reads the callist
    For the moment only reads till eq_sim_n_islands
    Can be easily extended to more arguments 
    '''
    # nislands
    nislands=int(callist[1])

    # Ndeme
    if len((callist[2]).split(","))==1:
        Ndeme=int(callist[2])
    else:
        Ndeme=callist[2].split(",")

    # mig
    mig=float(callist[3])

    # sample
    if len(callist[4].split(","))==1:
        sample=int(callist[4])
    else:
        sample=callist[4].split(",")

    # Chromosome size
    if len(callist[5].split(","))==1:
        sizes=int(callist[4])
    else:
        sizes=callist[5].split(",")
    return nislands,Ndeme,mig,sample,sizes

def get_json(model):
    '''
    Create a json with the model attributes
    '''

    atts = [ p for p in inspect.getmembers(model) if not(p[0].startswith('__'))]
    atts=dict(atts)
    sentence= "mkdir -p "+model.name
    os.system(sentence) 
    #if not(type(atts["mig_matrix"])==int or type(atts["mig_matrix"])==list):
   
    if model.mtype !="Free":
        if model.mtype != "panmictic":
            print(model.mtype)
            print(atts["mig_matrix"])
            print(len(atts["mig_matrix"]))
            print(type(atts["mig_matrix"]))
            atts["mig_matrix"]=[mim.tolist() for mim in atts["mig_matrix"]]
        #atts["mig_matrix"]=atts["mig_matrix"].tolist()
    j_name=model.name+"/"+model.name+".json"
    with open(j_name, 'w') as fp:
        print(model.sizes)
        json.dump(atts, fp)
    return j_name

def get_from_dict(mod,d):

    if d["mtype"]=="StSi":
        # (self,simn_n_islands, simn_N, simn_samples, simn_mig, simn_sizes, simn_mu=1E-8, simn_rho=1E-8, simn_gr=0)
        mod.N=d["N"]
        mod.chr=d["chr"]
        mod.events=d["events"]
        mod.gr=d["gr"]
        mod.islands=d["mig"]
        mod.mig_matrix=d["mig_matrix"]
        mod.mu=d["mu"]
        mod.n_events=d["n_events"]
        mod.n_mm=d["n_mm"]
        mod.name=d["name"]
        mod.par=d["par"]
        mod.rho=d["rho"]
        mod.samples=d["samples"]
        mod.mtype=d["mtype"]
    else: 
        pass
    return


def get_from_json(j_son):
    '''
    Use to read the model.json files and use the dictionary to fill the model class
    Input:
    - json file path
    Output:
    - dictionary
    '''
    with open(j_son) as json_file:
        data = json.load(json_file)
    return data
'''
def lfile_model(self,d):
        #(self, mod_type, mod_name, mod_n_islands, mod_N, mod_samples, mod_mm, mod_sizes, mod_n_mm, mod_n_events,mod_events, mod_chr, mod_gr, mod_mu=1E-8, mod_rho=1E-8)
        if d.type=="StSi":
            # (self,simn_n_islands, simn_N, simn_samples, simn_mig, simn_sizes, simn_mu=1E-8, simn_rho=1E-8, simn_gr=0)
            self.N=d.N
            self.chr=d.chr
            self.events=d.events
            self.gr=d.gr
            self.islands=d.mig
            self.mig_matrix=d.mig_matrix
            self.mu=d.mu
            self.n_events=d.n_events
            self.n_mm=d.n_mm
            self.name=d.name
            self.par=d.par
            self.rho=d.rho
            self.samples=d.samples
            self.type=d.type 
'''

###################### 1D-SST #############################################


def write_1d_ss_par(filepath, N, d, m):
    """
    Create a FastSimCoal2 .par file for a 1D stepping-stone:
    - N: haploid size per deme (identical for all)
    - d: number of demes
    - m: migration rate between adjacent demes
    - Sample size = 2 in central deme, 0 elsewhere
    Tail after migration matrix is fixed as specified.
    """
    central = d // 2  # zero-based index of central deme
    with open(filepath, "w") as f:
        # 1) number of demes
        f.write("//Number of population samples (demes)\n")
        f.write(f"{d}\n")

        # 2) effective sizes
        f.write("//Population effective sizes (number of genes 2*diploids)\n")
        for _ in range(d):
            f.write(f"{N}\n")

        # 3) sample sizes (only central deme gets 2)
        f.write("//Sample sizes (number of genes 2*diploids)\n")
        for i in range(d):
            f.write("2\n" if i == central else "0\n")

        # 4) growth rates (all zero)
        f.write("//Grow rates: negative grow rates implies population expansion\n")
        for _ in range(d):
            f.write("0\n")

        # 5) one migration matrix
        f.write("//Number of migration matrixes\n")
        f.write("1\n")
        f.write("//migration matrix\n")
        for i in range(d):
            row = []
            for j in range(d):
                if abs(i - j) == 1:
                    row.append(f"{m:.8f}")
                else:
                    row.append("0")
            f.write(" ".join(row) + "\n")

        # 6) no historical events
        f.write("//historical event: time, source, sink, migrants, new deme size, new growth rate, migration matrix index\n")
        f.write("0 events\n")

        # 7) ten 100 Mb chromosomes, μ=ρ=1e-8
        f.write("//Number of independent loci [chromosome] (Number of sequences of 300 bp per gamete)\n")
        f.write("10 0\n")
        for _ in range(10):
            f.write("//Chromosome structure 1 begins with number of loci\n")
            f.write("1\n")
            f.write("//per block: data type, number of loci, per generation recombination and mutation rates and optional parameters\n")
            f.write("DNA  100000000 1e-08 1e-08\n")

def par_filename_1d_ss(N, d, m):
    """
    Generate a descriptive filename for a 1D stepping-stone .par file.

    Format:
        1DSST_d_N_log10(m)_10x100000000.par

    - d: number of demes
    - N: haploid size per deme
    - m: migration rate

    Here log10(m) is rounded to 4 decimal places.
    """
    print("Creating par name  with m", m)
    logm = math.log10(m)
    return f"1DSST_{d}_{N}_{logm:.4f}_10x100000000.par"

def run_1d_free_model(N, d, m):
    """
    1. Use par_filename_1d_ss() to make a descriptive .par filename.
    2. Use write_1d_ss_par() to write that .par to disk.
    3. Call GYARADOS_PAR with type=3 (Free) and niter=1 on that .par.

    Returns the Model object that GYARADOS_PAR creates.
    """
    # 1) descriptive name
    filename = par_filename_1d_ss(N, d, m)
    
    # 2) write the .par file
    write_1d_ss_par(filename, N, d, m)
    
    # 3) launch a Free‐model run with GYARADOS_PAR
    model = gy.GYARADOS_PAR(type=3, par=filename, niter=1, p=d//2 ) # zero-based index of central deme)
    return model

########################################################################################################################
#################################               MODEL RUN FUNCTION                        ##############################
########################################################################################################################

def GYARADOS_PAR(type,L=None, N=None,sizes=None,sample=None,niter=None,chr=None,nislands=None,mig=None,p=1,mu=1e-8,rho=1e-8, evs=[], gr=0, par=None,M=None, Nt=None, mode=DEFAULT_GYARADOS_MODE, psmc_patterns=None, psmc_s=DEFAULT_PSMC_S, ditto=False):
    
    """
    Runs all the eq_sim_n_islands model  in parallel. 
    Calls for slurm job
    """
    print("Welcome to gyarados! Version 16.11.2023")
    run_modes = gy.parse_run_modes(mode)
    psmc_patterns = ",".join(gy.parse_psmc_patterns(psmc_patterns))
    psmc_s = str(psmc_s)
    ditto = gy.parse_bool(ditto)
    print("Run modes:", ",".join(sorted(run_modes)) if run_modes else "none")
    print("PSMC -p vectors:", psmc_patterns)
    print("PSMC -s:", psmc_s)
    print("Ditto:", ditto)
    os.system("mkdir -p RESULTS") # create results directory if it doesn't exist yet

    # TIME LOG FILE
    if (os.path.exists("gyarados_time_check.log") == False): # create log file for timinimg each task if doesnt exist
        f = open("gyarados_time_check.log", "w")
        line="\t".join(["MODEL_NAME", "TASK_TYPE", "TIME(S)", "BIN_SIZE/START_POINT"]) + "\n"
        f.write(line)
        f.close()

    # MODEL LOG FILE

    if (os.path.exists("gyarados_models.log") == False): # create log file for timinimg each task if doesnt exist
        x = open("gyarados_models.log", "w")
        if type==str(1):
            line="\t".join(["MODEL_NAME","MODEL_ID","N", "m", "d", "M","Nt","reps"])+"\n"
        elif type==str(4):
            line="\t".join(["MODEL_NAME","MODEL_ID","N", "m", "L", "M","reps"])+"\n"
        elif type==str(3): # Free model
            line="\t".join(["MODEL_NAME","MODEL_ID", "d", "n_mm", "events", "gr", "name"])+"\n"
        else:
            line="\t".join(["MODEL_NAME","MODEL_ID","N", "m", "d", "M","Nt","reps"])+"\n"
        x.write(line)
        x.close()

    
    if par:
        type=str(3)     
        r=gy.Free(par)
    
    else:

        size = sizes.split(',')
        chr=int(chr)
        values, counts = np.unique(size, return_counts=True)
        if len(values)==1 and chr!=1:
            sz=size*chr
            sz=list(map(int,sz))
        else:
            sz=list(map(int,size))
        niter=int(niter)
        print(evs)
        if evs!=str(0):
            evs_space=evs.replace(","," ")
            events=evs_space.split("/") # events are parsed as a string of comma separated numbers
        else:
            events=[]
        if int(type)==1:
            print("StSi model")
            if not nislands:
                return "no nislands specified"
            if not mig:
                return "no mig specified"
            r=gy.StSi(simn_n_islands=int(nislands),simn_N=int(N),simn_samples=int(sample),simn_mig=float(mig),simn_sizes=sz, simn_chr=chr, simn_mu=float(mu), simn_events=events, simn_gr=gr,simn_M=float(M), simn_Nt=Nt) # model object
        if int(type)==2:
            print("panmictic model")
            r=gy.panmictic(pan_N=int(N), pan_samples=int(sample), pan_sizes=sz, pan_chr=chr, pan_mu=float(mu),pan_rho=rho, pan_events=events, pan_gr=gr)
            p=1

        if int(type)==4:
            print("2D Stepping stone model")
            print("Two sampled demes implemented...")
            sampled_demes=list(map(int,p.split(",")))
            if len(sampled_demes)==2:
                print("sampled demes are:",sampled_demes[0], "and", sampled_demes[1])
                r=gy.SST(sst_L=int(L),sst_N=int(N), sst_samples=int(sample), sst_sampled_demes=sampled_demes,  sst_sizes=sz,sst_mig=mig, sst_chr=chr,sst_mu=float(mu), sst_M=float(M), sst_rho=rho, sst_gr=gr, sst_events=events)################
            else:
                print("One single sampled deme specified")
                print("Sampled deme is", int(sampled_demes[0]))
                r=gy.SST(sst_L=int(L),sst_N=int(N), sst_samples=int(sample), sst_sampled_demes=int(sampled_demes[0]),  sst_sizes=sz,sst_mig=mig, sst_chr=chr,sst_mu=float(mu), sst_M=float(M), sst_rho=rho, sst_gr=gr, sst_events=events)

    j_name=r.json
    print("json file for this model is allocated in", r.json)

    # Create RESULTS/model
    sentence="mkdir -p RESULTS/"+r.name

    # ADD MODEL LOG
    with open("gyarados_models.log", "a") as x:
        if int(type)==1:
            Nt=N
            print(r.name,r.id, str(r.N[0]),str(r.mig[0]),str(r.islands),str(M),str(Nt),str(niter), sep="\t", file=x)
        if int(type) == 2: 
            print(r.name,r.id, str(r.N), sep="\t", file=x)
            
        elif int(type)==4: 
            #line="\t".join(["MODEL_NAME","MODEL_ID","N", "m", "L", "M","reps"])+"\n"
            print(r.name, r.id, str(r.N[0]),str(r.mig[0]),str(r.L), str(r.Nm), str(niter), file=x)
        elif int(type)==3: # Free model
            print(r.name, r.id,  str(r.islands), str(r.n_mm), str(r.n_events), str(r.gr[0]), r.name, sep="\t", file=x)
        else:
            print(r.name,r.id," ".join([str(N) for N in r.N]),str(r.mig),str(r.islands),str(niter), sep="\t", file=x)


    print(sentence)
    os.system(sentence)
    # Copy json file
    sentence="cp "+j_name+" RESULTS/"+r.name+"/"
    print(sentence)
    os.system(sentence)

    sentence="cp "+r.par+" RESULTS/"+r.name+"/"
    print(sentence)
    os.system(sentence)

    ### Create dum fa PSMC
    dumfa=True  ############### THIS
    if dumfa:
        for c in range(len(list(r.sizes))):
            chr=c+1
            size=int(r.sizes[c])
            print("Creating dum fa")
            tmp_fa=r.name+"/"+r.name+"_"+str(chr)+".fa"# temporary fa for chromosome
            #line=("A"*60+"\n") 
            with open(tmp_fa,"w") as f:
                f.write(">"+str(chr)+"\n")
                f.write("A"*size+"\n")
                print(tmp_fa, "has been created")
    
    #### Each run a new bash launch 
    print("A total number of: ", str(niter)," iterations")

    ### Calculate IICR
    if "iicr" in run_modes:
        print("Calculating IICR...")
        if int(type)==4:
            print("Stepping stone model IICR calculation...")
            print("One IICR per each sampled population")
            print("Sampled demes are:", sampled_demes)
            for p in sampled_demes:
                print("Starting IICR with population ", p)
                gy.IICR_fullrun(r,int(p))
                sentence="rm -r "+r.name+"/*fn*fs*"
                os.system(sentence)
        #elif int(type)==3: ## FREE MODEL
        #    for p in range(1,int(r.islands)+1):
        #        print("Calculating IICR")
        #        gy.IICR_fullrun(r,int(p))
        elif int(type)==3: ### FREE p
            print("Calculating IICR with", p)
            gy.IICR_fullrun(r,int(p))

        else:
            print("Calculating IICR")
            gy.IICR_fullrun(r,int(p))
        sentence="rm -r "+r.name+"/*fn*fs*"
        os.system(sentence)
    else:
        print("Skipping IICR because mode is", mode)
#########################################################################################

            ###################### AQUI 

    if ditto:
        gy_ditto.run_ditto_for_model(
            r,
            p,
            niter=niter,
            mode=mode,
            psmc_patterns=psmc_patterns,
            psmc_s=psmc_s,
            run_gyarados=gy.GYARADOS_PAR
        )

    worker_modes = run_modes.intersection({"simulate", "stats", "psmc", "smcpp", "transition_matrix"})
    if worker_modes:
        quoted_mode = shlex.quote(str(mode))
        quoted_psmc_s = shlex.quote(str(psmc_s))
        quoted_psmc_patterns = shlex.quote(str(psmc_patterns))
        for i in range(1,int(niter)+1):
            if r.mtype!="panmictic":
                if int(r.Nt)<=20000:
                    sentence="sbatch gyarados_lowmem_work.sh "+ str(i) +" "+str(j_name)+" "+str(p)+" "+quoted_mode+" "+quoted_psmc_s+" "+quoted_psmc_patterns
                elif int(r.Nt)<=120000:
                    sentence="sbatch gyarados_midmem_work.sh "+ str(i) +" "+str(j_name)+" "+str(p)+" "+quoted_mode+" "+quoted_psmc_s+" "+quoted_psmc_patterns
                else:
                    sentence="sbatch gyarados_highmem_work.sh "+ str(i) +" "+str(j_name)+" "+str(p)+" "+quoted_mode+" "+quoted_psmc_s+" "+quoted_psmc_patterns
            else: 
                sentence="sbatch gyarados_midmem_work.sh "+ str(i) +" "+str(j_name)+" "+str(p)+" "+quoted_mode+" "+quoted_psmc_s+" "+quoted_psmc_patterns
            os.system(sentence)
            print(sentence)
    else:
        print("No repetition jobs submitted because mode is", mode)
    
     
    return r


def GYARADOS_WORK(i, j_name, p, mode=DEFAULT_GYARADOS_MODE, psmc_s=DEFAULT_PSMC_S, psmc_patterns=None, lightmem=False, full_VCF=False, tmrca=True, only_transition_matrix=False):
    '''
    Worker for n_islands. Runs all analyses for one repetition
    - i: repetition id
    - j_name: json with model data stored
    - p: population we are interested in
    '''
    ########os.system(bash work.sh i json p)
    start_time = time.time()
    run_modes = gy.parse_run_modes(mode)
    psmc_patterns = gy.parse_psmc_patterns(psmc_patterns)
    print("Worker modes:", ",".join(sorted(run_modes)) if run_modes else "none")
    print("Worker PSMC -p vectors:", ",".join(psmc_patterns))
    print("Worker PSMC -s:", psmc_s)
    

    d=gy.get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")
    print(type_model, type(type_model))

    if type_model=="StSi":
       r=gy.StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=gy.panmictic(d=d)
    elif type_model=="SST":
        r=gy.SST(d=d) # parse the model
    elif type_model=="Free":
        r=gy.Free(d=d) # parse the model
    else:
        print("Model incorrect")
        sys.exit()
    
    if type_model=="SST":
        sampling_demes=r.sampled_demes
    elif type_model=="Free":
        raw_sampling_demes=list(range(1,r.islands+1))
        sampling_demes = [deme for deme, val in zip(raw_sampling_demes, r.samples) if val != 0]
    else:
        sampling_demes=[p]
    
    for deme in sampling_demes:
                
        it=gy.Repetition_model(r,i,deme) # parse the repetition
        print("Starting repetition", it.name)
        print("WE start with deme", deme)
        # 1. Copy par file adding the ID of the repetition
        needs_sequence_simulation = bool(run_modes.intersection({"simulate", "transition_matrix"}))
        if needs_sequence_simulation and deme==sampling_demes[0]:
            ######### SKIP THE FSC2 part
            fsc_part=True
            if fsc_part:#
                ## DO FSC RUN

                shutil.copyfile(r.par,it.par)

                # 2. Run fsc2
                print("Running fsc2...",i)
                if tmrca:
                    gy.FSC2_run_with_mrca(it.par)
                else:
                    gy.FSC2_run(it.par)
                print("fsc2 run finished")
                
                basename=(it.par.split("/")[0]).split(".par")[0]
                ## move inside model folder
                print("Moving folder inside model folder")
                sentence="mv -f "+it.name+" "+r.name 
                os.system(sentence)
                print("Compressing Mrca")
                mrca_file=it.par.split(".par")[0]+"/"+basename+"_rep1_mrca.txt"
                sentence="gzip "+ mrca_file
                os.system(sentence) 

                # Compress gen table
                sentence="gzip "+it.old_gen_path
                os.system(sentence)

                # Move to results
                sentence="cp "+it.gen_path+" RESULTS/"+r.name+"/"
                print(sentence)
                os.system(sentence)
        else:
            print("Skipping FSC2 simulation for this deme or mode.")
        print("WE CONTINUE with deme", deme, "and")

        # 3. Get FSC2 stats from SFS

        print("Getting stats...",deme)
        transition_matrix_only = only_transition_matrix or (
            "transition_matrix" in run_modes and not run_modes.intersection({"stats", "psmc", "smcpp"})
        )
        if transition_matrix_only:
            print("Transition matrix mode requested; sequence simulation output was generated/copied, stopping before stats/inference for this deme.")
            continue
        else:
            print("only_transition_matrix is", only_transition_matrix)
        if "stats" not in run_modes:
            print("Skipping sequence summary statistics because mode is", mode)
            gen = None
        else:
            print("READING GEN FILE")
            gen=gy.FSC2_read_gen(it.gen_path)
            gen_save=True
            if gen_save:
                gen.to_csv(it.fullname+"_filtered_gen.gz", compression="gzip") #####maybe change this format 
                sentence="cp "+it.fullname+"_filtered_gen.gz"+" RESULTS/"+r.name+"/"
                print(sentence)
                os.system(sentence)

            evaluate_fsc2=False
            if evaluate_fsc2:
                for individual in range(1,int(r.samples[deme-1])+1):
                    print("Evaluating fsc2.. ")
                    gen1i=gy.FSC2_gen_filter_1i(gen,deme,individual)
                    shapes=[]
                    for chr in range(1,r.chr+1):
                        this_chr=gen1i[gen1i["Chrom"]==chr]
                        shapes.append(this_chr.shape[0])
                        
                    if 0 in shapes: # this means that any of the chromosomes has no polymorphic positions
                        print("Failure in FSC2")
                        return
                        sentence="rm -rf "+it.dir
                        os.system(sentence)
                        gy.GYARADOS_WORK(i,j_name,deme,mode=mode,psmc_s=psmc_s,psmc_patterns=",".join(psmc_patterns)) # call again
                    else:
                                
                        # Continue
                        print("Success in FSC2")

                        finish_time=time.time()
                    
                        runtime=round(finish_time-start_time,2)
                    
                        with open("gyarados_time_check.log","a") as f:
                            print(r.name,"FSC2_FILTER",str(runtime),"NA", sep="\t", file=f)

            else: 
                print("Do not evaluate this!")

            print("We are continuing with deme", deme)

            print("START ANALYSES!!!!!!!!!!!!!!!!")

            #usfs2d=gy.FSC2_2DUSFS(gen,r,it)
            print("deme we are working with is:", deme)

            tons,counts=gy.FSC2_fsfs(r,gen,deme) 
            sfs=counts
            print("SFS count for", it.name, sfs)
            if len(sfs)!=0:
                sst=gy.SFS_stats(sfs,tons,counts,it.fullname,r.sizes,r,deme) 

            if r.mtype == "panmictic": # if itsbig only one population remains in it.gen_path
                print("La reina de las cigalas")
            elif r.mtype == "SST":
                sampled_demes=sampling_demes
                if deme==sampling_demes[0]:
                    gy.FSC2_FST(gen,r,it, sampled_demes) # calculate fst
                    #continue # doing this because now I want only one sampling (the central one)
            elif r.mtype == "Free":  # Do Fst for each combination of demes 
                if deme==sampling_demes[0]:
                    # do combinatorial 
                    # and call fst for each combination
                    
                    print("Doing comnbinations for each fst")
                    comb_sampled=set(itertools.combinations(sampling_demes, 2))
                    for sampled_demes in comb_sampled:
                        gy.FSC2_FST(gen,r,it, sampled_demes) 

                else:
                    print("El rey de la gamba")
            else:
                gy.FSC2_FST(gen,r,it) # calculate fst   

            if full_VCF:
                gy.FSC2_gen2fullVCF(gen,it)

        if "smcpp" in run_modes:
            print("Running SMC++ with deme", deme)
            print("Running SMC++ PLOT...",i)
            gy.SMC_call(i,j_name,deme) # parallel call

        if "psmc" in run_modes:
            print("Running PSMC with deme", deme)
            print("Running PSMC...",i)
            for psmc_pattern in psmc_patterns:
                gy.PSMC_SNIF_call(i,j_name,deme,s=psmc_s,psmc_pattern=psmc_pattern) # parallel call

def GYARADOS_IICR(type, test=False,N=None,sizes=None,sample=None,nislands=None,mig=None,p=1,mu=1e-8,rho=1e-8, evs=[], gr=0, chr=5, par=None,M=None):
    
    """
    Runs all the eq_sim_n_islands model  in parallel. 
    Calls for slurm job
    """
    print("Welcome to gyarados! Version 16.11.2023")
    os.system("mkdir -p RESULTS") # create results directory if it doesn't exist yet

    # TIME LOG FILE
    if (os.path.exists("gyarados_time_check.log") == False): # create log file for timinimg each task if doesnt exist
        f = open("gyarados_time_check.log", "w")
        line="\t".join(["MODEL_NAME", "TASK_TYPE", "TIME(S)", "BIN_SIZE/START_POINT"]) + "\n"
        f.write(line)
        f.close()

    # MODEL LOG FILE

    if (os.path.exists("gyarados_models.log") == False): # create log file for timinimg each task if doesnt exist
        x = open("gyarados_models.log", "w")
        line="\t".join(["MODEL_NAME","MODEL_ID","N", "m", "d", "M","reps"])+"\n"
        x.write(line)
        x.close()
    
    if par:
        type=int(3)
        if not p:
            return "Please, specify a deme to study"
            
        r=gy.Free(par)
    
    else:

        size = sizes.split(',')
        chr=int(chr)
        values, counts = np.unique(size, return_counts=True)
        if len(values)==1 and chr!=1:
            sz=size*chr
            sz=list(map(int,sz))
        else:
            sz=list(map(int,size))
        #niter=int(niter)
        print(evs)
        if evs!=str(0):
            evs_space=evs.replace(","," ")
            events=evs_space.split("/") # events are parsed as a string of comma separated numbers
        else:
            events=[]
        if int(type)==1:
            print("StSi model")
            if not nislands:
                return "no nislands specified"
            if not mig:
                return "no mig specified"
            r=gy.StSi(simn_n_islands=int(nislands),simn_N=int(N),simn_samples=int(sample),simn_mig=float(mig),simn_sizes=sz, simn_chr=chr, simn_mu=float(mu), simn_events=events, simn_gr=gr,simn_M=M) # model object
        if int(type)==2:
            print("panmictic model")
            r=gy.panmictic(pan_N=int(N), pan_samples=int(sample), pan_sizes=sz, pan_chr=chr, pan_mu=float(mu),pan_rho=rho, pan_events=events, pan_gr=gr)
            p=1

    j_name=r.json
    print("json file for this model is allocated in", r.json)

    # Create RESULTS/model
    sentence="mkdir -p RESULTS/"+r.name

    print(sentence)
    os.system(sentence)
    # Copy json file
    sentence="cp "+j_name+" RESULTS/"+r.name+"/"
    print(sentence)
    os.system(sentence)

    sentence="cp "+r.par+" RESULTS/"+r.name+"/"
    print(sentence)
    os.system(sentence)

    ### Calculate IICR
    print("Calculating IICR...")
    if test:
        fn_list=[10000, 100000, 1000000, 10000000, 100000000, 1000000000]
        fs_list=[100, 1000, 10000, 100000]
        print("Proceed testing IICR...")
        for fn in fn_list:
            for fs in fs_list:
                gy.IICR_fullrun(r,int(p),fn,fs)
    else:
        
        gy.IICR_fullrun(r,int(p))
        sentence="rm -r "+r.name+"/*fn*fs*"
        os.system(sentence)
    
     ######################### AQUI
    return r



### SUMMARY: get distributions of PSMC, FSC2 stats, and SNIF


########################################################################################################################
#################################                FSC2 FUNCTION                             #############################
########################################################################################################################


def FSC2_run(par):
    '''
    Runs fsc2 (genobioinfo  cluster). Parse the rep name
    '''
    sentence="fsc -i "+par+" -n1 -I -G -g -s0 -x -k 100000000 -q -d"
    os.system(sentence)

def FSC2_run_with_mrca(par, compress=True):
    '''
    Runs fsc2 (genobioinfo cluster). Parse the rep name
    '''
    sentence="fsc -i "+par+" -n1 -I -G -g -s0 -x -k 100000000 -q -d --recordMRCA"
    os.system(sentence)

def FSC2_is_gen_big(gen_path, max_size=50):
    '''
    Check if fsc2 gen table is too big in case of lightmemory=True
    Default max size is 50 GB
    '''
    size_gen=((os.stat(gen_path).st_size)/(1024 * 1024))/1000
    return size_gen > max_size

def FSC2_awk_filter(gen_path, pop):
    '''
    Get a gen table with only the population to study and use this one 
    in order to save ram memory
    Consider : Fst calculation wont be performed
    '''
    with open(gen_path, "r") as gp:
        colums_names=(gp.readline()[:-1]).split("\t")

    cols=[1,2,3]+[colums_names.index(c) for c in colums_names if c.startswith("G_"+str(pop)+"_")]
    in_dols=", ".join(["$"+str(i) for i in cols ]) # inside of awk print function
    full_genpath=gen_path+".full"
    sentence="mv "+gen_path+" "+full_genpath  # change name of original table
    os.system(sentence)
    sentence="awk '{print "+in_dols+"}'"+full_genpath > gen_path # get the new one
    os.system(sentence)
    return(full_genpath)




def FSC2_filter_segregating(gen):
    '''
    Eliminate those positions where more than 2 alleles are segregating
    '''
    print(gen.shape[0]," initial positions")
    # Mark all the duplicated positions as False so they can be removed
    duplicated_coords=gen.duplicated(subset=["Chrom","Pos"], keep=False)
    if any(duplicated_coords):
        gen_filtered = gen.loc[duplicated_coords == False,:]
        print(gen_filtered.shape[0]," final positions")
        print(gen.shape[0]-gen_filtered.shape[0], "Eliminated positions")
    else:
        gen_filtered = gen
    return(gen_filtered)


def FSC2_filter_segregating2(gen):
    '''
    Eliminate those positions where more than 2 alleles are segregating
    '''
    print(gen.shape[0]," initial positions")
    counter=0
    for c in np.unique(gen.Chrom):
        this_chr=gen[gen.Chrom==c]
        values, counts = np.unique(this_chr.Pos, return_counts=True)
        ii = np.where(counts == 2)[0]
        for i in ii: 
            rm=values[i]
            counter+=1
            #print(rm)
        gen=gen[gen.Pos!=rm]
    print(gen.shape[0]," final positions")
    print(counter, "Eliminated positions")
    return(gen)

def FSC2_read_gen(gen_path, gzip=True, filter=True):
    '''
    Reads the .gen table from fsc2 and returns a pd DataFrame
    If filter, all the positions with more than one segregating allele will be eliminated
    Default: filter=True
    '''
    if gzip:

        gen = pd.read_csv(gen_path, sep='\t',index_col=False, compression='gzip')
    else:
        gen = pd.read_csv(gen_path, sep='\t',index_col=False)
    
    if filter:
        gen=gy.FSC2_filter_segregating(gen)


    return gen




def FSC2_gen_filter_1i(g, p, individual=1):

    '''
    Get only one individual from an specific population
    and preserve only heterozygous positions
    Used to perform PSMC when population substructure
    '''
    print("entering in FSC2_gen_filter_1i for population and individual ", p, individual)

    ind_col="G_"+str(p)+"_"+str(individual)
    print ("Filtering for individual", ind_col)
    #filter_col = [col for col in g if (col.startswith("G_"+str(p)+"_")) or (col.startswith("Pos")) or (col.startswith("Chrom"))]
    a=g[["Chrom","Pos",ind_col]]
    a=a.iloc[:,0:3]  # only first individual
    a=a[a.iloc[:,2]==1]  # avoid homozygous in this individual. only want heterozygote (1)
    print(list(a.columns)[2], " individual has ", a.shape[0], " heterozygote positions" )
    return a


def FSC2_gen_filter_1p(g,p):
    '''
    one population at a time (p)
    Reads pd dataframe of FSC2  .gen table and
    Returns dataframe with only one population counts.
    '''
    print("Filtering FSC2 one population", p)
    filter_col = [col for col in g if col.startswith("G_"+str(p)+"_")]
    print(filter_col)
    a=g[filter_col]
    return a


def FSC2_2DUSFS(gen, r, it, full_name=True, dir=False):
    '''
    2D unfolded SFS from the sampled demes 
    '''
    print("Starting 2dUSFS")
    all_snp_counts=[]
    if not r.sampled_demes:
        r.sampled_demes=[1,2]
    for deme in r.sampled_demes:
        print(deme)
        gen1p=gy.FSC2_gen_filter_1p(gen,deme)
        snp_counts=gen1p.sum(axis=1) # get counts for each snp
        all_snp_counts.append(snp_counts) 
    
    all_snp_counts = list(zip(*all_snp_counts))
    counts_snp_df = pd.DataFrame.from_records(all_snp_counts, columns=[str(i+1) for i in range(len(all_snp_counts[0]))])
    if r.samples[r.sampled_demes[0]-1]!=r.samples[r.sampled_demes[1]-1]:
        print("Samples are")
        print(r.samples[r.sampled_demes[0]-1])
        print(r.samples[r.sampled_demes[1]-1])
        print("SFS can't be computed is sample size is not the same!")
        return("Error")
    size_sfs=r.samples[r.sampled_demes[0]-1]
    print("Size of unfolded sfs is %d" %size_sfs)
    sfs_matrix=np.zeros(shape=(size_sfs+1,size_sfs+1))
    # Get each combination of counts
    for i in range(0, size_sfs+1):
        for j in range(0, size_sfs+1):         
            # Define the condition for filtering
            condition = (counts_snp_df.iloc[:, 0] == i) & (counts_snp_df.iloc[:, 1] == j)
            # Count the number of rows that match the condition
            if condition.any():
                count=counts_snp_df[condition].shape[0]
            else:
                count=0
            sfs_matrix[i-1,j-1]=count
            print(i,j,count)
    if full_name:
        sfs_matrix.tofile(it.fullname+"_2DUSFS.tsv", sep="\t")
        freq_sfs_matrix=sfs_matrix/gen.shape[0]
        freq_sfs_matrix.tofile(it.fullname+"_freq_2DUSFS.tsv", sep="\t")
        sentence="cp "+it.fullname+"_2DUSFS.tsv  RESULTS/"+r.name+"/"
        os.system(sentence)
        sentence="cp "+it.fullname+"_freq_2DUSFS.tsv  RESULTS/"+r.name+"/"
        os.system(sentence)
        return(it.fullname+"_2DUSFS.tsv")

    else:
        sfs_matrix.tofile(dir+"/"+r.name+"_2DUSFS.tsv", sep="\t")
        freq_sfs_matrix=sfs_matrix/gen.shape[0]
        freq_sfs_matrix.tofile(dir+"/"+r.name+"_freq_2DUSFS.tsv", sep="\t")
        return(dir+"/"+r.name+"_2DUSFS.tsv")
            

def FSC2_fsfs(r,gen,p, filter_1pop=True):
    '''
    Gets the Magikarp Model type object pd dataframe of the gen table and the population index (str)
    Returns the plain folded sfs count (from singletons) without normalizing.
    '''
    if filter_1pop:
        gen_1p=gy.FSC2_gen_filter_1p(gen,int(p))
    
    if r.mtype!="panmictic":
        snp_counts=np.where(gen_1p.sum(axis=1)>r.samples[int(p)-1]//2,r.samples[int(p)-1]-gen_1p.sum(axis=1),gen_1p.sum(axis=1))
        tons, counts = np.unique(snp_counts, return_counts=True)
        tons=tons[1:]
        counts=counts[1:]
    else:
        sample=int(r.samples[0]) ########################### MEH
        max_count=sample//2
        print("Sample is",sample, type(sample),". Max count for fSFS is ", max_count)
        snp_counts=np.where(gen_1p.sum(axis=1)>max_count,sample-gen_1p.sum(axis=1),gen_1p.sum(axis=1))
        tons, counts = np.unique(snp_counts, return_counts=True)
        
    return tons,counts

def FSC2_direct_fsfs(r_samples,n_sampled_pops,gen,p, filter_1pop=True):
    '''
    Gets the Magikarp Model type object pd dataframe of the gen table and the population index (str)
    Returns the plain folded sfs count (from singletons) without normalizing.
    '''
    if filter_1pop:
        gen_1p=gy.FSC2_gen_filter_1p(gen,int(p))
    
    if n_sampled_pops!=1:
        snp_counts=np.where(gen_1p.sum(axis=1)>r_samples[int(p)-1]//2,r_samples[int(p)-1]-gen_1p.sum(axis=1),gen_1p.sum(axis=1))
        tons, counts = np.unique(snp_counts, return_counts=True)
        tons=tons[1:]
        counts=counts[1:]
    else:
        
        sample=int(r_samples[0])
        max_count=sample//2
        print("Sample is",sample, type(sample),". Max count for fSFS is ", max_count)
        snp_counts=np.where(gen_1p.sum(axis=1)>max_count,sample-gen_1p.sum(axis=1),gen_1p.sum(axis=1))
        tons, counts = np.unique(snp_counts, return_counts=True)
    return tons,counts

def FSC2_gen2VCF(gen,it,p=0):
    """
    Generates a vcf-like file from the fsc2 .gen table
    - gen: pd dataframe of gen table
    - p: population. If non specified or 0, all the samples are retrieved
    Returns: A tuple
    - [0]: path_to_vcf_file
    - [1]: list of individual samples
    """
    thisP=gy.FSC2_gen_filter_1p(gen,p) # get only genotypes for 1 pop
    inds=thisP.columns # save ind column names for header
    # Transform genotypes
    thisP_mat=thisP.to_numpy(dtype="str") # transform to numpy str array to substitute
    thisP_mat[thisP_mat=="0"]="0/0"
    thisP_mat[thisP_mat=="1"]="0/1"
    thisP_mat[thisP_mat=="2"]="1/1"
    # Create pd. VCF dataframe
    in_vcf=pd.DataFrame()
    in_vcf["#CHROM"]=gen.Chrom
    in_vcf["POS"]=gen.Pos
    in_vcf["ID"]="."
    in_vcf["REF"]=gen.Anc_all
    in_vcf["ALT"]=gen.Der_all
    in_vcf["QUAL"]="."
    in_vcf["FILTER"]="PAS"
    in_vcf["INFO"]="DP=100"
    in_vcf["FORMAT"]="GT"
    ind_df=pd.DataFrame(thisP_mat,columns=inds)
    in_vcf.reset_index(drop=True, inplace=True) # reset indexes to concat
    ind_df.reset_index(drop=True,inplace=True)
    in_vcf=pd.concat([in_vcf,ind_df],axis=1)
    tsv_file=it.fullname+"_deme_"+str(p)+".tempvcf"
    vcf_file=it.fullname+"_deme_"+str(p)+".vcf"
    in_vcf.to_csv(tsv_file,index=False, sep="\t") # write tsv 
    sentence="cat header.txt "+tsv_file+" > "+vcf_file 
    os.system(sentence) # call for header paste and create vcf
    return(vcf_file,inds)



def FSC2_FST(gen, r, it, sampled_demes=[1,2]):
    #if r.mtype=="SST" or r.mtype=="Free":
    #    sampled_demes=r.sampled_demes
    #elif r.mtype=="StSi":
    #    sampled_demes=[1,2]
    #else:
    #    print("No fst to be calculated here!")
    #    return
    print("Sampled demes are:", sampled_demes)
    frequencies=pd.DataFrame()
    deme_sizes=[r.samples[sampled_demes[0]-1], r.samples[sampled_demes[1]-1]]
    Nt=sum([r.samples[d-1]/2 for d in sampled_demes])
    print("Nt is", Nt)
    print("total sample size is: ", Nt)
    for deme in sampled_demes:
        print("Samples in this deme are:",r.samples[deme-1] /2)
        this_deme_gen=gy.FSC2_gen_filter_1p(gen,deme)
        #print(this_deme_gen.head())
        # 1. Get allele frequencies per population and overall populations (2 more columns per dataframe)
        p=this_deme_gen.sum(axis=1)/r.samples[deme-1]
        q=1-p
        #frequencies=pd.DataFrame({"p"+str(deme):p,"q"+str(deme):q})
        frequencies["p"+str(deme)]=p
        frequencies["q"+str(deme)]=q
        frequencies["pond_p"+str(deme)]=p*(r.samples[deme-1])
        frequencies["pond_q"+str(deme)]=q*(r.samples[deme-1])
        # 2. Get H-W genotype frequencies per population
        frequencies["HWAA"+str(deme)]=frequencies["p"+str(deme)]**2
        frequencies["HWAa"+str(deme)]=2*frequencies["p"+str(deme)]*frequencies["q"+str(deme)]
        frequencies["HWaa"+str(deme)]=frequencies["q"+str(deme)]**2
        # 3. Get observed heterozygotes frequencies per population (sum rows with 1n and divide by total number of individuals in population)
        frequencies["Hobs"+str(deme)]=this_deme_gen.eq(1).sum(axis=1)/(r.samples[deme-1]/2)
        frequencies["N_Hobs"+str(deme)]=frequencies["Hobs"+str(deme)]*(r.samples[deme-1] /2)
        # 4. Calculate local expected heterozygosity per population (1-(p^2+q^2))
        frequencies["Hexp"+str(deme)]=1-(frequencies["p"+str(deme)]**2+frequencies["q"+str(deme)]**2)   
        frequencies["N_Hexp"+str(deme)]=frequencies["Hexp"+str(deme)]*(r.samples[deme-1]/2)
        # 5. Calculate Fs out of each population: Fs = (Hexp-Hobs)/Hexp
        this_fs=(frequencies["Hexp"+str(deme)]-frequencies["Hobs"+str(deme)])/frequencies["Hexp"+str(deme)]
        print("this fs are", this_fs)
        frequencies["Fs"+str(deme)]=this_fs
    ######## per population finishes here
    # 6. Calculate mean frequencies (metapop frequencies)
    filter_col= [col for col in frequencies if col.startswith("pond_p")]
    ps=frequencies[filter_col]
    print("ps are", ps)
    #### frequencies["mean_p"]=ps.sum(axis=1)/2 ######### ONLY works if all demes have the same size 
    frequencies["prev_mean_p"]=ps.sum(axis=1) ######### ONLY works if all demes have the same size 
    frequencies["mean_p"]=frequencies["prev_mean_p"]/(int(Nt)*2) ######### ONLY works if all demes have the same size 
    filter_col= [col for col in frequencies if col.startswith("pond_q")]
    ps=frequencies[filter_col]
    #print("ps is", ps)
    frequencies["prev_mean_q"]=ps.sum(axis=1) ######### ONLY works if all demes have the same size 
    frequencies["mean_q"]=frequencies["prev_mean_q"]/(int(Nt)*2) ######### ONLY works if all demes have the same size 
    #### frequencies["mean_q"]=ps.sum(axis=1)/2  ######### ONLY works if all demes have the same size 
    
    # 7. Calculate H indexes
    
    print("Total number of INDIVIDUALS in population", Nt)
    # 7.1. HI = sum(Hobsi x Ni)/Nt . Observed heterozygote individuals in all subpopulations. Pondered mean
    filter_col= [col for col in frequencies if col.startswith("N_Hobs")]
    ps=frequencies[filter_col]
    frequencies["Hi"]=ps.sum(axis=1)/(Nt)
    # 7.2. HS = sum(Hexpi x Ni)/Nt . Expected heterozygotes in all subpopulations. Pondered mean
    filter_col= [col for col in frequencies if col.startswith("N_Hexp")]
    ps=frequencies[filter_col]
    frequencies["Hs"]=ps.sum(axis=1)/(Nt)
    # 7.3. HT = 1 - (pmean^2 + qmean^2) . Expected heterozygotes in metapopulation
    frequencies["Ht"]=1-(frequencies["mean_p"]**2+frequencies["mean_q"]**2)
    # 8. Calculate F statistics
    # 8.1. Fis = (Hs-Hi)/Hs
    frequencies["Fis"]=(frequencies["Hs"]-frequencies["Hi"])/frequencies["Hs"]
    # 8.2. Fst = (Ht - Hs)/Ht
    frequencies["Fst"]=(frequencies["Ht"]-frequencies["Hs"])/frequencies["Ht"]
    # 8.3. Fit = (Ht - Hi)/Ht
    frequencies["Fit"]=(frequencies["Ht"]-frequencies["Hi"])/frequencies["Ht"]
    # Save dataframe
    print("Saving frequencies dataframe...")
    #frequencies.to_csv(it.fullname+"_demes_"+str(sampled_demes[0])+"_"+str(sampled_demes[1])+"_full_fst.tsv", index=False,sep="\t") ### HERE
    #sentence="cp "+it.fullname+"_demes_"+str(sampled_demes[0])+"_"+str(sampled_demes[1])+"_full_fst.tsv RESULTS/"+r.name+"/"
    #print(sentence)
    #os.system(sentence)
    # Copy to results
    # Create summary dataframe
    print("Creating summary dataframe...")
    mean_f=pd.DataFrame({
        "Fis":float(frequencies["Fis"].mean()),
        "Fst":float(frequencies["Fst"].mean()),
        "Fit":float(frequencies["Fit"].mean())}, index=[1])
    print(mean_f)
    mean_f.to_csv(it.fullname+"_demes_"+str(sampled_demes[0])+"_"+str(sampled_demes[1])+"_mean_fst.tsv", index=False,sep="\t")
    sentence="cp "+it.fullname+"_demes_"+str(sampled_demes[0])+"_"+str(sampled_demes[1])+"_mean_fst.tsv RESULTS/"+r.name+"/"
    print(sentence)
    os.system(sentence)
    return mean_f


def FSC2_gen2fullVCF(gen, it):
    EMH=True # control to change column names
    filter_col=[col for col in list(gen.columns) if col.startswith("G_")]
    inds=filter_col
    only_gen=gen[filter_col]
    if EMH==True: # Transform the names of individuals
        print("transforming individuals names")
        # Dictionary in populations from simulations
        dict_pops={1:"AFR",
                2:"EUR",
                3:"EA",
                4:"MSEA",
                5:"Phil",
                6:"PNG",
                7:"N",
                8:"D",
                9:"EMH"}
        # Change the name of individuals
        ind_in=[i.split("_")[2] for i in inds]
        ind_pop=[dict_pops[int(i.split("_")[1])] for i in inds]
        inds=[ind_pop[i]+"_"+ind_in[i] for i in range(len(ind_in))]
    gen_mat=only_gen.to_numpy(dtype="str") # transform to numpy str array to substitute
    gen_mat[gen_mat=="0"]="0/0"
    gen_mat[gen_mat=="1"]="0/1"
    gen_mat[gen_mat=="2"]="1/1"
    in_vcf=pd.DataFrame()
    in_vcf["#CHROM"]=gen.Chrom
    in_vcf["POS"]=gen.Pos
    in_vcf["ID"]="."
    in_vcf["REF"]=gen.Anc_all
    in_vcf["ALT"]=gen.Der_all
    in_vcf["QUAL"]="."
    in_vcf["FILTER"]="PASS"
    in_vcf["INFO"]="DP=100"
    in_vcf["FORMAT"]="GT"
    ind_df=pd.DataFrame(gen_mat,columns=inds)
    ind_df.insert(0,"Ancestral", "0/0", True)
    in_vcf.reset_index(drop=True, inplace=True) # reset indexes to concat
    ind_df.reset_index(drop=True,inplace=True)
    in_vcf=pd.concat([in_vcf,ind_df],axis=1)
    tsv_file=it.fullname+"_full.tempvcf"
    vcf_file=it.fullname+"_full.vcf"
    in_vcf.to_csv(tsv_file,index=False, sep="\t") # write tsv 
    sentence="cat header.txt "+tsv_file+" > "+vcf_file 
    os.system(sentence) # call for header paste and create vcf

def FSC2_FST4(gen,r,it):
    '''
    Calculate pairwise FST for all metapopulations
    '''
    print("Calculating FST for all populations")

    # Per each population 

    lp=[]# list of pandas dataframes (gen table per population)
    hs=[]
    if r.mtype=="StSi":
        ndemes=2
        for i in range(ndemes):
            # 1. Get 1 gen table 
            d=i+1 # deme we are at
            this_p=gy.FSC2_gen_filter_1p(gen,d)
            
            # 2. Calculate heterozygosity per locus
            # H= 1 - sum(pi^2) 
            # p = count_rows/r.samples[i]
            this_p["p"]=this_p.sum(axis=1)/r.samples[i]
            this_p["q"]=1-this_p.p
            this_p["H"]=1-(this_p.p**2+this_p.q**2)
            hs.append(this_p.H.mean())
            lp.append(this_p)
        pairwise=list(itertools.product(np.arange(1,ndemes+1),np.arange(1,ndemes+1))) # combinations of r.islands
    elif r.mtype=="SST":
        ndemes=2
        demes=r.sampled_demes
        for i in range(ndemes):
            d=demes[i]
            this_p=gy.FSC2_gen_filter_1p(gen,d)
            print("This gen table from deme", d, "has", this_p.shape[0],"positions")
            
            # 2. Calculate heterozygosity per locus
            # H= 1 - sum(pi^2) 
            # p = count_rows/r.samples[i]
            # Calculate allele frequencies
            this_p["p"]=this_p.sum(axis=1)/(r.samples[d-1])  # p frequency
            this_p["q"]=1-this_p.p # q frequency
            this_p["H"]=1-(this_p.p**2+this_p.q**2) # calculate the heterozygosity in this position
            print("heading for this gen table in deme", d)
            print(this_p.head())
            hs.append(this_p.H.mean())
            lp.append(this_p)
        pairwise=list(itertools.product(np.arange(1,ndemes+1),np.arange(1,ndemes+1))) # combinations of r.islands
        pairwise_demes=list(itertools.product([demes[0],demes[1]],[demes[0],demes[1]])) # combinations of r.islands
    else:
        print("Model not implemented yet")
        return False
    
        
    
    print("pairwise", pairwise)
    fst=np.zeros((ndemes,ndemes)) # matrix
    for i in range(0,len(pairwise)):
        combi=pairwise[i]
        this_deme=pairwise_demes[i]
        d1=this_deme[0]
        d2=this_deme[1]
        this_hs=hs[combi[0]-1]
        print("pairwise comparison", this_deme[0])
        combi0=gy.FSC2_gen_filter_1p(gen,this_deme[0])
        print("pairwise_comparison", this_deme[1])
        combi1=gy.FSC2_gen_filter_1p(gen,this_deme[1])
        combi_gen=pd.concat([combi0,combi1],axis=1) 
            # 2. Calculate heterozygosity per locus
            # H= 1 - sum(pi^2) 
            # p = count_rows/r.samples[i]
        combi_gen["p"]=combi_gen.sum(axis=1)/(r.samples[d1-1]*2)  ### r.samples is always the same (not divided bc its double)
        combi_gen["q"]=1-combi_gen.p
        combi_gen["H"]=1-(combi_gen.p**2+combi_gen.q**2)
        print("heading for this gen table from this set of demes", pairwise_demes)
        print(this_p.head())
        Hm=combi_gen.H.mean()
        this_fst=(Hm-this_hs)/Hm
        fst[combi[0]-1,combi[1]-1]=this_fst
        np.savetxt(it.fullname+"_demes_"+str(demes[0])+"_"+str(demes[1])+"_fst.tsv", fst, delimiter="\t")
        #np.savetxt("my_fst.tsv", fst, delimiter="\t")
        # Copy it to result folder
        sentence="cp "+it.fullname+"_demes_"+str(demes[0])+"_"+str(demes[1])+"_fst.tsv RESULTS/"+r.name+"/"
        print(sentence)
        os.system(sentence)

def FSC2_FST3(gen,r,it):
    '''
    Calculate pairwise FST for all metapopulations
    '''
    print("Calculating FST for all populations")

    # Per each population 

    lp=[]# list of pandas dataframes (gen table per population)
    hs=[]
    if r.mtype=="StSi":
        ndemes=2
        for i in range(ndemes):
            # 1. Get 1 gen table 
            d=i+1 # deme we are at
            this_p=gy.FSC2_gen_filter_1p(gen,d)
            
            # 2. Calculate heterozygosity per locus
            # H= 1 - sum(pi^2) 
            # p = count_rows/r.samples[i]
            this_p["p"]=this_p.sum(axis=1)/r.samples[i]
            this_p["q"]=1-this_p.p
            this_p["H"]=1-(this_p.p**2+this_p.q**2)
            hs.append(this_p.H.mean())
            lp.append(this_p)
        pairwise=list(itertools.product(np.arange(1,ndemes+1),np.arange(1,ndemes+1))) # combinations of r.islands
    elif r.mtype=="SST":
        ndemes=2
        demes=r.sampled_demes
        for i in range(ndemes):
            d=demes[i]
            this_p=gy.FSC2_gen_filter_1p(gen,d)
            print("This gen table from deme", d, "has", this_p.shape[0],"positions")
            
            # 2. Calculate heterozygosity per locus
            # H= 1 - sum(pi^2) 
            # p = count_rows/r.samples[i]
            # Calculate allele frequencies
            this_p["p"]=this_p.sum(axis=1)/r.samples[d-1]  # p frequency
            this_p["q"]=1-this_p.p # q frequency
            this_p["H"]=1-(this_p.p**2+this_p.q**2) # calculate the heterozygosity in this position
            print("heading for this gen table in deme", d)
            print(this_p.head())
            hs.append(this_p.H.mean())
            lp.append(this_p)
        pairwise=list(itertools.product(np.arange(1,ndemes+1),np.arange(1,ndemes+1))) # combinations of r.islands
        pairwise_demes=list(itertools.product([demes[0],demes[1]],[demes[0],demes[1]])) # combinations of r.islands
    else:
        print("Model not implemented yet")
        return False
    
        
    
    print("pairwise", pairwise)
    fst=np.zeros((ndemes,ndemes)) # matrix
    for i in range(0,len(pairwise)):
        combi=pairwise[i]
        this_deme=pairwise_demes[i]
        this_hs=hs[combi[0]-1]
        print("pairwise comparison", this_deme[0])
        combi0=gy.FSC2_gen_filter_1p(gen,this_deme[0])
        print("pairwise_comparison", this_deme[1])
        combi1=gy.FSC2_gen_filter_1p(gen,this_deme[1])
        combi_gen=pd.concat([combi0,combi1],axis=1) 
            # 2. Calculate heterozygosity per locus
            # H= 1 - sum(pi^2) 
            # p = count_rows/r.samples[i]
        combi_gen["p"]=combi_gen.sum(axis=1)/(r.samples[i]*2)  ### r.samples is always the same
        combi_gen["q"]=1-combi_gen.p
        combi_gen["H"]=1-(combi_gen.p**2+combi_gen.q**2)
        Hm=combi_gen.H.mean()
        this_fst=(Hm-this_hs)/Hm
        fst[combi[0]-1,combi[1]-1]=this_fst
        np.savetxt(it.fullname+"_demes_"+str(demes[0])+"_"+str(demes[1])+"_fst.tsv", fst, delimiter="\t")
        #np.savetxt("my_fst.tsv", fst, delimiter="\t")
        # Copy it to result folder
        sentence="cp "+it.fullname+"_demes_"+str(demes[0])+"_"+str(demes[1])+"_fst.tsv RESULTS/"+r.name+"/"
        print(sentence)
        os.system(sentence)

def FSC2_FST2(gen,r,it):
    '''
    Calculate pairwise FST for all metapopulations
    '''

    # Per each population 

    lp=[]# list of pandas dataframes (gen table per population)
    hs=[]
    if r.mtype=="StSi":
        ndemes=2
        
    else:
        ndemes=r.islands
    for i in range(ndemes):
        # 1. Get 1 gen table 
        d=i+1 # deme we are at
        this_p=gy.FSC2_gen_filter_1p(gen,d)
        
        # 2. Calculate heterozygosity per locus
        # H= 1 - sum(pi^2) 
        # p = count_rows/r.samples[i]
        this_p["p"]=this_p.sum(axis=1)/r.samples[i]
        this_p["q"]=1-this_p.p
        this_p["H"]=1-(this_p.p**2+this_p.q**2)
        hs.append(this_p.H.mean())
        lp.append(this_p)
        
    pairwise=list(itertools.product(np.arange(1,ndemes+1),np.arange(1,ndemes+1))) # combinations of r.islands
    fst=np.zeros((ndemes,ndemes)) # matrix
    for combi in pairwise:
        this_hs=hs[combi[0]-1]
        combi0=gy.FSC2_gen_filter_1p(gen,combi[0])
        combi1=gy.FSC2_gen_filter_1p(gen,combi[1])
        combi_gen=pd.concat([combi0,combi1],axis=1)
            # 2. Calculate heterozygosity per locus
            # H= 1 - sum(pi^2) 
            # p = count_rows/r.samples[i]
        combi_gen["p"]=combi_gen.sum(axis=1)/(r.samples[i]*2)
        combi_gen["q"]=1-combi_gen.p
        combi_gen["H"]=1-(combi_gen.p**2+combi_gen.q**2)
        Hm=combi_gen.H.mean()
        this_fst=(Hm-this_hs)/Hm
        fst[combi[0]-1,combi[1]-1]=this_fst
        np.savetxt(it.fullname+"_fst.tsv", fst, delimiter="\t")
        #np.savetxt("my_fst.tsv", fst, delimiter="\t")
        # Copy it to result folder
        sentence="cp "+it.fullname+"_fst.tsv  RESULTS/"+r.name+"/"
        print(sentence)
        os.system(sentence)

def FSC2_FST_direct(gen_path,ndemes,nsamples,name):
    '''
    Calculate pairwise FST for all metapopulations
    Do it directly from a gen table without gyarados/magikarp framework
    - gen_path: path to gen table (gz)
    - number of simulated demes
    - nsamples: list/array with samples simulated per deme
    - name : output name
    '''

    # Per each population 
    print("Reading gen table in %s" % gen_path)
    gen=gy.FSC2_read_gen(gen_path)

    lp=[]# list of pandas dataframes (gen table per population)
    hs=[]

    print("Getting intra deme heterozygosity...")

    for i in range(ndemes):
        
        # 1. Get 1 gen table 
        d=i+1 # deme we are at
        print("Deme %d" % d)

        this_p=gy.FSC2_gen_filter_1p(gen,d)
        
        # 2. Calculate heterozygosity per locus
        # H= 1 - sum(pi^2) 
        # p = count_rows/r.samples[i]
        this_p["p"]=this_p.sum(axis=1)/nsamples[i]
        this_p["q"]=1-this_p.p
        this_p["H"]=1-(this_p.p**2+this_p.q**2)
        hs.append(this_p.H.mean())
        lp.append(this_p)
        
    pairwise=list(itertools.product(np.arange(1,ndemes+1),np.arange(1,ndemes+1))) # combinations of r.islands
    print(pairwise)
    fst=np.zeros((ndemes,ndemes)) # matrix
    print("Getting inter demes heterozygosity")
    for combi in pairwise:
        print(combi)
        this_hs=hs[combi[0]-1]
        combi0=gy.FSC2_gen_filter_1p(gen,combi[0])
        combi1=gy.FSC2_gen_filter_1p(gen,combi[1])
        combi_gen=pd.concat([combi0,combi1],axis=1)
            # 2. Calculate heterozygosity per locus
            # H= 1 - sum(pi^2) 
            # p = count_rows/samples[i]
        samples_deme1=nsamples[combi[0]-1]
        samples_deme2=nsamples[combi[1]-1]
        all_samples=samples_deme1+samples_deme2
        print("samples_deme1 = ",samples_deme1, "\nsamples_deme2= ",samples_deme2, "\ntotal_samples = ",all_samples)
        combi_gen["p"]=combi_gen.sum(axis=1)/(all_samples)
        combi_gen["q"]=1-combi_gen.p
        combi_gen["H"]=1-(combi_gen.p**2+combi_gen.q**2)
        Hm=combi_gen.H.mean()
        this_fst=(Hm-this_hs)/Hm
        fst[combi[0]-1,combi[1]-1]=this_fst
    np.savetxt(name+"_fst.tsv", fst, delimiter="\t")
    print("Now you know it, bc you simulated it. \nOutput file: "+name+"_fst.tsv")
        #np.savetxt("my_fst.tsv", fst, delimiter="\t")
        # Copy it to result folder
        #sentence="cp "+it.fullname+"_fst.tsv  RESULTS/"+r.name+"/"
        #print(sentence)
        #os.system(sentence)

##############################################################################################################################
#################################               STAIRWAY PLOT 2 FUNCTIONS                             ########################
##############################################################################################################################

def STW_call(i,j_name,p):
    '''
    Call for stw plot inferention in another job
    '''
    sentence="sbatch call_stw_gy.sh "+ str(i) +" "+str(j_name)+" "+str(p)
    #sentence="bash call_stw_gy.sh "+ str(i) +" "+str(j_name)+" "+str(p)
    os.system(sentence)

def STW_fullrun(i,j_name,p):
    '''
    Runs all the STW plot inferention considering another job
    '''
    start_time = time.time()
    #os.system(bash work.sh i json p)
    d=gy.get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")
    print(type_model, type(type_model))

    if type_model=="StSi":
       r=gy.StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=gy.panmictic(d=d)
    elif type_model=="SST":
        r=gy.SST(d=d) # parse the model
    elif type_model=="Free":
        r=gy.Free(d=d) # parse the mode
    else:
        print("Model incorrect")
        sys.exit()

    print(d["mtype"])
    print("The model is", d.get("mtype"))

    it=gy.Repetition_model(r,i,p) # parse the repetition

    gen=gy.FSC2_read_gen(it.gen_path) # read the gen table

    tons,counts=gy.FSC2_fsfs(r,gen,p)

    sfs=counts
    print(sfs)
    print(*tons,sep=" ")

    gy.STW_write_blueprint(r,it,p,sfs)
    final_summary_stw=gy.STW_run(it)
    
    # Copy to results
    sentence="cp "+final_summary_stw+" RESULTS/"+r.name+"/"
    print(sentence)
    os.system(sentence)

    finish_time=time.time()
    runtime=round(finish_time-start_time, 2)
    with open("gyarados_time_check.log","a") as f:
        print(r.name,"STW",str(runtime),"NA", sep="\t", file=f)
    
    # Sample final_summary_stw
    gy.STW_sample(final_summary_stw,r,it)

    #sentence="rm "+it.dir+"/*blueprint*"
    #os.system(sentence)
    #sentence="rm -rf "+it.dir+"/stairwayplot"
    #os.system(sentence)

    del gen

def STW_write_blueprint(r,it,p,sfs):
    
    '''
    Inputs:
    - r : MK model object
    - it : MK repetition object
    - p: population (int)
    - sfs: folded sfs count (int)
    '''
    stairway_dir = os.path.abspath("stairway_plot_es")
    nseq=int(r.samples[int(p)-1])
    print("Nseq:",nseq, type(nseq))
    print(len(sfs), "entries ",nseq, "samples")
    with open(it.blueprint, "w") as b:
        b.write("popid: "+it.name+"\n")
        b.write("nseq: "+str(nseq)+"\n")
        L=sum([int(s) for s in r.sizes])
        b.write("L: "+str(L)+"\n")
        b.write("whether_folded: true\n")
        b.write("SFS:")
        for c in sfs:
            b.write(" "+str(c))
        b.write("\npct_training: 0.67\n")
        b.write("nrand: "+str(round((nseq-2)/4))+" "+str(round((nseq-2)/2))+" "+str(round((nseq-2)*3/4))+" "+str(round(nseq-2))+"\n")
        b.write("project_dir: "+it.dir+"/stairwayplot_"+str(p)+"\n")
        b.write("ninput: 200\n")
        b.write("stairway_plot_dir: "+stairway_dir+"\n")
        b.write("mu: "+str(r.mu)+"\n") 
        b.write("years_per_generation: 1"+"\n")
        if r.mtype != "panmictic":
            b.write("plot_title: "+it.name+"_pop_"+str(p)+"\n")
        else:
            b.write("plot_title: "+it.name+"\n")
        b.write("xrange: 0,0\nyrange: 0,0\nxspacing: 2\nyspacing: 2\nfontsize: 12")


def STW_run(it):
    '''
    Calls stairwayplot software
    '''
    sentence="java -cp stairway_plot_es Stairbuilder "+it.blueprint
    os.system(sentence)
    sentence="bash "+it.blueprint+".sh"
    final_summary_stw=it.dir+"/stairwayplot_"+str(it.pop)+"/"+it.name+"_pop_"+str(it.pop)+".final.summary"
    os.system(sentence)
    return(final_summary_stw)


def STW_sample(final_summary_stw, r, it, save_tmp=False, scheme="log"):
    print("Sampling StairwayPlot2...")
    
    sample_dicts={}
    params=["#"+par for par in r.params]

    if scheme == "both":
        print("Do both sampling schemes")
        scheme = ["linear", "log"]
    else:
        scheme = [scheme]
    print("Length of sampling scheme list is: " + str(len(scheme)))
    print(*scheme)
    for sc in scheme:
        
        if sc == "linear":
            print(sc,": linear sampling")
            xvector=lin_vector
            print("The vector length is: " + str(len(xvector)))
            tyop="gyarados_"+r.mtype+"_stw.lin.out"
            # prepare results
            if r.mtype == "SST" or r.mtype == "Free":
                iicr_df=pd.read_csv(r.iicr_lin_path[it.pop], sep="\t") 
            else:
                iicr_df=pd.read_csv(r.iicr_lin_path, sep="\t")
            if (os.path.exists("RESULTS/"+tyop) == False):
                o = open("RESULTS/"+tyop, "w")
                params=["#"+par for par in r.params]
                header=["#ID"]+params+["#Repetition","#Time","#IICR", "#STW_Inference"]
                line="\t".join(header)+"\n"
                print(line)
                o.write(line)
                o.close()
            tmp_all="gyarados_"+r.name+"_stw_rep_"+str(it.rep)+"_deme_"+str(it.pop)+".lin.out"
        elif sc == "log":
            print(sc,": log sampling")
            xvector=log_vector
            print("The vector length is: " + str(len(xvector)))
            if r.mtype == "SST" or r.mtype == "Free":
                iicr_df=pd.read_csv(r.iicr_log_path[it.pop], sep="\t") 
            else:
                iicr_df=pd.read_csv(r.iicr_log_path, sep="\t")
            tyop="gyarados_"+r.mtype+"_stw.log.out"
            if (os.path.exists("RESULTS/"+tyop) == False):
                o = open("RESULTS/"+tyop, "w")
                params=["#"+par for par in r.params]
                header=["#ID"]+params+["#Repetition","#Time","#IICR", "#STW_Inference"]
                line="\t".join(header)+"\n"
                print(line)
                o.write(line)
                o.close()
            tmp_all="gyarados_"+r.name+"_stw_rep_"+str(it.rep)+"_deme_"+str(it.pop)+".log.out"
            if r.mtype == "SST" or r.mtype == "Free":
                iicr_df=pd.read_csv(r.iicr_log_path[it.pop], sep="\t") 
            else:
                iicr_df=pd.read_csv(r.iicr_log_path, sep="\t")
        else:
            print("Error, sc is", sc)
            return "Incorrect sampling scheme"
        stw_raw=pd.read_csv(final_summary_stw, sep="\t")
        stw_in=pd.DataFrame({"generations":list(stw_raw["year"]), "N":list(stw_raw["Ne_median"])})
        ## Create the xvector df empty
        nans=np.empty(len(xvector))
        nans[:]=np.nan
        l_df=pd.DataFrame({"generations":xvector, "N":nans})
        ## rbind with the original dataframe
        stw_in=pd.concat([stw_in,l_df])
        stw_in=stw_in.sort_values(by=["generations"])
        stw_in=stw_in.fillna(method="ffill")
        stw_sampled=stw_in[stw_in.generations.isin(xvector)]
        stw_sampled.drop_duplicates(subset=["generations"]) # Remove duplicates JIC
        this_stw_sampling=list(stw_sampled["N"])
        print("Adding to the", type(sample_dicts),"a list of length", len(this_stw_sampling))
        #sample_dicts[sc]=this_stw_sampling
        #print("Type in sampling:", type(sample_dicts.get(sc)))
        
        if save_tmp:
            with open("RESULTS/"+tmp_all, "w") as tmp:
                for ind in this_stw_sampling:
                    line=" ".join(str(element) for element in ind)+"\n"
                    tmp.write(line)
        # prepare results
        
        str_iicr=" ".join(str(element) for element in list(iicr_df["scaled_lmd"]))
        str_time=" ".join(str(element) for element in xvector)
        str_stw=" ".join(str(element) for element in this_stw_sampling)


       # write results
        if r.mtype=="StSi":
            print("printing list to fill")
            listofill=[r.id, str(r.N[0]),str(r.islands),str(r.mig[0]), str(r.M),str(it.rep),str_time,str_iicr,str_stw]
            print(*listofill)
        elif r.mtype == "SST":
            print("printing list to fill")
            listofill=[r.id, str(r.N[0]),str(r.mig[0]), str(r.M),str(r.L),str(it.pop),str(it.rep),str_time,str_iicr,str_stw]
            print(*listofill) ###############################################################################################################
        elif r.mtype == "Free":
            print("printing list to fill")
            listofill=[r.id,  str(r.islands), str(r.n_mm), str(r.n_events), str(r.gr[0]), r.name,str(it.pop),str(it.rep),str_time,str_iicr,str_stw]
        else:
            print("Model "+r.mtype+" still not implemented")

        with open("RESULTS/"+tyop, "a") as f:
            print("Write new line")
            line="\t".join(listofill)+"\n"
            f.write(line)
        
    return tyop
        


##############################################################################################################################################################
#########################################################                PSMC FUNCTIONS                     ##################################################
##############################################################################################################################################################

def PSMC_SNIF_call(i,j_name,p,s=DEFAULT_PSMC_S,psmc_pattern=DEFAULT_PSMC_PATTERN):
    '''
    Call SNIF and PSMC inferention
    '''
    sentence="sbatch call_psmc_snif_gy.sh "+ str(i) +" "+str(j_name)+" "+str(p)+" "+str(s)+" "+shlex.quote(str(psmc_pattern))
    #sentence="bash call_psmc_snif_gy.sh "+ str(i) +" "+str(j_name)+" "+str(p)+" "+str(s)
    os.system(sentence)


def PSMC_estimate_sample_per_individual(individual,svalue, it, r, gen, p, psmc_pattern=DEFAULT_PSMC_PATTERN):

    print("Running gy.PSMC_consensus_fa")
    consensus=gy.PSMC_consensus_fa(it,r,gen,p, individual, psmc_pattern=psmc_pattern) # TELL INDIVIDUAL ####### THIS
    consensus_fq=gy.PSMC_convert_fastq(consensus,it, individual,p, psmc_pattern=psmc_pattern)  ############### THIS
    ### UP TO THIS POINT
    print("svalue: %s" % svalue)
    # run for each bin size. If two_bin_sizes==False, only run once as ussually
    # as snif results folder is removed in each round snif results are not mixed up
    # between different bin size
    svalue=str(svalue)
    print("Run psmcfa")
   
    psmcfa=gy.PSMC_convert_input(consensus_fq,it,individual,p,svalue, psmc_pattern=psmc_pattern)   ########### THIS
    psmc=gy.PSMC_run(psmcfa,it,r,individual,p,svalue,psmc_pattern=psmc_pattern)   ############ THIS
    # Remove intermediate files
    #sentence="rm "+consensus
    #os.system(sentence)
    #sentence="rm "+consensus_fq
    #os.system(sentence)
    #sentence="rm "+psmcfa
    #os.system(sentence)
    #psmc=it.fullname+"_s"+svalue+".psmc"
    lpsmc=gy.PSMC_last_iteration(it,r,individual,p,svalue,psmc_pattern=psmc_pattern) # get last iteration ########## THIS
    #lpsmc=it.fullname+"_s100_deme_1_ind_"+str(individual)+".lpsmc"
    print("I got a PSMC for %s bin size in individual" % str(svalue))
    this_sample_dict=gy.PSMC_sample(psmc=lpsmc,svalue=svalue,mu=float(r.mu))   ######## THIS
    return(this_sample_dict)

def PSMC_SNIF_fullrun(j_name,i,p,s=DEFAULT_PSMC_S, psmc_pattern=DEFAULT_PSMC_PATTERN, two_bin_sizes=False,study_bin_size=False):

    '''
    Run all the PSMC and SNIF inferention considering another job
    - j_name: json file defining the model
    - i: repetition
    - p: population 

    Will produce a psmc, lpsmc and all the SNIF inferention outputs. And all the psmc intermediates
    '''

    start_time = time.time()

    d=gy.get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")
    print(type_model, type(type_model))

    if type_model=="StSi":
       r=gy.StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=gy.panmictic(d=d)
    elif type_model=="SST":
        r=gy.SST(d=d) # parse the model
    elif type_model=="Free":
        r=gy.Free(d=d) # parse the mode
    else:
        print("Model incorrect")
        sys.exit()

    print(d["mtype"])
    print("The model is", d.get("mtype"))

    it=gy.Repetition_model(r,i,p) # parse the repetition
    gen=gy.FSC2_read_gen(it.gen_path)   
    
    print("Calling PSMC...",i)
    psmc_patterns = gy.parse_psmc_patterns(psmc_pattern)
    print("PSMC -p vectors:", ",".join(psmc_patterns))
    ## Do each individual
    individuals = int(r.samples[int(p)-1]/2)
    print("Number of individuals sampled %s" % individuals)



     # Control if one bin size or two
    if two_bin_sizes:
        svalues=[100,50]
    elif study_bin_size:
        svalues=[100,70,50,30,20]
        individuals=1
    else: 
        svalues=[s]
    
    print("There are %d bin sizes:" % len(svalues))
    print(*svalues)

    for current_psmc_pattern in psmc_patterns:
        print("Running PSMC vector:", current_psmc_pattern)
        for svalue in svalues:
            results=[]
            #pool = multiprocessing.Pool(processes=individuals) # as many processes as individuals
            for individual in range(1,individuals+1):
                results.append(PSMC_estimate_sample_per_individual(individual,svalue,it, r, gen , p, psmc_pattern=current_psmc_pattern ))

            sample_dicts = results
            print("Now we have %d entries (individuals) in the dict list" % len(sample_dicts))
            # Call mean and write
            gy.PSMC_mean_inferences(sample_dicts=sample_dicts, svalue=svalue,r=r,it=it,p=p,psmc_pattern=current_psmc_pattern) #### THIS
    
    #del pool 
    #del sample_dicts
    #del genc
    # remove intermediate files from PSMC
    # .consensus.fatmp
    # .consensus.fq.gz
    # .consensus.psmcfa
    # sentence="rm "+it.dir+"/*consensus.fatmp"
    # os.system(sentence)
    # sentence="rm "+it.dir+"/*consensus.fq.gz"
    # os.system(sentence)
    # sentence="rm "+it.dir+"/*consensus.psmcfa"
    # os.system(sentence)
    # sentence="rm "+it.dir+"/*psmc"
    # os.system(sentence)
    ########### As I AM DOING Only one rep at a time ######################################################
    # sentence="rm "+r.name+"/*fa"
    # os.system(sentence)




def PSMC_SNIF_fullrun_par(j_name,i,p,s=100, two_bin_sizes=False,study_bin_size=False):

    '''
    Run all the PSMC and SNIF inferention considering another job
    - j_name: json file defining the model
    - i: repetition
    - p: population 

    Will produce a psmc, lpsmc and all the SNIF inferention outputs. And all the psmc intermediates
    '''

    start_time = time.time()

    d=gy.get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")
    print(type_model, type(type_model))

    if type_model=="StSi":
       r=gy.StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=gy.panmictic(d=d)
    else:
        print("Model incorrect")
        sys.exit()

    print(d["mtype"])
    print("The model is", d.get("mtype"))

    it=gy.Repetition_model(r,i,p) # parse the repetition
    gen=gy.FSC2_read_gen(it.gen_path)
    
    print("Calling PSMC...",i)
    ## Do each individual
    individuals = int(r.samples[int(p)-1]/2)
    print("Number of individuals sampled %s" % individuals)



     # Control if one bin size or two
    if two_bin_sizes:
        svalues=[100,50]
    elif study_bin_size:
        svalues=[100,70,50,30,20]
        individuals=1
    else: 
        svalues=[s]
    
    print("There are %d bin sizes:" % len(svalues))
    print(*svalues)

    for svalue in svalues:
        results=[]
        pool = multiprocessing.Pool(processes=individuals) # as many processes as individuals
        for individual in range(1,individuals+1):
            results.append(pool.apply_async(PSMC_estimate_sample_per_individual, args=(individual,svalue,it, r, gen , p )))
        pool.close()
        pool.join()
        sample_dicts = [p.get() for p in results]
        print("Now we have %d entries (individuals) in the dict list" % len(sample_dicts))
        # Call mean and write
        gy.PSMC_mean_inferences(sample_dicts=sample_dicts, svalue=svalue,r=r,it=it)
    
    del pool 
    del sample_dicts
    del gen
    # remove intermediate files from PSMC
    # .consensus.fatmp
    # .consensus.fq.gz
    # .consensus.psmcfa
    sentence="rm "+it.dir+"/*consensus.fatmp"
    os.system(sentence)
    sentence="rm "+it.dir+"/*consensus.fq.gz"
    os.system(sentence)
    sentence="rm "+it.dir+"/*consensus.psmcfa"
    os.system(sentence)
    ##sentence="rm "+it.dir+"/*psmc"
    ###os.system(sentence)
    ########### As I AM DOING Only one rep at a time ######################################################
    sentence="rm "+r.name+"/*fa"
    os.system(sentence)


def PSMC_SNIF_fullrun3(j_name,i,p,s=100, two_bin_sizes=False,study_bin_size=False):

    '''
    Run all the PSMC and SNIF inferention considering another job
    - j_name: json file defining the model
    - i: repetition
    - p: population 

    Will produce a psmc, lpsmc and all the SNIF inferention outputs. And all the psmc intermediates
    '''

    start_time = time.time()

    d=gy.get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")
    print(type_model, type(type_model))

    if type_model=="StSi":
       r=gy.StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=gy.panmictic(d=d)
    else:
        print("Model incorrect")
        sys.exit()

    print(d["mtype"])
    print("The model is", d.get("mtype"))

    it=gy.Repetition_model(r,i,p) # parse the repetition
    gen=gy.FSC2_read_gen(it.gen_path)
    
    print("Calling PSMC...",i)
    ## Do each individual
    individuals = int(r.samples[int(p)-1]/2)
    print("Number of individuals sampled %s" % individuals)



     # Control if one bin size or two
    if two_bin_sizes:
        svalues=[100,50]
    elif study_bin_size:
        svalues=[100,70,50,30,20]
        individuals=1
    else: 
        svalues=[s]
    
    print("There are %d bin sizes:" % len(svalues))
    print(*svalues)

    def PSMC_estimate_sample_per_individual(individual,svalue):

            print("Running gy.PSMC_consensus_fa")
            consensus=gy.PSMC_consensus_fa(it,r,gen,p, individual) # TELL INDIVIDUAL
            consensus_fq=gy.PSMC_convert_fastq(consensus,it, individual) 
            ### UP TO THIS POINT
            print("svalue: %s" % svalue)
            # run for each bin size. If two_bin_sizes==False, only run once as ussually
            # as snif results folder is removed in each round snif results are not mixed up
            # between different bin size
            svalue=str(svalue)
            print("Run psmcfa")
            psmcfa=gy.PSMC_convert_input(consensus_fq,it,individual,svalue)
            psmc=gy.PSMC_run(psmcfa,it,r,individual,svalue)
            #psmc=it.fullname+"_s"+svalue+".psmc"
            lpsmc=gy.PSMC_last_iteration(it,r,individual,svalue) # get last iteration
            print("I got a PSMC for %s bin size in individual" % str(svalue))
            this_sample_dict=gy.PSMC_sample(psmc=lpsmc,svalue=svalue,mu=float(r.mu))
            return(this_sample_dict)

    for svalue in svalues:
        results=[]
        if __name__ == "__main__":
            pool = multiprocessing.Pool(processes=len(individuals)) # as many processes as individuals
            for individual in individuals:
                results.append(pool.apply_async(PSMC_estimate_sample_per_individual, args=(individual,svalue )))
            pool.close()
            pool.join()
            sample_dicts = [p.get() for p in results]
            print("Now we have %d entries (individuals) in the dict list" % len(sample_dicts))
            # Call mean and write
            gy.PSMC_mean_inferences(sample_dicts=sample_dicts, svalue=svalue,r=r,it=it)



def PSMC_SNIF_fullrun2(j_name,i,p,s=100, two_bin_sizes=True,study_bin_size=False):

    '''
    Run all the PSMC and SNIF inferention considering another job
    - j_name: json file defining the model
    - i: repetition
    - p: population 

    Will produce a psmc, lpsmc and all the SNIF inferention outputs. And all the psmc intermediates
    '''

    start_time = time.time()

    d=gy.get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")
    print(type_model, type(type_model))

    if type_model=="StSi":
       r=gy.StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=gy.panmictic(d=d)
    else:
        print("Model incorrect")
        sys.exit()

    print(d["mtype"])
    print("The model is", d.get("mtype"))

    it=gy.Repetition_model(r,i,p) # parse the repetition
    gen=gy.FSC2_read_gen(it.gen_path)
    
    print("Calling PSMC...",i)
    ## Do each individual
    individuals = int(r.samples[int(p)-1]/2)
    print("Number of individuals sampled %s" % individuals)

    for individual in range(1,individuals+1):

        print("Running gy.PSMC_consensus_fa")
        consensus=gy.PSMC_consensus_fa(it,r,gen,p, individual) # TELL INDIVIDUAL
        consensus_fq=gy.PSMC_convert_fastq(consensus,it, individual) 
        ### UP TO THIS POINT

     # Control if one bin size or two
    if two_bin_sizes:
        svalues=[100]
    elif study_bin_size:
        svalues=[100,70,50,30,20]
        individuals=1
    else: 
        svalues=[s]
    
    all_lpsmc={} # initialize dictionary
    for svalue in svalues:
        all_lpsmc[str(svalue)]=[] # initialize dictionary entry
    for individual in range(1,individuals+1):
        
        consensus_fq=it.fullname+"_ind_"+str(individual)+".consensus.fq.gz" ## USE THIS TO ITERATE
        print("Start with individual "+str(individual)+" and file \n %s" % consensus_fq)
    
        
        # Considering results of genetic diversity, run for 2 bin sizes
        
       
        
        for svalue in svalues: 

            print("svalue: %s" % svalue)
            # run for each bin size. If two_bin_sizes==False, only run once as ussually
            # as snif results folder is removed in each round snif results are not mixed up
            # between different bin size
            svalue=str(svalue)
            print("Run psmcfa")
            psmcfa=gy.PSMC_convert_input(consensus_fq,it,individual,svalue)
            psmc=gy.PSMC_run(psmcfa,it,r,individual,svalue)
            #psmc=it.fullname+"_s"+svalue+".psmc"
            lpsmc=gy.PSMC_last_iteration(it,r,individual,svalue) # get last iteration
            print("I got a PSMC for %s bin size in individual" % str(svalue))
            print(lpsmc)

            all_lpsmc[svalue].append(lpsmc)

        ## UP TO HERE, EACH INDIVIDUAL
        # Do SST and run snif with summary
    print("All my PSMCs are here:")
    print(all_lpsmc)

    for svalue in svalues:
        sample_dicts=[]
        this_s_all=all_lpsmc.get(str(svalue))
        for last_psmc in this_s_all:
            # sample PSMC
            this_sample_dict=gy.PSMC_sample(psmc=last_psmc,svalue=svalue,mu=float(r.mu))
            sample_dicts.append(this_sample_dict)
        gy.PSMC_mean_inferences(sample_dicts=sample_dicts, svalue=svalue,r=r,it=it)

    resume_files=[] #########
    snif_run=False
    if snif_run:
        for psmc in resume_files:#####
            snif_start=time.time()

            snif_results_folder=gy.SNIF_run(psmc,r,it)  ####################### runs snif 

            snif_finish=time.time()
            runtime=round(snif_finish-snif_start, 2)

            with open("gyarados_time_check.log","a") as f:
                print(r.name,"SNIF",str(runtime),str(svalue), sep="\t", file=f)

            # Compress folder
            snif_compressed_folder=it.fullname+"_s"+str(svalue)+"_snif.tar.gz "
            sentence="tar -czf "+snif_compressed_folder+" "+snif_results_folder
            print(sentence)
            os.system(sentence)

            # remove snif_results_folder
            sentence="rm -rf "+snif_results_folder
            print(sentence)
            os.system(sentence)

            # Copy to results
            sentence="cp "+snif_compressed_folder+" RESULTS/"+r.name+"/"
            print(sentence)
            os.system(sentence)
    

            finish_time=time.time()
    
            runtime=round(finish_time-start_time,2)
    
        with open("gyarados_time_check.log","a") as f:
            print(r.name,"ALL_PSMC_SNIF",str(runtime),"NA", sep="\t",file=f)


def PSMC_sample(psmc,svalue, mu, scheme="log"):
    print("Sampling PSMC...")
    svalue=int(svalue)
    sample_dict={}

    if scheme == "both":
        print("Do both sampling schemes")
        scheme = ["linear", "log"]
    else:
        scheme = [scheme]
    print("Length of sampling scheme list is: " + str(len(scheme)))
    print(*scheme)
    for sc in scheme:
        
        if sc == "linear":
            print(sc,": linear sampling")
            xvector=lin_vector
            print("The vector length is: " + str(len(xvector)))
        elif sc == "log":
            print(sc,": log sampling")
            xvector=log_vector
            print("The vector length is: " + str(len(xvector)))
        else:
            print("Error, sc is", sc)
            return "Incorrect sampling scheme"
        file=psmc
        with open(file) as f:
            all_file=f.readlines()[1:-1]
            all_file=[x.strip() for x in all_file] # remove \n at the end of each line
            times = [line.split("\t")[2] for line in all_file if line.startswith("RS")]
            lambdas = [line.split("\t")[3] for line in all_file if line.startswith("RS")]
            theta = float([line.split("\t")[1] for line in all_file if line.startswith("TR")][0])
        N0=theta/(4*float(mu)*svalue)
        generations=[2*N0*float(t) for t in times]
        N=[N0*float(lmd) for lmd in lambdas]
        psmc_in=pd.DataFrame({"generations":generations,"N":N})
        ## Create the xvector df empty
        nans=np.empty(len(xvector))
        nans[:]=np.nan
        x_df=pd.DataFrame({"generations":xvector, "N":nans})
        ## rbind with the original dataframe
        psmc_in=pd.concat([psmc_in,x_df])
        psmc_in=psmc_in.sort_values(by=["generations"])
        print("filling na...")
        psmc_in=psmc_in.fillna(method="ffill")
        psmc_sampled=psmc_in[psmc_in.generations.isin(xvector)]
        psmc_sampled=psmc_sampled.drop_duplicates(subset=["generations"]) 
        this_psmc_sampling=list(psmc_sampled["N"])
        print("Adding to the", type(sample_dict),"a list of length", len(this_psmc_sampling))
        sample_dict[sc]=this_psmc_sampling
        print("Type in sampling:", type(sample_dict.get(sc)))
    return sample_dict


def PSMC_mean_inferences(sample_dicts, svalue,r, it, save_tmp=True, p=None, psmc_pattern=DEFAULT_PSMC_PATTERN):
    svalue=str(svalue)
    pattern_suffix = gy.psmc_pattern_suffix(psmc_pattern)
    print("Calculating mean of psmc inferences for model ", r.id)
    #os.system("touch "+str(r.id)+"_rep"+str(it.rep)+".txt")
    log_sampled=[]
    lin_sampled=[]
    print(type(sample_dicts))
    params=["#"+par for par in r.params]
    for dic in sample_dicts:
        print(type(dic))
        if "log" in dic:
            log_sampled.append(dic.get("log"))
        elif "linear" in dic:
            lin_sampled.append(dic.get("linear"))
        else:
            print("Not accepted sample schemes")
    print("There are "+str(len(log_sampled))+" log samples and "+str(len(lin_sampled))+" linear samples")   

    # Do the mean
    if len(log_sampled):
        print("Mean for log sampled")
        log_sampled=np.array(log_sampled)
        mean_log_psmc=np.mean(log_sampled, axis=0)
        # write in file if it doesn't exist
        
        tyop="gyarados_"+r.mtype+"_s"+str(svalue)+pattern_suffix+"_psmc.log.out"
        if (os.path.exists("RESULTS/"+tyop) == False):
            o = open("RESULTS/"+tyop, "w")
            params=["#"+par for par in r.params]
            header=["#ID"]+params+["#Repetition","#Time","#IICR", "#PSMC_Inference"]
               
            line="\t".join(header)+"\n"
            print(line)
            o.write(line)
            o.close()

        #tmp_all="gyarados_"+r.mtype+"_s"+str(svalue)+"_psmc_rep_"+str(it.rep)+".log.out"
        #if save_tmp:
        #    with open("RESULTS/"+tmp_all, "w") as tmp:
        #        for ind in list(log_sampled):
        #            line=" ".join(str(element) for element in ind)+"\n"
        #            tmp.write(line)

        # prepare results
        if r.mtype == "SST" or r.mtype == "Free":
            log_iicr_df=pd.read_csv(r.iicr_log_path[p], sep="\t") 
        else:
            log_iicr_df=pd.read_csv(r.iicr_log_path, sep="\t")
        str_log_iicr=" ".join(str(element) for element in list(log_iicr_df["scaled_lmd"]))
        str_log_time=" ".join(str(element) for element in log_vector)
        str_log_psmc=" ".join(str(element) for element in mean_log_psmc)    
        
        # write results
        if r.mtype=="StSi":
                print("printing list to fill")
                listofill=[str(r.id),str(r.N[0]),str(r.islands),str(r.mig[0]),str(it.rep),str_log_time,str_log_iicr,str_log_psmc]
                print(*listofill)
        elif r.mtype=="SST":
            print("printing list to fill")
            listofill=[str(r.id),str(r.N[0]),str(r.mig[0]), str(r.M),str(r.L),str(p),str(it.rep),str_log_time,str_log_iicr,str_log_psmc]
            print(*listofill)
        elif r.mtype == "Free":
            print("printing list to fill")
            listofill=[r.id,  str(r.islands), str(r.n_mm), str(r.n_events), str(r.gr[0]), r.name,str(it.pop),str(it.rep),str_log_time,str_log_iicr,str_log_psmc]
        else:
            print("Model "+r.mtype+" still not implemented")


        with open("RESULTS/"+tyop, "a") as logf:
            print("Write new line")
            line="\t".join(listofill)+"\n"
            logf.write(line)      

    if len(lin_sampled):
        print("Mean for linear sampling")
        lin_sampled=np.array(lin_sampled)
        mean_lin_psmc=np.mean(lin_sampled, axis=0)
        # write in file if it doesn't exist

        tyop="gyarados_"+r.mtype+"_s"+str(svalue)+pattern_suffix+"_psmc.lin.out"
        if (os.path.exists("RESULTS/"+tyop) == False):
            o = open("RESULTS/"+tyop, "w")
            params=["#"+par for par in r.params]
            header=["#ID"]+params+["#Repetition","#Time","#IICR", "#PSMC_Inference"]
            line="\t".join(header)+"\n"
            print(line)
            o.write(line)
            o.close()
        #tmp_all="gyarados_"+r.mtype+"_s"+str(svalue)+"_psmc_rep_"+str(it.rep)+".lin.out"
        #if save_tmp:
        #    with open("RESULTS/"+tmp_all, "w") as tmp:
        #        for ind in list(lin_sampled):
        #            line=" ".join(str(element) for element in ind)+"\n"
        #            tmp.write(line)
                    
        # prepare results
        if r.mtype == "SST" or r.mtype == "Free":
            lin_iicr_df=pd.read_csv(r.iicr_lin_path[p], sep="\t") 
        else:
            lin_iicr_df=pd.read_csv(r.iicr_lin_path, sep="\t")
        str_lin_iicr=" ".join(str(element) for element in list(lin_iicr_df["scaled_lmd"]))
        str_lin_time=" ".join(str(element) for element in lin_vector)
        str_lin_psmc=" ".join(str(element) for element in mean_lin_psmc)    
        
        # write results



        # write results
        if r.mtype=="StSi":
            print("printing list to fill")
            listofill=[r.id, str(r.N[0]),str(r.islands),str(r.mig[0]), str(r.M),str(it.rep),str_lin_time,str_lin_iicr,str_lin_psmc]
            print(*listofill)
        elif r.mtype == "SST":
            print("printing list to fill")
            listofill=[r.id, str(r.N[0]),str(r.mig[0]), str(r.M),str(r.L),str(p),str(it.rep),str_lin_time,str_lin_iicr,str_lin_psmc]
            print(*listofill) ###############################################################################################################
        elif r.mtype == "Free":
            print("printing list to fill")
            listofill=[r.id,  str(r.islands), str(r.n_mm), str(r.n_events), str(r.gr[0]), r.name,str(it.pop),str(it.rep),str_log_time,str_log_iicr,str_log_psmc]
        else:
            print("Model "+r.mtype+" still not implemented")
                    
        with open("RESULTS/"+tyop, "a") as linf:
            print("Write new line")
            line="\t".join(listofill)+"\n"
            linf.write(line)

def PSMC_consensus_fa(it,r,gen,p,individual,psmc_pattern=DEFAULT_PSMC_PATTERN):
    '''
    - it: Mk repetition model object
    - r: MK model object
    - p: population
    - gen: pd. gen table object
    Outs the consensus.fa file for 1 individual. Intermediate file for PSMC
    

    '''
    print("Entering PSMC_consensus_fa")
    co=[]
    pattern_suffix = gy.psmc_pattern_suffix(psmc_pattern)
    consensus=it.fullname+"_deme_"+str(p)+"_ind_"+str(individual)+pattern_suffix+".consensus.fa"
    
    #Doing it for all the individuals sampled

   
   # while a!=int(r.chr) and individual<=int(r.samples[int(p)-1]):
   #     print("Inside the while loop")
   #     print(a, individual)
   #     a=0
   #     individual+=1
   #     sentence="rm "+it.fullname+"_*"+".consensus.fatmp"
   #     if individual!=1:
   #         os.system("echo 'Removing failed consensus'")
   #         os.system(sentence)

    for c in range(len(list(r.sizes))):
        chr=c+1
        size=int(r.sizes[c])
        tmp_fa=r.name+"/"+r.name+"_"+str(chr)+".fa"# temporary fa for chromosome
            #line=("A"*60+"\n") 
    # 1.2. Change the heterozygote positions
        this_chr=gen[gen["Chrom"]==chr]
        if int(p):
            # Sample the individual we want (first from the designed population)
            this_chr=gy.FSC2_gen_filter_1i(this_chr,p,individual=individual)
            #this_chr.to_csv(it.fullname+"_"+str(chr)+".genind")
                # call again FSC2 for this repetition
                #sentence="rm -rf "+it.dir
                #gy.EQ_SIM_NISLANDS_WORK(it.rep, r.json, p) # call all again
                # and cancel all the jobs related 
        consensus_tmp=it.fullname+"_"+str(chr)+"_deme_"+str(p)+"_ind_"+str(individual)+pattern_suffix+".consensus.fatmp" # we edit this
        shutil.copyfile(tmp_fa, consensus_tmp) # copy it in its corresponding path
        f=os.open(consensus_tmp,os.O_RDWR) # open consensus.fa file
        m=mmap.mmap(f,0) # map the file. m is an mmap object

        # m is like a list. in each position are the characters
        # of my fa file in binary (includding header)
        # Each position in .gen is +2 in .fa index
        positions=list(this_chr.Pos)
        print(len(positions))
        lastpos=positions[-1]
        # print(*positions)
        position_count=0 
        for snp in positions:
            # +2 and +3 to get the end (needs to be like this)
            init=snp+2
            end=snp+3
            m[init:end]=b"S" # b=binary
            position_count+=1
        print(position_count, "added positions in",chr)
        #for pos in list(this_chr.Pos):
            # +2 and +3 to get the end (needs to be like this)
        #   if m[snp+2:snp+3]!=b"S":
        #       print(snp) # b=binary
        co.append(this_chr)
        os.close(f) # close
        print(consensus_tmp, "has been created")

    print("Analyses performed with individual", str(individual), "from population", str(p))


    time.sleep(180)
    
    if int(r.chr)==1: # only one chromosome. Fix the bug of non copying the consensus
        print("Only one chromosome in this model")
        #shutil.copyfile(consensus_tmp, consensus)
        print(consensus_tmp)
        sentence="cat "+consensus_tmp+" | fold -b60 > "+consensus
        os.system(sentence)
        print(sentence)
    else:
        print("More than one chromosome")
        # Paste them 
        # Past all the consensus.fatmp in a single consensus fa
        sentence="cat "+it.fullname+"_*"+"_deme_"+str(p)+"_ind_"+str(individual)+pattern_suffix+".consensus.fatmp | fold -b60 > "+consensus ###########################
        os.system(sentence)  
        print(sentence)
    
    return consensus

def PSMC_consensus_fa2(it,r,gen,p):
    '''
    - it: Mk repetition model object
    - r: MK model object
    - p: population
    - gen: pd. gen table object
    Outs the consensus.fa file for 1 individual. Intermediate file for PSMC
    

    '''
    co=[]
    consensus=it.fullname+".consensus.fa"

    for c in range(len(list(r.sizes))):
        chr=c+1
        size=int(r.sizes[c])
        tmp_fa=r.name+"/"+r.name+"_"+str(chr)+".fa"# temporary fa for chromosome
            #line=("A"*60+"\n") 
    # 1.2. Change the heterozygote positions
        this_chr=gen[gen["Chrom"]==chr]
        if int(p):
            # Sample the individual we want (first from the designed population)
            this_chr=gy.FSC2_gen_filter_1i(this_chr,p)
            this_chr.to_csv(it.fullname+"_"+str(chr)+".genind")
        consensus_tmp=it.fullname+"_"+str(chr)+".consensus.fatmp" # we edit this
        shutil.copyfile(tmp_fa, consensus_tmp) # copy it in its corresponding path
        f=os.open(consensus_tmp,os.O_RDWR) # open consensus.fa file
        m=mmap.mmap(f,0) # map the file. m is an mmap object

        # m is like a list. in each position are the characters
        # of my fa file in binary (includding header)
        # Each position in .gen is +2 in .fa index
        positions=list(this_chr.Pos)
        print(len(positions))
        lastpos=positions[-1]
        # print(*positions) 
        for snp in positions:
            # +2 and +3 to get the end (needs to be like this)
            init=snp+2
            end=snp+3
            m[init:end]=b"S" # b=binary
        #for pos in list(this_chr.Pos):
            # +2 and +3 to get the end (needs to be like this)
        #   if m[snp+2:snp+3]!=b"S":
        #       print(snp) # b=binary
        co.append(this_chr)
        os.close(f) # close
        print(consensus_tmp, "has been created")

    # Paste them 
    # Past all the consensus.fatmp in a single consensus fa
    sentence="cat "+it.fullname+"_*"+".consensus.fatmp | fold -b60 > "+consensus  
    print(sentence)
    os.system(sentence)
    
    return consensus


def PSMC_convert_fastq(consensus,it,individual,p,psmc_pattern=DEFAULT_PSMC_PATTERN):
    '''
    Convert the consensus .fa into a fastq.gz
    '''
    #it.fullname+"_ind_"+str(individual)+".consensus.fa"
    pattern_suffix = gy.psmc_pattern_suffix(psmc_pattern)
    consensus_fq=it.fullname+"_deme_"+str(p)+"_ind_"+str(individual)+pattern_suffix+".consensus.fq.gz"
    sentence="cat "+consensus+" | seqtk seq -F 'I' | gzip > "+consensus_fq
    os.system(sentence)
    print(consensus_fq+" created")

    return consensus_fq

def PSMC_convert_input(consensus_fq,it, individual,p,s=100,psmc_pattern=DEFAULT_PSMC_PATTERN):
    '''
    Get proper psmc input
    '''
    s=str(s)
    pattern_suffix = gy.psmc_pattern_suffix(psmc_pattern)
    psmcfa=it.fullname+"_s"+s+"_deme_"+str(p)+"_ind_"+str(individual)+pattern_suffix+".consensus.psmcfa"
    sentence="fq2psmcfa -q0 -s "+str(s)+" "+consensus_fq+" > "+psmcfa
    os.system(sentence)
    print(psmcfa+" created")
    return psmcfa

def PSMC_last_iteration(it,r,individual,p,s=100,psmc_pattern=DEFAULT_PSMC_PATTERN):

    '''
    Reads psmc output file (name.psmc) and extracts only the last iteration
    Creates a new file name.lpsmc
    Returns name.lpsmc path
    '''
    s=str(s)
    output_prefix = gy.psmc_output_prefix(it, s, p, individual, psmc_pattern)
    with open(output_prefix+'.psmc') as n:
        last=n.read().split("//")[-2] # last iteration: pre last file
    with open(output_prefix+".lpsmc", "w") as l:
        l.write(last)
    print(output_prefix+".lpsmc has been created")
    sentence="tail -n3 "+output_prefix+".lpsmc"
    os.system(sentence)
    # Copy to results
    sentence="cp "+output_prefix+".lpsmc RESULTS/"+r.name+"/"
    print(sentence)
    os.system(sentence)
    return output_prefix+".lpsmc"


def PSMC_run(psmcfa,it,r,individual,p,s=100, onerep=True, psmc_pattern=DEFAULT_PSMC_PATTERN):
    print("Removing consensus fa and all intermediate files")
    if onerep:
        print("Keeping model reference .fa files for other individuals and PSMC vectors")
    pattern_suffix = gy.psmc_pattern_suffix(psmc_pattern)
    sentence="rm "+it.fullname+"*"+pattern_suffix+".consensus.fatmp" 
    os.system(sentence)
    sentence="rm "+it.fullname+"*"+pattern_suffix+".consensus.fa" 
    os.system(sentence)
    s=str(s)
    start_time = time.time()
    output_prefix = gy.psmc_output_prefix(it, s, p, individual, psmc_pattern)
    sentence='psmc -t15 -r1 -p '+shlex.quote(str(psmc_pattern))+' -d -o '+output_prefix+'.psmc '+psmcfa # conventional
    #sentence='psmc -t15 -r1 -p "27*2+4+6" -o '+it.fullname+'_s'+s+'_deme_'+str(p)+'_ind_'+str(individual)+'.psmc '+psmcfa # Times1
    #sentence='psmc -t15 -r1 -p "2*2+2*2+25*2+1*4+1*6" -o '+it.fullname+'_s'+s+'_deme_'+str(p)+'_ind_'+str(individual)+'.psmc '+psmcfa # Times2
    #sentence='psmc -t15 -r1 -p "16*1+19*2+1*4+1*6" -o '+it.fullname+'_s'+s+'_deme_'+str(p)+'_ind_'+str(individual)+'.psmc '+psmcfa #Times2
    # sentence='psmc -t40 -r1 -p "4+25*2+4+6" -o '+it.fullname+'_s'+s+'_deme_'+str(p)+'_ind_'+str(individual)+'.psmc '+psmcfa # conventional #nisha
    os.system(sentence)
    psmc=output_prefix+'.psmc'
    # Copy to results
    sentence="cp "+psmc+" RESULTS/"+r.name+"/"
    print(sentence)
    os.system(sentence)
    print("PSMC has been run succesfully in ", psmc)


    finish_time=time.time()
    
    runtime=round(finish_time-start_time,2)
    sentence="rm "+it.fullname+"*"+pattern_suffix+".consensus*" 
    os.system(sentence)

    with open("gyarados_time_check.log","a") as f:
        print(r.name,"PSMC",str(runtime),str(s)+"/"+str(individual)+"/"+gy.psmc_pattern_label(psmc_pattern), sep="\t", file=f)
    print("Remove psmcfa")
    sentence="rm "+psmcfa  
    os.system(sentence)

    return psmc



    


##############################################################################################################
#########################                      SMC++ FUNCTIONS                  ##############################
##############################################################################################################

def SMC_call(i,j_name,p):
    sentence="sbatch call_smcpp_gy.sh "+ str(i) +" "+str(j_name)+" "+str(p)
    #sentence="bash call_smcpp_gy.sh "+ str(i) +" "+str(j_name)+" "+str(p)
    os.system(sentence)


def SMC_vcf2smc(vcf,inds,it,r,p):
    '''
    Runs the vcf2smc function from smc++ to get an smc.gz file per chromosome 
    from the vcf file
    Input:
    - vcf: path to the vcf file (take from FSC2_gen2vcf)
    - inds: individuals array (take from FSC2_gen2vcf)
    - it: rep object
    - r: model object
    - p: population
    '''
    undis="G_"+str(p)+":"+",".join(inds) # undis individuals sentence
    # bgzip the vcf
    print("BGzipping vcf file")
    print("population", p)
    vcfgz=vcf+".gz"
    sentence="bgzip -c "+vcf+" > "+vcfgz
    print(sentence)
    os.system(sentence)
    # index the vcf
    print("Indexing vcf file")
    sentence="tabix "+vcfgz
    os.system(sentence)
    # create the smc++ folder 
    sentence="mkdir "+it.dir+"/smcpp" # creates smc++ folder
    os.system(sentence)
    # Run vcf2smc per chromosome
    # smc++ vcf2smc -d G_1_1 G_1_1 toy.vcf.gz toy.4.smc.gz 4 G_1:G_1_1,G_1_2,G_1_3,G_1_4,G_1_5,G_1_6,G_1_7,G_1_8,G_1_9,G_1_10 --length 10000000
    for dis in inds: #It will create a file like it.name_ind_1.4.smc.gz
        print("dis: %s" % dis)
        for i in range(r.chr): 
            c=i+1 # chromosome we are now
            print("Running vcf2smc for chromosome",str(c))
            smcgz=it.dir+"/smcpp/"+it.name+"_deme_"+str(p)+"_ind_"+str(dis)+"."+str(c)+".smc.gz"
            sentence="smc++ vcf2smc -d "+dis+" "+dis+" "+vcfgz+" "+smcgz+" "+str(c)+" "+undis+" --length "+str(r.sizes[i])
            print(sentence)
            os.system(sentence)


def SMC_estimate(it,r,dis,p,s=0):
    '''
    Runs smc++ estimation using all the smc.gz files generated for this repetition
    '''
    #smc++ estimate 1e-8 toy.*.smc.gz
    #sentence="smc++ estimate "+str(r.mu)+" "+it.dir+"/smcpp/*_deme_"+str(p)+"_ind_"+dis+".*.smc.gz -o "+it.dir+"/smcpp --base _deme_"+str(p)+"_ind_"+str(dis)+"_s"+str(s)+" --timepoints "+str(s)+" 100000"
    sentence="smc++ estimate "+str(r.mu)+" "+it.dir+"/smcpp/*_deme_"+str(p)+"_ind_"+dis+".*.smc.gz -o "+it.dir+"/smcpp --base _deme_"+str(p)+"_ind_"+str(dis)+"_s"+str(s)+" --timepoints "+str(s)+" 100000 --knots 40"
    
    print("Running SMC++ estimation")
    print(sentence)
    os.system(sentence)
    # Change the name with the starting point
    model_json = it.dir+"/smcpp/_deme_"+str(p)+"_ind_"+str(dis)+"_s"+str(s)+".final.json"
    return model_json

def SMC_arrange(model_json,it,dis,p,s=0):
    '''
    Calls the smc++ plot function to get the csv output (and a png)
    '''
    # smc++ plot plot.png --csv model.final.json
    sentence="smc++ plot "+it.dir+"/smcpp/"+it.name+"_deme_"+str(p)+"_ind_"+str(dis)+"_start_"+str(s)+".png"+" --csv "+model_json
    os.system(sentence)
    csv_smcpp=it.dir+"/smcpp/"+it.name+"_deme_"+str(p)+"_ind_"+str(dis)+"_start_"+str(s)+".csv"
    return csv_smcpp

def SMC_sample(csv_smcpp, scheme="log"):
    print("Sampling SMCpp")
    sample_dicts={}
    smcpp_raw=pd.read_csv(csv_smcpp)

    if scheme == "both":
        print("Do both sampling schemes")
        scheme = ["linear", "log"]
    else:
        scheme = [scheme]
    print("Length of sampling scheme list is: " + str(len(scheme)))
    print(*scheme)
    for sc in scheme:
        
        if sc == "linear":
            print(sc,": linear sampling")
            xvector=lin_vector
            print("The vector length is: " + str(len(xvector)))
        elif sc == "log":
            print(sc,": log sampling")
            xvector=log_vector
            print("The vector length is: " + str(len(xvector)))
        else:
            print("Error, sc is", sc)
            return "Incorrect sampling scheme"
        smcpp_in=pd.DataFrame({"generations":list(smcpp_raw["x"]), "N":list(smcpp_raw["y"])})
        ## Create the xvector df empty
        nans=np.empty(len(xvector))
        nans[:]=np.nan
        x_df=pd.DataFrame({"generations":xvector, "N":nans})
        ## rbind with the original dataframe
        smcpp_in=pd.concat([smcpp_in,x_df])
        smcpp_in=smcpp_in.sort_values(by=["generations"])
        print("filling na...")
        smcpp_in=smcpp_in.fillna(method="ffill")
        smcpp_sampled=smcpp_in[smcpp_in.generations.isin(xvector)]
        smcpp_sampled=smcpp_sampled.drop_duplicates(subset=["generations"]) # Remove duplicates JIC
        this_smcpp_sampling=list(smcpp_sampled["N"])
        print("Adding to the", type(sample_dicts),"a list of length", len(this_smcpp_sampling))
        sample_dicts[sc]=this_smcpp_sampling
        print("Type in sampling:", type(sample_dicts.get(sc)))
        
    return sample_dicts    

def SMC_mean_inferences(sample_dicts,r, it,p=None, save_tmp=True):
    print("Calculating mean of smcpp inferences for model ", r.id)
    #os.system("touch "+str(r.id)+"_rep"+str(it.rep)+".txt")
    log_sampled=[]
    lin_sampled=[]
    print(type(sample_dicts))
    params=["#"+par for par in r.params]
    for dic in sample_dicts:
        print(type(dic))
        if "log" in dic:
            log_sampled.append(dic.get("log"))
        if "linear" in dic:
            lin_sampled.append(dic.get("linear"))
        else:
            print("Not accepted sample schemes")
    print("There are "+str(len(log_sampled))+" log samples and "+str(len(lin_sampled))+" linear samples")   

    # Do the mean
    if len(log_sampled):
        print("Mean for log sampled")
        log_sampled=np.array(log_sampled)
        mean_log_smcpp=np.mean(log_sampled, axis=0)
        # write in file if it doesn't exist
        
        tyop="gyarados_"+r.mtype+"_smcpp.log.out"
        if (os.path.exists("RESULTS/"+tyop) == False):
            o = open("RESULTS/"+tyop, "w")
            params=["#"+par for par in r.params]
            header=["#ID"]+params+["#Repetition","#Time","#IICR", "#SMCpp_Inference"]
            line="\t".join(header)+"\n"
            print(line)
            o.write(line)
            o.close()

        #tmp_all="gyarados_"+r.name+"_smcpp_rep_"+str(it.rep)+".log.out"
        #if save_tmp:
        #    with open("RESULTS/"+tmp_all, "w") as tmp:
        #        for ind in list(log_sampled):
        #            line=" ".join(str(element) for element in ind)+"\n"
        #           tmp.write(line)

        # prepare results
        # prepare results
        if r.mtype == "SST" or r.mtype == "Free":
            log_iicr_df=pd.read_csv(r.iicr_log_path[p], sep="\t") 
        else:
            log_iicr_df=pd.read_csv(r.iicr_log_path, sep="\t")
        
        str_log_iicr=" ".join(str(element) for element in list(log_iicr_df["scaled_lmd"]))
        str_log_time=" ".join(str(element) for element in log_vector)
        str_log_smcpp=" ".join(str(element) for element in mean_log_smcpp)    
        
        # write results
        if r.mtype=="StSi":
                print("printing list to fill")
                listofill=[str(r.id),str(r.N[0]),str(r.islands),str(r.mig[0]),str(it.rep),str_log_time,str_log_iicr,str_log_smcpp]
                print(*listofill)
        elif r.mtype=="SST":
            print("printing list to fill")
            listofill=[str(r.id),str(r.N[0]),str(r.mig[0]), str(r.Nm),str(r.L),str(p),str(it.rep),str_log_time,str_log_iicr,str_log_smcpp]
            print(*listofill)
        elif r.mtype == "Free":
            print("printing list to fill")
            listofill=[r.id,  str(r.islands), str(r.n_mm), str(r.n_events), str(r.gr[0]), r.name,str(it.pop),str(it.rep),str_log_time,str_log_iicr,str_log_smcpp]
        else:
            print("Model "+r.mtype+" still not implemented")


        with open("RESULTS/"+tyop, "a") as logf:
            print("Write new line")
            line="\t".join(listofill)+"\n"
            logf.write(line)      

    if len(lin_sampled):
        print("Mean for linear sampling")
        lin_sampled=np.array(lin_sampled)
        mean_lin_smcpp=np.mean(lin_sampled, axis=0)
        # write in file if it doesn't exist

        tyop="gyarados_"+r.mtype+"_smcpp.lin.out"
        if (os.path.exists("RESULTS/"+tyop) == False):
            o = open("RESULTS/"+tyop, "w")
            params=["#"+par for par in r.params]
            header=["#ID"]+params+["#Repetition","#Time","#IICR", "#SMCpp_Inference"]
            line="\t".join(header)+"\n"
            print(line)
            o.write(line)
            o.close()
        #tmp_all="gyarados_"+r.name+"_smcpp_rep_"+str(it.rep)+".lin.out"
        #if save_tmp:
        #    with open("RESULTS/"+tmp_all, "w") as tmp:
        #        for ind in list(lin_sampled):
        #            line=" ".join(str(element) for element in ind)+"\n"
        #            tmp.write(line)
                 
        # prepare results
        if r.mtype == "SST" or r.mtype == "Free":
            lin_iicr_df=pd.read_csv(r.iicr_lin_path[p], sep="\t") 
        else:
            lin_iicr_df=pd.read_csv(r.iicr_lin_path, sep="\t")
        str_lin_iicr=" ".join(str(element) for element in list(lin_iicr_df["scaled_lmd"]))
        str_lin_time=" ".join(str(element) for element in lin_vector)
        str_lin_smcpp=" ".join(str(element) for element in mean_lin_smcpp)    
        
        # write results



        # write results
        if r.mtype=="StSi":
            print("printing list to fill")
            listofill=[r.id, str(r.N[0]),str(r.islands),str(r.mig[0]), str(r.M),str(it.rep),str_lin_time,str_lin_iicr,str_lin_smcpp]
            print(*listofill)
        elif r.mtype == "SST":
            print("printing list to fill")
            listofill=[r.id, str(r.N[0]),str(r.mig[0]), str(r.Nm),str(r.L),str(p),str(it.rep),str_lin_time,str_lin_iicr,str_lin_smcpp]
            print(*listofill)
        elif r.mtype == "Free":
            print("printing list to fill")
            listofill=[r.id,  str(r.islands), str(r.n_mm), str(r.n_events), str(r.gr[0]), r.name,str(it.pop),str(it.rep),str_lin_time,str_lin_iicr,str_lin_smcpp]  ###############################################################################################################
        else:
            print("Model "+r.mtype+" still not implemented")
                    
        with open("RESULTS/"+tyop, "a") as linf:
            print("Write new line")
            line="\t".join(listofill)+"\n"
            linf.write(line)


def SMC_estimate_sample_per_individual(dis, it, r, p,s=0):
    # Prepare to run each individual in a parallel process
    # This function should be created inside the main SMCPP function
    # it, r and i should be local variables
    print("Starting smcpp with individual %s" % dis)
    start_inf_time=time.time()
    print("Estimating demographic model in SMC++")
    model_json=gy.SMC_estimate(it,r,dis,p) # RUN SMCPP estimation
    #model_json=it.dir+"/smcpp/model.final.json"
    print("Arrange and plot results for SMC++")
    csv_smcpp=gy.SMC_arrange(model_json,it,dis,p) # GET THE CSV FROM SMCPP ESTIMATION
    # Copy to results
    finish_time_inference=time.time()
    inference_time=round(finish_time_inference-start_inf_time,2)
    with open("gyarados_time_check.log","a") as f:
        print(r.name,"SMCPP",str(inference_time),dis, sep="\t", file=f)
    sentence="cp "+csv_smcpp+" RESULTS/"+r.name+"/"
    print(sentence)
    os.system(sentence)

    print("Starting sampling for %s" % dis)
    # Sample vector
    this_sample_dicts=gy.SMC_sample(csv_smcpp) # SAMPLE THE VECTOR OF TIMES OUT OF THE CSV
    print("After sampling, the type is:", type(this_sample_dicts))
    print("For individual %s, we have the following keys:" % dis)
    return this_sample_dicts  # RETURN THE VECTOR OF SAMPLED TIMES


def SMC_run(i,j_name,p):
    #SMC_run(gen,it,r,p)
    #os.system(bash work.sh i json p)
    start_time=time.time()
    d=gy.get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")
    print("Main is:", __name__)

    if type_model=="StSi":
       r=gy.StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=gy.panmictic(d=d)
    elif type_model=="SST":
        r=gy.SST(d=d) # parse the model
    elif type_model=="Free":
        r=gy.Free(d=d) # parse the mode
    else:
        print("Model incorrect")
        sys.exit()

    print(d["mtype"])
    print("The model is", d.get("mtype"))

    it=gy.Repetition_model(r,i,p) # parse the repetition
    gen=gy.FSC2_read_gen(it.gen_path)

    print("Starting smc++ run...",str(it.rep))
    vcf,inds=gy.FSC2_gen2VCF(gen,it,p)
    gy.SMC_vcf2smc(vcf,inds,it,r,p)
    finish_time_vcf=time.time()
    time_vcf=finish_time_vcf-start_time
    # Do for each sampled individual
    sample_dicts=[]
    start_time=time.time()

    results=[]

    #pool = multiprocessing.Pool(processes=len(inds)) # as many processes as individuals
    for dis in inds:
        results.append(SMC_estimate_sample_per_individual(dis,it,r, p))
    
    #sample_dicts = [p.get() for p in results]
    sample_dicts = results
    print("Now we have %d entries (individuals) in the dict list" % len(sample_dicts))
    # Call mean and write
    gy.SMC_mean_inferences(sample_dicts,r,it,p)

    with open("gyarados_time_check.log","a") as f:
        print(r.name,"SMCPP",str(time.time()-start_time),"ALL_RUN", sep="\t", file=f)

    #sentence="rm "+it.dir+"/*vcf"
    #os.system(sentence)
    #sentence="rm "+it.dir+"/*vcf.gz"
    #os.system(sentence)
    #sentence="rm "+it.dir+"/*tbi"
    #os.system(sentence)
    #sentence="rm -rf "+it.dir+"/smcpp/"
    #os.system(sentence)
    
def SMC_run_par(i,j_name,p):
    #SMC_run(gen,it,r,p)
    #os.system(bash work.sh i json p)
    start_time=time.time()
    d=gy.get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")
    print("Main is:", __name__)

    if type_model=="StSi":
       r=gy.StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=gy.panmictic(d=d)
    else:
        print("Model incorrect")
        sys.exit()

    print(d["mtype"])
    print("The model is", d.get("mtype"))

    it=gy.Repetition_model(r,i,p) # parse the repetition
    gen=gy.FSC2_read_gen(it.gen_path)

    print("Starting smc++ run...",str(it.rep))
    vcf,inds=gy.FSC2_gen2VCF(gen,it,p)
    gy.SMC_vcf2smc(vcf,inds,it,r,p)
    finish_time_vcf=time.time()
    time_vcf=finish_time_vcf-start_time
    # Do for each sampled individual
    sample_dicts=[]
    start_time=time.time()

    results=[]

    pool = multiprocessing.Pool(processes=len(inds)) # as many processes as individuals
    for dis in inds:
        results.append(pool.apply_async(SMC_estimate_sample_per_individual, args=(dis,it,r)))
    pool.close()
    pool.join()
    sample_dicts = [p.get() for p in results]
    print("Now we have %d entries (individuals) in the dict list" % len(sample_dicts))
    # Call mean and write
    gy.SMC_mean_inferences(sample_dicts,r,it)

    del pool
    del sample_dicts
    del gen
    with open("gyarados_time_check.log","a") as f:
        print(r.name,"SMCPP",str(time.time()-start_time),"ALL_RUN", sep="\t", file=f)

    sentence="rm "+it.dir+"/*vcf"
    #os.system(sentence)
    sentence="rm "+it.dir+"/*vcf.gz"
    #os.system(sentence)
    sentence="rm "+it.dir+"/*tbi"
    #os.system(sentence)
    sentence="rm -rf "+it.dir+"/smcpp/"
    #os.system(sentence)

def SMC_run4(i,j_name,p):
    #SMC_run(gen,it,r,p)
    #os.system(bash work.sh i json p)
    start_time=time.time()
    d=gy.get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")
    print("Main is:", __name__)

    if type_model=="StSi":
       r=gy.StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=gy.panmictic(d=d)
    else:
        print("Model incorrect")
        sys.exit()

    print(d["mtype"])
    print("The model is", d.get("mtype"))

    it=gy.Repetition_model(r,i,p) # parse the repetition
    gen=gy.FSC2_read_gen(it.gen_path)

    print("Starting smc++ run...",str(it.rep))
    vcf,inds=gy.FSC2_gen2VCF(gen,it,p)
    gy.SMC_vcf2smc(vcf,inds,it,r,p)
    finish_time_vcf=time.time()
    time_vcf=finish_time_vcf-start_time
    # Do for each sampled individual
    sample_dicts=[]

    ## FUNCTION TO RUN IN PARALLEL ###
    def SMC_estimate_sample_per_individual(dis):
        # Prepare to run each individual in a parallel process
        # This function should be created inside the main SMCPP function
        # it, r and i should be local variables
        print("Starting smcpp with individual %s" % dis)
        start_inf_time=time.time()
        print("Estimating demographic model in SMC++")
        model_json=gy.SMC_estimate(it,r,dis,s=i) # RUN SMCPP estimation
        #model_json=it.dir+"/smcpp/model.final.json"
        print("Arrange and plot results for SMC++")
        csv_smcpp=gy.SMC_arrange(model_json,it,dis,s=i) # GET THE CSV FROM SMCPP ESTIMATION
        # Copy to results
        finish_time_inference=time.time()
        inference_time=round(finish_time_inference-start_inf_time,2)
        with open("gyarados_time_check.log","a") as f:
            print(r.name,"SMCPP",str(inference_time),dis, sep="\t", file=f)
        sentence="cp "+csv_smcpp+" RESULTS/"+r.name+"/"
        print(sentence)
        os.system(sentence)

        print("Starting sampling for %s" % dis)
        # Sample vector
        this_sample_dicts=gy.SMC_sample(csv_smcpp) # SAMPLE THE VECTOR OF TIMES OUT OF THE CSV
        print("After sampling, the type is:", type(this_sample_dicts))
        print("For individual %s, we have the following keys:" % dis)
        return this_sample_dicts  # RETURN THE VECTOR OF SAMPLED TIMES
## end ###
    results=[]

    pool = multiprocessing.Pool(processes=len(inds)) # as many processes as individuals
    for dis in inds:
        results.append(pool.apply_async(SMC_estimate_sample_per_individual, args=(dis, )))
    pool.close()
    pool.join()
    sample_dicts = [p.get() for p in results]
    print("Now we have %d entries (individuals) in the dict list" % len(sample_dicts))
    # Call mean and write
    gy.SMC_mean_inferences(sample_dicts,r,it)
    

def SMC_run3(i, j_name, p):
    
    #SMC_run(gen,it,r,p)
    #os.system(bash work.sh i json p)
    start_time=time.time()
    d=gy.get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")
    print("Main is:", __name__)

    if type_model=="StSi":
       r=gy.StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=gy.panmictic(d=d)
    else:
        print("Model incorrect")
        sys.exit()

    print(d["mtype"])
    print("The model is", d.get("mtype"))

    it=gy.Repetition_model(r,i,p) # parse the repetition
    gen=gy.FSC2_read_gen(it.gen_path)

    print("Starting smc++ run...",str(it.rep))
    vcf,inds=gy.FSC2_gen2VCF(gen,it,p)
    gy.SMC_vcf2smc(vcf,inds,it,r,p)
    finish_time_vcf=time.time()
    time_vcf=finish_time_vcf-start_time
    # Do for each sampled individual
    sample_dicts=[]
    for dis in inds:
        print("Starting smcpp with individual %s" % dis)
        start_inf_time=time.time()
        print("Estimating demographic model in SMC++")
        model_json=gy.SMC_estimate(it,r,dis,s=i)
        #model_json=it.dir+"/smcpp/model.final.json"
        print("Arrange and plot results for SMC++")
        csv_smcpp=gy.SMC_arrange(model_json,it,dis,s=i)
        # Copy to results
        finish_time_inference=time.time()
        inference_time=round(finish_time_inference-start_inf_time,2)
        with open("gyarados_time_check.log","a") as f:
            print(r.name,"SMCPP",str(inference_time),dis, sep="\t", file=f)
        sentence="cp "+csv_smcpp+" RESULTS/"+r.name+"/"
        print(sentence)
        os.system(sentence)

        print("Starting sampling for %s" % dis)
        # Sample vector
        this_sample_dicts=gy.SMC_sample(csv_smcpp)
        print("After sampling, the type is:", type(this_sample_dicts))
        print("For individual %s, we have the following keys:" % dis)
        
        print(list(this_sample_dicts.keys()))
        sample_dicts.append(this_sample_dicts)
        print("Now we have %d entries (individuals) in the dict list" % len(sample_dicts))

        # Call mean and write
    gy.SMC_mean_inferences(sample_dicts,r,it)
        


def SMC_run2(i, j_name, p,change_start=False):
    
    #SMC_run(gen,it,r,p)
    #os.system(bash work.sh i json p)
    start_time=time.time()
    d=gy.get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")

    if type_model=="StSi":
       r=gy.StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=gy.panmictic(d=d)
    else:
        print("Model incorrect")
        sys.exit()

    print(d["mtype"])
    print("The model is", d.get("mtype"))

    it=gy.Repetition_model(r,i,p) # parse the repetition
    gen=gy.FSC2_read_gen(it.gen_path)

    print("Starting smc++ run...",str(it.rep))
    vcf,inds=gy.FSC2_gen2VCF(gen,it,p)
    gy.SMC_vcf2smc(vcf,inds,it,r,p)
    finish_time_vcf=time.time()
    time_vcf=finish_time_vcf-start_time
    # Do for each sampled individual
    for dis in inds:
        print("Starting smcpp with individual %s" % dis)
        if change_start:
            for i in [0,20]:
                print("Starting time is %d" % i)
                start_inf_time=time.time()
                print("Estimating demographic model in SMC++")
                model_json=gy.SMC_estimate(it,r,dis,s=i)
                #model_json=it.dir+"/smcpp/model.final.json"
                print("Arrange and plot results for SMC++")
                csv_smcpp=gy.SMC_arrange(model_json,it,dis,s=i)
                # Copy to results
                finish_time_inference=time.time()
                inference_time=round(finish_time_inference-start_inf_time,2)
                with open("gyarados_time_check.log","a") as f:
                    print(r.name,"SMCPP",str(inference_time),str(i)+"/"+dis, sep="\t", file=f)
                sentence="cp "+csv_smcpp+" RESULTS/"+r.name+"/"
                print(sentence)
                os.system(sentence)
        else:
            print("Estimating demographic model in SMC++")
            start_inf_time=time.time()
            model_json=gy.SMC_estimate(it,r,dis)
            #model_json=it.dir+"/smcpp/model.final.json"
            print("Arrange and plot results for SMC++")
            csv_smcpp=gy.SMC_arrange(model_json,it, dis)

            # Copy to results

            finish_time_inference=time.time()
            inference_time=round(finish_time_inference-start_inf_time,2)
            with open("gyarados_time_check.log","a") as f:
                print(r.name,"SMCPP",str(inference_time),str(0), sep="\t", file=f)
            sentence="cp "+csv_smcpp+" RESULTS/"+r.name+"/"
            print(sentence)
            os.system(sentence)
        finish_time=time.time()
        runtime=round(finish_time-start_time, 2)
        with open("gyarados_time_check.log","a") as f:
            print(r.name,"SMCPP_full",str(runtime),"NA", sep="\t", file=f)
    





##############################################################################################################
#####################                           SNIF FUNCTIONS                    ############################
##############################################################################################################

def SNIF_run(psmc,r,it):
    shutil.copy(psmc,"snif/") # copy psmc file to snif
    print(os.getcwd()) 
    os.chdir("snif") # change directory to snif
    print(os.getcwd())
    
    # call snif
    print("it.name is", it.name)
    psmc_cp=os.path.basename(psmc)
    chr=int(r.chr)
    size=list(map(int,r.sizes))
    name=psmc_cp



    print("psmc:"+psmc_cp)
    print("chr:"+str(chr))
    print(*list(map(str,size)),sep=" ")
    print(name)
    
    for c in [1,2,3,4,5]:
        for w in [1,0.75,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.1]:
            
                h_inference_parameters = InferenceParameters(
            data_source = psmc_cp,
            source_type = SourceType.PSMC,
            IICR_type = IICRType.Seq_sim,
            ms_reference_size = 500,
            ms_simulations = int(1e5),
            psmc_mutation_rate = 1e-8,
            psmc_number_of_sequences = chr,
            psmc_length_of_sequences = size,
            infer_scale = True,
            data_cutoff_bounds = (1e2, 2e5),
            data_time_intervals = 64,
            distance_function = ErrorFunction.ApproximatePDF,
            distance_parameter = w,
            distance_max_allowed = 7e3,
            distance_computation_interval = (1e2, 1e7),
            rounds_per_test_bounds = (1, 3),
            repetitions_per_test = 10,
            number_of_components = c,
            bounds_islands = (2, 50),
            bounds_migrations_rates = (0.005, 70),
            bounds_deme_sizes = (1,1),
            bounds_event_times = (100, 1.5e5),
            bounds_effective_size = (100, 5000)
    )

                h_settings = Settings(
                    static_library_location = './libs/libsnif.so',
                    custom_filename_tag = name,
                    output_directory = './'+name,
                    default_output_dirname = '_SNIF_results'
                )

                infer(inf = h_inference_parameters, settings = h_settings)
                

    ### moveinferention to original location

    shutil.copytree("./"+name, "../"+it.dir+"/snif_results")
    print("Results in:", "../"+it.dir+"/snif_results")
    os.chdir("..")

    return(it.dir+"/snif_results")


def SNIF_run3(psmc,r,it):
    '''
    Runs SNIF inference algorithm in a PSMC file calling run_snif.py script. Prepared to run in each repetition
    Needs to have snif.py and libs in the same folder as magikarp.py
    Inputs:
    - psmc: psmc file path
    - r: class model 
    - it: class repetition
    All the output files will be in a folder inside the repetition folder

    '''
    # sys.argv[1]= psmc file
    # sys.argv[2]= sizes
    # sys.argv[3]= short name
    # sys.argv[4]= snif rep directory
    shutil.copy(psmc,"snif/")
    snifdir="../"+it.dir+"/snif"
    parse_sizes=",".join([str(x) for x in r.sizes])
    print(os.getcwd()) 
    os.chdir("snif")
    print(os.getcwd())
    sentence="python run_snif.py "+it.name+" "+str(chr)+" "+parse_sizes
    print("Calling SNIF for",it.name)
    os.system(sentence)

def SNIF_run2(psmc, r, it):

    '''
    Runs SNIF inference algorithm in a PSMC file. Prepared to run in each repetition
    Needs to have snif.py and libs in the same folder as magikarp.py
    Inputs:
    - psmc: psmc file path
    - r: class model 
    - it: class repetition
    All the output files will be in a folder inside the repetition folder

    '''
    s=[int(x) for x in r.sizes]
    
    for c in [1,2,3,4,5]:
            for w in [1,0.75,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.1]:

                h_inference_parameters = InferenceParameters(
            data_source = psmc,
            source_type = SourceType.PSMC,
            IICR_type = IICRType.Seq_sim,
            ms_reference_size = 500,
            ms_simulations = int(1e5),
            psmc_mutation_rate = 1e-8,
            psmc_number_of_sequences = r.chr,
            psmc_length_of_sequences = s,
            infer_scale = True,
            data_cutoff_bounds = (1e2, 2e5),
            data_time_intervals = 64,
            distance_function = ErrorFunction.ApproximatePDF,
            distance_parameter = w,
            distance_max_allowed = 7e3,
            distance_computation_interval = (1e2, 1e7),
            rounds_per_test_bounds = (1, 3),
            repetitions_per_test = 1,
            number_of_components = c,
            bounds_islands = (2, 50),
            bounds_migrations_rates = (0.005, 20),
            bounds_deme_sizes = (1,1),
            bounds_event_times = (3e2, 1.5e5/11),
            bounds_effective_size = (100, 5000)
    )

                h_settings = Settings(
                    static_library_location = '/home/anieto/work/complete2/Snif/libs/libsnif.so',
                    custom_filename_tag = it.name,
                    output_directory = snifdir,
                    default_output_dirname = '_SNIF_results'
                )

                infer(inf = h_inference_parameters, settings = h_settings)

    print("SNIF if running for ", psmc)



###########################################################################################################
##################                            STATS FROM SFS                        #######################
###########################################################################################################

def normalize_sfs(sfs):
    '''
    Normalizes by Lapierre 2017 and returns a vector of normalized sfs values
    '''
    c=list(sfs)
    eta_2=[i/sum(c) for i in c] # divide by total of snps
    ind=len(eta_2)
    print(ind)
    lp=[]
    for i in range(1,ind):
        print(i)
        i2=i-1 # index for eta_2
        print(i2)
        this_count=eta_2[i2]
        new_count=this_count*i*((2*ind)-i)/(2*ind)
        lp.append(new_count)
    lp.append(eta_2[ind-1]*ind)
    return(lp)

def theta_pi(sfs):
    '''
    Calculate mean pairwise distance (theta_pi) from the folded SFS
    '''
    # Checked with lab libraries
    # Mean pairwise distance (theta_pi)
    v=[] # vector to do the summatories|
    n=len(sfs)*2 # as it is folded
    for i in range(1,len(sfs)+1): # last sfs not considered
        v.append(((n-i)*i)*(sfs[i-1])) # vector to do the summatories
    mpd=sum(v)/((n*(n-1))/2) # divide by combinatory
    return mpd

def segregating_sites(sfs):
    '''
    Get the number of segregating sites from a folded SFS. 
    '''
    return sum(sfs)  

### Needed to calculate Tajima's D

def calc_a1(sample_size):
    '''
    calculates a1 (needed for TD's)
    '''
    a=0
    for i in range(1,sample_size): # for ss=20 goes from 1 to 19
        a=a+(1/i)
    return a

def calc_a2(sample_size):
    '''
    calculates a2 (needed for TD's)
    '''
    a=0
    for i in range(1,sample_size): # for ss=20 goes from 1 to 19
        a=a+(1/i**2)
    return a

def calc_b2(sample_size):
    '''
    calculates b2 (needed for TD's)
    '''
    a=(2*((sample_size**2)+sample_size+3))/(9*sample_size*(sample_size-1))
    return a

def calc_b1(sample_size):
    '''
    calculates b1 (needed for TD's)
    ''' 
    a=(sample_size+1)/(3*(sample_size-1))
    return a

def calc_c1(sample_size):
    a=calc_b1(sample_size)-(1/calc_a1(sample_size))
    return a

def calc_c2(sample_size):
    a=calc_b2(sample_size)-((sample_size+2)/(calc_a1(sample_size)*sample_size))+(calc_a2(sample_size)/(calc_a1(sample_size)**2))
    return a

def calc_e1(sample_size):
    a=calc_c1(sample_size)/calc_a1(sample_size)
    return a

def calc_e2(sample_size):
    a=calc_c2(sample_size)/((calc_a1(sample_size)**2)+calc_a2(sample_size))
    return a

####

def theta_s(sfs):
    '''
    Calculates Waterson theta
    '''
    sample_size=2*len(sfs)
    S=segregating_sites(sfs)
    tw=S/calc_a1(sample_size)
    return tw


def TajimaD(sfs,sz):
    '''
    Calculates Tajima's D
    '''
    sizes=map(int,sz)
    total_size=sum(sizes)
    print("Total size: ",total_size)
    theta_P=gy.theta_pi(sfs)/total_size
    print("theta_P: ", theta_P)
    theta_S=float(gy.theta_s(sfs)/total_size)
    print("theta_S: ",theta_S)
    sample_size=2*len(sfs) # as it is folded
    S=segregating_sites(sfs)
    deno=(((calc_e1(sample_size)*S)+(calc_e2(sample_size)*S*(S-1)))**0.5)
    print("deno:",deno)
    TD=(gy.theta_pi(sfs)-gy.theta_s(sfs))/deno
    print("Tajima's D:",TD)
    return TD


def SFS_stats(sfs,tons,counts,it_fullname,sz,r,p):
    '''
    Creates a tsv with the results of Theta_P, Theta_S, S, Tajima's D, SFS count and normalized SFS
    Input:
    - sfs (array with the counts)
    - it (model iteration object)
    Out:
    - sst: pandas dataframe

    '''
    start_time=time.time()
    ###### DF for stats
    sizes=map(int,sz)
    total_size=sum(sizes)
    print("Total size is:", total_size)
    sst=pd.DataFrame(columns=["theta_P", "theta_S", "S", "TD"])
    ss=pd.DataFrame({"theta_P":gy.theta_pi(sfs)/total_size, "theta_S":gy.theta_s(sfs)/total_size, "S":gy.segregating_sites(sfs), "TD":gy.TajimaD(sfs,sz)}, index=[0])
    sst=pd.concat([sst,ss])
    sst.to_csv(it_fullname+"_deme_"+str(p)+"_stats.tsv", index=False, sep="\t")
    print("Stats from SFS are in",it_fullname+"_deme_"+str(p)+"_stats.tsv")

    # Copy to results
    sentence="cp "+it_fullname+"_deme_"+str(p)+"_stats.tsv RESULTS/"+r.name+"/"
    print(sentence)
    os.system(sentence)

    ######## DF for sfs

    res = {tons[i]: counts[i] for i in range(len(tons))}
    columns=list(tons)
    columns.append("Type")
    fsfs_df = pd.DataFrame(columns=columns, index=[0,1])
    fill_counts=list(counts)
    fill_counts.append("Counts")
    fill_normalized=gy.normalize_sfs(counts)
    fill_normalized.append("Normalized")
    fsfs_df.iloc[0]=fill_counts
    fsfs_df.iloc[1]=fill_normalized
    fsfs_df.to_csv(it_fullname+"_deme_"+str(p)+"_sfs.tsv", index=False, sep="\t")

    # Copy to results
    sentence="cp "+it_fullname+"_deme_"+str(p)+"_sfs.tsv RESULTS/"+r.name+"/"
    print(sentence)
    os.system(sentence)


    finish_time=time.time()
    
    runtime=round(finish_time-start_time,2)
    
    with open("gyarados_time_check.log","a") as f:
        print(r.name,"SFS_STATS",str(runtime),"NA", sep="\t", file=f)
    return sst

###########################################################################################################
###############################        MS FUNCTIONS               #########################################
###########################################################################################################


def MS_to_gen_table(ms_file,ip):
    '''
    This function converts the ms output obtained with the flags -T -L to a gen table in a pandas dataframe. 
    This will allow the user to apply any SFS and FSC related functions. 
    It assumes that the simulation has been perfomed only with samples from the population of interest (only one deme)
    Input:
    - ms_file: path to the ms file
    Output;
    - ms_gen_table: a pandas dataframe
    '''

    f = open(ms_file, "r")
    o=f.read()
    ms_command=o.split("//")[0] # the ms command
    print("ms command is: ",ms_command)
    chromosomes=o.split("//")[1:]  # data for each chromosome
    for i in range(len(chromosomes)):  # loop over each chromosome
        c=chromosomes[i] # all info in a chromosome
        c=c[c.find("positions:"):] # eliminate all the tree
        # To get the positions:
        # 1. split c by \n and take the first one (the rest are sequences)
        # 2. Split by space " " and eliminate the first one (its "positions:")
        # 3. Eliminate the last one (its a blank space)
        # 4. Map to float
        positions=list(map(float,((c.split("\n")[0]).split(" ")[1:])[:-1]))

        # To get the binary sequences (haplotypes)
        # 1. Split c by \n and take from the second one. Eliminate last one (blank)
        # 2. Map to create a list in which each position is an element
        sequences=c.split("\n")[1:][:-1]
        sequences=list(map(list,sequences))
        if len(sequences)%2!=0:
            sequences=sequences[:-1]

        # Repeat chromosome as many times as positions are
        chr_list=[i+1]*len(positions)

        # Arrange in a dataframe
        columns=["Chrom","Pos", "Anc_all","Der_all"]
        this_chr=pd.DataFrame()
        this_chr["Chrom"]=chr_list
        this_chr["Pos"]=positions
        this_chr["Anc_all"]="Z"
        this_chr["Der_all"]="Q"

        # Get genotypes
        ind=0
        p=0
        print(len(sequences))
        ind_index=range(1,len(sequences)//2+1)
        for s in range(0,len(sequences),2):
            first_h=np.array(list(map(int,sequences[s])))
            second_h=np.array(list(map(int,sequences[s+1])))
            genotype=first_h+second_h  # note: if sample size is 2, no homozygous positions are included
            if s%ip==0:
                p+=1
            print(ind_index[ind])
            ind_name="G_"+str(p)+"_"+str(ind_index[ind])  # I suppose i am only sampling one population at a time (not planning to do more)
            this_chr[ind_name]=genotype
            ind+=1
        if i==0:
            ms_gen_table=this_chr
        else:
            ms_gen_table=pd.concat([ms_gen_table.reset_index(drop=True),this_chr.reset_index(drop=True)])
    f.close()

    return ms_gen_table

###########################################################################################################
#################################              IICRESTIMATOR                 ##############################
###########################################################################################################


def compute_t_vector(start, end, number_of_values, vector_type):
    '''
    From IICREstimator
    '''

    if vector_type == 'linear':
        x_vector = np.linspace(start, end, number_of_values)
    elif vector_type == 'log':
        n = number_of_values
        x_vector = [0.1*(np.exp(i * np.log(1+10*end)/n)-1)
                    for i in range(n+1)]
        x_vector[0] = x_vector[0]+start
    else:
        # For the moment, the default output is a linspace distribution
        x_vector = np.linspace(start, end, number_of_values)
    return np.array(x_vector)


def is_array_like(obj, string_is_array = False, tuple_is_array = True):
    '''
    From IICREstimator
    '''
    result = hasattr(obj, "__len__") and hasattr(obj, '__getitem__') 
    if result and not string_is_array and isinstance(obj, str):
        result = False
    if result and not tuple_is_array and isinstance(obj, tuple):
        result = False
    return result



def compute_IICR_n_islands(t, params):
    '''
    From IICREstimator
    '''
    n = params["n"]
    M = params["M"]
    s = params["sampling_same_island"]

    if(is_array_like(n)):
        raise TypeError("Having multiple number of islands is not yet supported!")

    if(is_array_like(M)):
        tau = params["tau"]
        c = params["size"]

        if(not (is_array_like(tau) or is_array_like(c))):
            raise TypeError("Both 'tau' and 'size' must be array types!")
        
        if(len(M) != len(tau)):
            raise ValueError("Vectors 'M' and 'tau' must have the same length!")
        
        if(tau[0] != 0):
            raise ValueError("The time of the first event must be 0!")

        if(len(M) != len(c)):
            raise ValueError("Vectors 'M' and 'size' must have the same length!")
        print("Module from IICREstimator missing")
        #return compute_piecewise_stationary_IICR_n_islands(n, M, tau, c, t, s)
    
    return compute_stationary_IICR_n_islands(n, M, t, s)


def compute_stationary_IICR_n_islands(n, M, t, s=True):

    '''
    From IICREstimator
    '''
    # This method evaluates the lambda function in a vector
    # of time values t.
    # If 's' is True we are in the case when two individuals where
    # sampled from the same island. If 's' is false, then the two
    # individuals where sampled from different islands.

    # Computing constants
    gamma = np.true_divide(M, n-1)
    delta = (1+n*gamma)**2 - 4*gamma
    alpha = 0.5*(1+n*gamma + np.sqrt(delta))
    beta =  0.5*(1+n*gamma - np.sqrt(delta))

    # Now we evaluate
    x_vector = t
    if s:
        numerator = (1-beta)*np.exp(-alpha*x_vector) + (alpha-1)*np.exp(-beta*x_vector)
        denominator = (alpha-gamma)*np.exp(-alpha*x_vector) + (gamma-beta)*np.exp(-beta*x_vector)
    else:
        numerator = beta*np.exp(-alpha*(x_vector)) - alpha*np.exp(-beta*(x_vector))
        denominator = gamma * (np.exp(-alpha*(x_vector)) - np.exp(-beta*(x_vector)))

    lambda_t = np.true_divide(numerator, denominator)

    return lambda_t



############################################################################################################
############################                    IICR CURVE                         #########################
############################################################################################################

def IICR_fullrun(r,p=1, fn=10000000, fs=1): ############ CHANGE NUMBER OF SIMULATED FRAGMENTS IICR
    '''
    Run FSC2 to get the distribution of T2 for an r model. Calculate the IICR.
    - Input: r magikarp model
    - Output: IICR dataframe (that is also saved, along with a plot)
    '''
    #fn=1000000000 # Excepcional sim with a lot of lineages
    #fs=100 # same 
    print("Start IICR with fn " + str(fn)+" and fs "+str(fs))
    start_time=time.time()
    par_name=gy.IICR_create_par(r,p, fn, fs)  ##population is specified
    # IF SST or FREE
    # par_name=r.name+"_fn"+str(fn)+"_fs"+str(fs)+"_deme_"+str(p)+"_IICR" 
    # IF StSi or panmictic
    #par_name=r.name+"_fn"+str(fn)+"_fs"+str(fs)+"_IICR"
    mrca_folder=gy.IICR_FSC2_run_tMRCA(r, par_name)
    fsc_file=mrca_folder+par_name+"_mrca.txt"
    print("IICR file is " + str(fsc_file))
    sentence="mv -f "+par_name+" "+r.name 
    
    os.system(sentence)

    tmrca=gy.IICR_FSC2_read_tMRCA(fsc_file,r.N[p-1]) #### POPULATION SPECIFIED
    sentence="mkdir -p "+r.name+"/"+r.name+"_IICR" ### if it doesnt exist
    os.system(sentence)
    sentence= "cp "+fsc_file+" "+r.name+"/"+r.name+"_IICR/"
    os.system(sentence)

    IICR_lin_df=gy.IICR_calculate(tmrca,r,mrca_folder,type="linear", island=p, par_name=par_name)
    IICR_log_df=gy.IICR_calculate(tmrca,r,mrca_folder,type="log", island=p, par_name=par_name)
    
    
    
    # Remove mrca
    sentence="rm "+r.name+"/"+r.name+"_IICR/*mrca.txt"
    os.system(sentence)
    
    finish_time=time.time()
    
    runtime=round(finish_time-start_time,2)

    with open("gyarados_time_check.log","a") as f:
        print(r.name,"IICR",str(runtime),str(fn)+"/"+str(fs), sep="\t", file=f)

    return(IICR_lin_df,IICR_log_df)


def IICR_create_par(r,pop=1, fn=10000000, fs=1000):
    # Build the N (deme sizes)
    # 2 in the first one, 0 in the rest 

    if r.mtype == "StSi":
        mrca_N=["2\n"]+["0\n"]*(r.islands-1) # ASSUMES POPULATION IS 1
    if r.mtype == "panmictic":
        mrca_N=["2\n"]
    if r.mtype == "Free":
        mrca_N=["0\n"]*(r.islands)
        mrca_N[pop-1]="2\n"
        #mrca_N="\n".join(list_N)
    if r.mtype == "SST": 
        # Create array full of 0 
        mrca_N=["0\n"]*(r.islands)
        # Change the value from the population per 2
        mrca_N[pop-1]="2\n"
        
        #print("I will append the following:")
        #print(mrca_N)
    


    # Build the chromosome sizes
    # We will sample 100Millions chunks of 1000
    # r=0, mu=mu


    # read the original par file
    o=open(r.par)
    p=o.readlines()

    print("Demographic events header should have the word 'event' in par file to continue")

    bool_headers = [line.startswith("//") for line in p]
    headers_index= list(filter(lambda i: bool_headers[i], range(len(bool_headers))))
    headers = [p[i] for i in headers_index]
    ev_header=[header for header in headers if "event" in header][0]
    str_chromosomes="//Number of independent loci [chromosome] (Number of sequences of 300 bp per gamete)\n%d 0\n//Chromosome structure 1 begins with number of loci\n1\n//per block: data type, number of loci, per generation recombination and mutation rates and optional parameters\nDNA  %d 0 " % (fn,fs)
    str_chromosomes=str_chromosomes+str(r.mu)

    # Substitute the deme sizes
    start=headers_index[2] #start in sample sizes header
    new_N_index=0
    for n in range(1,len(mrca_N)+1):
        subs=start+n
        print("Real population size is", p[subs])
        p[subs]=mrca_N[new_N_index]
        print("New population size is", p[subs])
        new_N_index=new_N_index+1

    #print("For the moment, my par file is like this:", p)
    # Eliminate the chromosomes part, add the new one
    erase_from=headers_index[headers.index(ev_header)+1]#included
    p=p[:erase_from]
    #print("Now my par file is like this:", p)
    p.append(str_chromosomes)

    # Create new par file
    mrca_par="".join(p)
    if r.mtype=="StSi" or r.mtype=="panmictic":
        par_name=r.name+"_fn"+str(fn)+"_fs"+str(fs)+"_IICR"
    elif r.mtype=="SST" or r.mtype=="Free": ############# CONSIDERING FREE MODELS BC SAMPLING CAN BE DIFFERENT TOO
        par_name=r.name+"_fn"+str(fn)+"_fs"+str(fs)+"_deme_"+str(pop)+"_IICR"

    mrca_par_path=r.name+"/"+par_name+".par"
    #mrca_par_path="new_mrca.par"
    print("Par too long?????????")
    print(mrca_par_path)
    with open(mrca_par_path,"w") as npar:
        npar.write(mrca_par)

    return par_name


def IICR_FSC2_run_tMRCA(r, par_name):
    print("Running IICR_FSC2_run_tMRCA with par file: " + par_name)
    par_path=r.name+"/"+par_name+".par"
    sentence="fsc -i "+par_path+" -n1 -I -x -k 100000000 -q --recordMRCA"
    #sentence="./fsc27093 -i "+par_path+" -n1 -I -x -k 100000000 -q --recordMRCA" ########MNHN CLUSTER
    os.system(sentence)
    mrca_folder=r.name+"/"+par_name+"/"
    return mrca_folder 


def IICR_FSC2_read_tMRCA(fsc_file,N):
    mrca_df=pd.read_csv(fsc_file,sep="\t", skiprows=4)
    print("Considering simmulated population size: ", N)
    N=int(N)
    mrca_df=mrca_df[mrca_df["Time_(gen)"] != "Time_(gen)"]
    tmrca_fsc=list(mrca_df["Time_(gen)"])
    tmrca_fsc=list(map(float,tmrca_fsc))
    tmrca_fsc.sort()
    tmrca_fsc=np.array(tmrca_fsc)/(2*N)
    return tmrca_fsc



def IICR_calculate(tmrca,r,mrca_folder,island,par_name,type="linear") :

    
    tmrca = 2*np.array(tmrca)
    Nd=r.N[island-1]/2
    print("Number of diploid individuals is:", Nd)
    
    if type=="linear":
        x=np.array(lin_vector)/(2*Nd) 
        this_path_IICR_df=mrca_folder+par_name+"_lin.tsv"
        if r.mtype=="StSi" or r.mtype=="panmictic":
            path_IICR_df=r.iicr_lin_path ################ THIS
        if r.mtype=="SST" or r.mtype=="Free": 
            path_IICR_df=r.iicr_lin_path[island]
    if type=="log":
        x=np.array(log_vector)/(2*Nd)
        this_path_IICR_df=mrca_folder+par_name+"_log.tsv"
        if r.mtype=="StSi" or r.mtype=="panmictic":
            path_IICR_df=r.iicr_log_path ############### THIS
        if r.mtype=="SST" or r.mtype=="Free": 
            path_IICR_df=r.iicr_log_path[island]    
    x[0] = 0 # The first element of actual_x_vector should be 0
    
    half_dx = np.true_divide(x[1:]-x[:-1], 2)
    # Computes the cumulative distribution and the distribution
    x_diff = x[:-1] + half_dx
    x_diff = np.array([0] + list(x_diff) + [x[-1]+half_dx[-1]])
    
    hist = np.histogram(tmrca, bins = x)[0]
    hist_diff = np.histogram(tmrca, bins = x_diff)[0]
    
    F_x = hist.cumsum()
    F_x = np.array([0]+list(F_x))
    
    # now we compute the pdf (the derivative of the cdf)
    dy= hist_diff
    dx = x_diff[1:] - x_diff[:-1]
    f_x = np.true_divide(dy, dx)

    lmd = np.true_divide(len(tmrca)-F_x, f_x)
    generations=2 * Nd *x
    scaled_lmd=Nd * lmd
    IICR_df=pd.DataFrame({"generations":generations, "lmd":lmd,"scaled_lmd":scaled_lmd,"x":x,"F_x":F_x,"f_x":f_x})
    #IICR_df["M"]=M
    IICR_df.to_csv(this_path_IICR_df, sep="\t", header=True, index=False, compression="gzip")
    
    # copy to original path
    sentence="cp "+this_path_IICR_df+" "+path_IICR_df
    os.system(sentence)
    # copy to results folder
    sentence="cp "+path_IICR_df+" RESULTS/"+r.name+"/"
    os.system(sentence)

    ##### Theoretical if nisland
    if r.mtype == "StSi":
        M=2*r.N[island]*(r.islands-1)*r.mig[0]
        path_theor_IICR_df=mrca_folder+r.name+"_theor_IICR"

        
        if type=="linear":
            t_k=lin_vector
            path_to_save_theor=path_theor_IICR_df+"_lin.tsv"
        if type=="log":
            t_k=log_vector
            path_to_save_theor=path_theor_IICR_df+"_log.tsv"
        
        t_k=np.true_divide(t_k, r.N[island])
        params={"n":r.islands, "M":M, "sampling_same_island":1}
        theor_IICR=gy.compute_IICR_n_islands(t_k, params)
        IICR_theor_df=pd.DataFrame({"theor_IICR":r.N[island]*theor_IICR,"theor_time":2*r.N[island]*t_k,"M":M})
        IICR_theor_df.to_csv(path_to_save_theor, sep="\t", header=True, index=False)
        sentence="cp "+path_to_save_theor+" RESULTS/"+r.name+"/"
        os.system(sentence)
        

    return(IICR_df)


def IICR_calculate3(tmrca,r,mrca_folder,island=0):
    # This method computes the empirical distribution given the
    # observations.
    # The functions are evaluated in the x_vector parameter
    # by default x_vector is computed as a function of the data
    # by default the differences 'dx' are a vector 

    tmrca=2*tmrca

    path_IICR_df=mrca_folder+r.name+"_IICR"
    
    M=2*r.N[island]*(r.islands-1)*r.mig
    x=gy.compute_t_vector(0,max(tmrca)+10,500,"log")
        
    x[0]=0

    half_dx = np.true_divide(x[1:]-x[:-1], 2) # get increment
    # Computes the cumulative distribution and the distribution
    x_inc = x[:-1] + half_dx
    x_inc = np.array([0] + list(x_inc) + 
                                [x[-1]+half_dx[-1]])

    hist = np.histogram(tmrca, bins = x)[0]
    hist_inc = np.histogram(tmrca, bins = x_inc)[0]

    cdf= hist.cumsum()
    cdf= np.array([0]+list(cdf)) # add 0

    # now we compute the pdf (the derivative of the cdf)
    dy = hist_inc
    dx = x_inc[1:] - x_inc[:-1]
    pdf_t = np.true_divide(dy, dx)


    f_x=np.true_divide(np.array(pdf_t), sum(np.array(pdf_t)))
    F_x = np.array(cdf) 


    lmd = np.true_divide(len(tmrca)-F_x, f_x)

    IICR_df=pd.DataFrame({"f_x":pdf_t,"F_x":cdf, "x":x, "lmd":lmd})
    IICR_df["p(T2>t)"]=1-IICR_df["F_x"]
    IICR_df["t"]=2*r.N[island]*x
    IICR_df["scaled_lmd"]=r.N[island]*lmd
    IICR_df["x"]=x
    IICR_df["M"]=M


    ##### Theoretical if nisland
    if r.mtype == "StSi":
        path_theor_IICR_df=mrca_folder+r.name+"_theor_IICR"
        T_max = np.log10(100000)
        t_k = np.logspace(1, T_max, 1000)
        t_k = np.true_divide(t_k, r.N[island])
        params={"n":r.islands, "M":M, "sampling_same_island":1}
        theor_IICR=gy.compute_IICR_n_islands(t_k, params)
        IICR_theor_df=pd.DataFrame({"theor_IICR":r.N[island]*theor_IICR,"theor_time":2*r.N[island]*t_k,"M":M})
        IICR_theor_df.to_csv(path_theor_IICR_df+".tsv", sep="\t", header=True, index=False)
        
    

    
    IICR_df.to_csv(path_IICR_df+".tsv", sep="\t", header=True, index=False)

    # Add the theoretical curve if asked to
    
    
    # plot the IICR
    fig, ax = plt.subplots() # open plot
    ax.plot(IICR_df.t,IICR_df.scaled_lmd,label=str(M))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    plt.xlabel("Time (in gen/2N)")
    plt.ylabel("IICR")
    plt.suptitle("IICR in "+r.name)
    plt.savefig(path_IICR_df+".png")

    return IICR_df


def IICR_calculate2(tmrca,r,mrca_folder):
    '''
    Calculates the IICR using the tmrca file from fsc2
    Input: 
    - tmrca=*mrca.txt file from fsc2 usin --recordMRCA
    - r = magikarp model 
    Out:
    IICR dataframe
    '''
    path_IICR_df=mrca_folder+r.name+"_IICR"
    x=gy.compute_t_vector(0,max(tmrca)+10,500,"log")
    actual_x_vector=np.array(x)
    counts = np.histogram(tmrca, bins = actual_x_vector)[0]
    cdf_x = counts.cumsum() # counts of coalescent events in cummulative
    cdf_x = cdf_x/sum(counts)
    cdf_x = np.array([0]+list(cdf_x)) # just add the 0
    cdf_vals=cdf_x


    # Alternatively, you can use the inverse CDF to calculate the PDF
    # This assumes that the CDF is a step function
    inv_cdf_vals = np.zeros(len(x))
    inv_cdf_vals[1:] = np.diff(cdf_vals) / np.diff(x)
    pdf_vals = inv_cdf_vals / np.sum(inv_cdf_vals)

    # Verify that the PDF integrates to 1
    assert np.isclose(np.sum(pdf_vals), 1.0)

    # create dataframe
    IICR_df=pd.DataFrame({"f_x":pdf_vals,"F_x":cdf_vals, "t":x})
    IICR_df["p(T2>t)"]=1-IICR_df["F_x"]
    IICR_df["lmd"]=IICR_df["p(T2>t)"]/IICR_df["f_x"]
    IICR_df["1-lmd"]=IICR_df["F_x"]/IICR_df["f_x"]
    IICR_df.to_csv(path_IICR_df+".tsv", sep="\t", header=True)

    # plot the IICR
    fig, ax = plt.subplots() # open plot
    ax.plot(IICR_df.t,IICR_df.lmd,label=str(r.mig))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    plt.xlabel("Time (in gen/2N)")
    plt.ylabel("IICR")
    plt.suptitle("IICR in "+r.name)
    plt.savefig(path_IICR_df+".png")

    return IICR_df


###########################################################################################################
################                            GA_BRAKER                                ######################                          
###########################################################################################################





############################################################################################################
#################         FUNCTIONS PIERRE: GOES TO ALREADY RUN FSC TO PSMC         ########################
############################################################################################################



def pierre_FSC2PSMC(name,sizes,p):

    '''
    Goes from .gen table from FSC2 to psmc output
    Runs psmc
    Input:
    - name: directory/name of .gen table (without extension .gen)
    - sizes: list of sizes for chromosomes
    - p: population you sample from (deme)
    '''
    # prints
    print(name)
    print(*sizes)
    print(p)
    # 1. Consensus fa file
    consensus=gy.pierre_PSMC_consensus_fa(name,sizes,p)
    # 2. Consensus fastq file
    consensus_fq=gy.pierre_PSMC_convert_fastq(consensus,name)
    # 3. psmcfa file (output psmc)
    psmcfa=gy.pierre_PSMC_convert_input(consensus_fq,name)
    # 4. Run psmc
    psmc=gy.pierre_PSMC_run(psmcfa,name)

    return psmc


def pierre_PSMC_consensus_fa(name,sizes,p):
    '''
    - name: dir/name (without extension)
    - sizes: list of sizes for each chromosome
    - p: population you sample from (deme)
    Outs the consensus.fa file for 1 individual. Intermediate file for PSMC
    

    '''
    gen_path=name+".gen"
    gen=gy.FSC2_read_gen(gen_path)
    co=[]
    consensus=name+".consensus.fa"

    for c in range(len(sizes)):
        chr=c+1
        size=int(sizes[c])
    # 1.1. Create dum fa
       
        print("Creating dum fa")
        tmp_fa=name+"_"+str(chr)+".fa"# temporary fa for chromosome
            #line=("A"*60+"\n") 
        #if it.rep == 1:
        with open(tmp_fa,"w") as f:
            f.write(">"+str(chr)+"\n")
            f.write("A"*size+"\n")
            print(tmp_fa, "has been created")
    # 1.2. Change the heterozygote positions
        this_chr=gen[gen["Chrom"]==chr]
        if int(p):
            # Sample the individual we want (first from the designed population)
            this_chr=gy.FSC2_gen_filter_1i(this_chr,p)
        consensus_tmp=name+"_"+str(chr)+".consensus.fatmp" # we edit this
        shutil.copyfile(tmp_fa, consensus_tmp) # copy it in its corresponding path
        f=os.open(consensus_tmp,os.O_RDWR) # open consensus.fa file
        m=mmap.mmap(f,0) # map the file. m is an mmap object

        # m is like a list. in each position are the characters
        # of my fa file in binary (includding header)
        # Each position in .gen is +2 in .fa index
        positions=list(this_chr.Pos)
        print(len(positions))
        lastpos=positions[-1]
        # print(*positions) 
        for snp in positions:
            # +2 and +3 to get the end (needs to be like this)
            init=snp+2
            end=snp+3
            m[init:end]=b"S" # b=binary
        #for pos in list(this_chr.Pos):
            # +2 and +3 to get the end (needs to be like this)
        #   if m[snp+2:snp+3]!=b"S":
        #       print(snp) # b=binary
        co.append(this_chr)
        os.close(f) # close
        print(consensus_tmp, "has been created")

    # Paste them 
    # Past all the consensus.fatmp in a single consensus fa
    sentence="cat "+name+"_*"+".consensus.fatmp | fold -b60 > "+consensus  
    print(sentence)
    os.system(sentence)
    
    return consensus

def pierre_PSMC_convert_fastq(consensus,name):
    '''
    Convert the consensus .fa into a fastq.gz
    '''
    consensus_fq=name+".consensus.fq.gz"
    sentence="cat "+consensus+" | seqtk seq -F 'I' | gzip > "+consensus_fq
    os.system(sentence)
    print(consensus_fq+" created")

    return consensus_fq

def pierre_PSMC_convert_input(consensus_fq,name):
    '''
    Get proper psmc input
    '''
    psmcfa=name+".consensus.psmcfa"
    sentence="fq2psmcfa -q0 "+consensus_fq+" > "+psmcfa
    os.system(sentence)
    print(psmcfa+" created")
    return psmcfa

    
def pierre_PSMC_run(psmcfa,name):

    sentence='psmc -t15 -r1 -p "4+25*2+4+6" -o '+name+'.psmc '+psmcfa
    os.system(sentence)
    psmc=name+".psmc"
    print("PSMC has been run succesfully in ", psmc)
