import os
import mmap
import shutil
import inspect
import itertools
import json
import math
import random as ra
import re
import string
import sys
import time

import numpy as np
import pandas as pd

import ditto as gy_ditto

TOOLS = {
    "fastsimcoal2": "fsc28",
    "psmc": "psmc",
    "fq2psmcfa": "fq2psmcfa",
    "seqtk": "seqtk",
    "smcpp": "smc++",
    "bgzip": "bgzip",
    "tabix": "tabix",
}

def configure_tools(tools):
    TOOLS.update(tools)

def run_external(command):
    print(command)
    if os.system(command) != 0:
        raise RuntimeError("External command failed: " + command)

def write_vcf(tsv_file, vcf_file):
    header_file = os.path.join(os.path.dirname(__file__), "header.txt")
    with open(vcf_file, "w") as output, open(header_file) as header, open(tsv_file) as table:
        shutil.copyfileobj(header, output)
        shutil.copyfileobj(table, output)

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

lin_vector = np.linspace(50, 100000, num=2000)
log_vector = logspace(50001, 50, 200)
log_vector_sst = log_vector
log_vector_in = log_vector

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
            d=get_from_json(mod_name+"/"+mod_name+".json")

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
                self.json=get_json(self)
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
            if type(simn_N)==int:
                self.N=[simn_N]*simn_n_islands
            elif len(simn_N.split(","))==simn_n_islands:
                self.N=simn_N.split(",")
            else:
                raise ValueError("ERROR: Population sizes not matching the number of islands.\nIf all islands are the same, set only one size")

            if type(simn_samples)==int:
                self.samples=[0]*simn_n_islands
                # I only need two islands to calculate the Fst
                self.samples[0]=simn_samples
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

def get_json(model):
    '''
    Create a json with the model attributes
    '''

    atts = [ p for p in inspect.getmembers(model) if not(p[0].startswith('__'))]
    atts=dict(atts)
    sentence= "mkdir -p "+model.name
    os.system(sentence)

    if model.mtype !="Free":
        if model.mtype != "panmictic":
            atts["mig_matrix"]=[mim.tolist() for mim in atts["mig_matrix"]]
        #atts["mig_matrix"]=atts["mig_matrix"].tolist()
    j_name=model.name+"/"+model.name+".json"
    with open(j_name, 'w') as fp:
        json.dump(atts, fp)
    return j_name

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

def GYARADOS_PAR(type,L=None, N=None,sizes=None,sample=None,niter=None,chr=None,nislands=None,mig=None,p=1,mu=1e-8,rho=1e-8, evs=[], gr=0, par=None,M=None, Nt=None, mode=DEFAULT_GYARADOS_MODE, psmc_patterns=None, psmc_s=DEFAULT_PSMC_S, ditto=False, backend="local", iicr_replicates=10000000):

    """
    Run an island-model workflow sequentially on the current machine.
    """
    if backend != "local":
        raise ValueError(
            "The portable runner supports backend='local' only. "
            "The former SLURM implementation is preserved in legacy/."
        )
    print("Welcome to gyarados! Version 16.11.2023")
    run_modes = parse_run_modes(mode)
    psmc_patterns = ",".join(parse_psmc_patterns(psmc_patterns))
    psmc_s = str(psmc_s)
    ditto = parse_bool(ditto)
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
        r=Free(par)

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
            r=StSi(simn_n_islands=int(nislands),simn_N=int(N),simn_samples=int(sample),simn_mig=float(mig),simn_sizes=sz, simn_chr=chr, simn_mu=float(mu), simn_events=events, simn_gr=gr,simn_M=float(M), simn_Nt=Nt) # model object
        if int(type)==2:
            print("panmictic model")
            r=panmictic(pan_N=int(N), pan_samples=int(sample), pan_sizes=sz, pan_chr=chr, pan_mu=float(mu),pan_rho=rho, pan_events=events, pan_gr=gr)
            p=1

        if int(type)==4:
            print("2D Stepping stone model")
            print("Two sampled demes implemented...")
            sampled_demes=list(map(int,p.split(",")))
            if len(sampled_demes)==2:
                print("sampled demes are:",sampled_demes[0], "and", sampled_demes[1])
                r=SST(sst_L=int(L),sst_N=int(N), sst_samples=int(sample), sst_sampled_demes=sampled_demes,  sst_sizes=sz,sst_mig=mig, sst_chr=chr,sst_mu=float(mu), sst_M=float(M), sst_rho=rho, sst_gr=gr, sst_events=events)################
            else:
                print("One single sampled deme specified")
                print("Sampled deme is", int(sampled_demes[0]))
                r=SST(sst_L=int(L),sst_N=int(N), sst_samples=int(sample), sst_sampled_demes=int(sampled_demes[0]),  sst_sizes=sz,sst_mig=mig, sst_chr=chr,sst_mu=float(mu), sst_M=float(M), sst_rho=rho, sst_gr=gr, sst_events=events)

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
    dumfa="psmc" in run_modes
    if dumfa:
        for c in range(len(list(r.sizes))):
            chr=c+1
            size=int(r.sizes[c])
            print("Creating dum fa")
            tmp_fa=r.name+"/"+r.name+"_"+str(chr)+".fa"# temporary fa for chromosome
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
                IICR_fullrun(r,int(p),fn=int(iicr_replicates))
                sentence="rm -r "+r.name+"/*fn*fs*"
                os.system(sentence)
        elif int(type)==3: ### FREE p
            print("Calculating IICR with", p)
            IICR_fullrun(r,int(p),fn=int(iicr_replicates))

        else:
            print("Calculating IICR")
            IICR_fullrun(r,int(p),fn=int(iicr_replicates))
        sentence="rm -r "+r.name+"/*fn*fs*"
        os.system(sentence)
    else:
        print("Skipping IICR because mode is", mode)

    if ditto:
        gy_ditto.run_ditto_for_model(
            r,
            p,
            niter=niter,
            mode=mode,
            psmc_patterns=psmc_patterns,
            psmc_s=psmc_s,
            run_gyarados=lambda **kwargs: GYARADOS_PAR(
                backend=backend,
                iicr_replicates=iicr_replicates,
                **kwargs
            )
        )

    worker_modes = run_modes.intersection({"simulate", "stats", "psmc", "smcpp", "transition_matrix"})
    if worker_modes:
        for i in range(1,int(niter)+1):
            GYARADOS_WORK(
                i,
                j_name,
                p,
                mode=mode,
                psmc_s=psmc_s,
                psmc_patterns=psmc_patterns,
            )
    else:
        print("No repetition jobs submitted because mode is", mode)

    return r

def GYARADOS_WORK(i, j_name, p, mode=DEFAULT_GYARADOS_MODE, psmc_s=DEFAULT_PSMC_S, psmc_patterns=None, backend="local"):
    '''
    Worker for n_islands. Runs all analyses for one repetition
    - i: repetition id
    - j_name: json with model data stored
    - p: population we are interested in
    '''
    if backend != "local":
        raise ValueError("The portable worker supports backend='local' only.")
    run_modes = parse_run_modes(mode)
    psmc_patterns = parse_psmc_patterns(psmc_patterns)
    print("Worker modes:", ",".join(sorted(run_modes)) if run_modes else "none")
    print("Worker PSMC -p vectors:", ",".join(psmc_patterns))
    print("Worker PSMC -s:", psmc_s)

    d=get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")
    if type_model=="StSi":
       r=StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=panmictic(d=d)
    elif type_model=="SST":
        r=SST(d=d) # parse the model
    elif type_model=="Free":
        r=Free(d=d) # parse the model
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

        it=Repetition_model(r,i,deme) # parse the repetition
        print("Starting repetition", it.name)
        print("WE start with deme", deme)
        # 1. Copy par file adding the ID of the repetition
        needs_sequence_simulation = bool(run_modes.intersection({"simulate", "transition_matrix"}))
        if needs_sequence_simulation and deme==sampling_demes[0]:
            shutil.copyfile(r.par,it.par)
            print("Running fsc2...",i)
            FSC2_run_with_mrca(it.par)

            basename=(it.par.split("/")[0]).split(".par")[0]
            os.system("mv -f "+it.name+" "+r.name)
            mrca_file=it.par.split(".par")[0]+"/"+basename+"_rep1_mrca.txt"
            os.system("gzip "+mrca_file)
            os.system("gzip "+it.old_gen_path)
            os.system("cp "+it.gen_path+" RESULTS/"+r.name+"/")
        else:
            print("Skipping FSC2 simulation for this deme or mode.")

        transition_matrix_only = (
            "transition_matrix" in run_modes
            and not run_modes.intersection({"stats", "psmc", "smcpp"})
        )
        if transition_matrix_only:
            print("Transition matrix mode requested; sequence simulation output was generated/copied, stopping before stats/inference for this deme.")
            continue
        if "stats" not in run_modes:
            print("Skipping sequence summary statistics because mode is", mode)
            gen = None
        else:
            print("READING GEN FILE")
            gen=FSC2_read_gen(it.gen_path)
            gen.to_csv(it.fullname+"_filtered_gen.gz", compression="gzip")
            os.system("cp "+it.fullname+"_filtered_gen.gz RESULTS/"+r.name+"/")

            tons,counts=FSC2_fsfs(r,gen,deme)
            sfs=counts
            print("SFS count for", it.name, sfs)
            if sum(sfs)!=0:
                sst=SFS_stats(sfs,tons,counts,it.fullname,r.sizes,r,deme)

            if r.mtype == "panmictic":
                pass
            elif r.mtype == "SST":
                sampled_demes=sampling_demes
                if deme==sampling_demes[0]:
                    FSC2_FST(gen,r,it, sampled_demes)
            elif r.mtype == "Free":
                if deme==sampling_demes[0]:
                    comb_sampled=set(itertools.combinations(sampling_demes, 2))
                    for sampled_demes in comb_sampled:
                        FSC2_FST(gen,r,it, sampled_demes)
            else:
                fst_demes = [
                    index + 1
                    for index, sample_count in enumerate(r.samples)
                    if int(sample_count) > 0
                ]
                if len(fst_demes) >= 2:
                    FSC2_FST(gen,r,it, sampled_demes=fst_demes[:2])
                else:
                    print("Skipping FST: fewer than two demes contain samples.")

        if "smcpp" in run_modes:
            print("Running SMC++ with deme", deme)
            SMC_run(i,j_name,deme)

        if "psmc" in run_modes:
            print("Running PSMC with deme", deme)
            for psmc_pattern in psmc_patterns:
                PSMC_fullrun(j_name,i,deme,s=psmc_s,psmc_pattern=psmc_pattern)

def FSC2_run_with_mrca(par, compress=True):
    '''
    Run fastsimcoal2 and record MRCA values.
    '''
    sentence=TOOLS["fastsimcoal2"]+" -i "+par+" -n1 -I -G -g -s0 -x -k 100000000 -q -d --recordMRCA"
    run_external(sentence)

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
        gen=FSC2_filter_segregating(gen)

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
    a=g[filter_col]
    return a

def FSC2_fsfs(r,gen,p, filter_1pop=True):
    '''
    Gets the Magikarp Model type object pd dataframe of the gen table and the population index (str)
    Returns the plain folded sfs count (from singletons) without normalizing.
    '''
    if filter_1pop:
        gen_1p=FSC2_gen_filter_1p(gen,int(p))
    else:
        gen_1p=gen

    if r.mtype!="panmictic":
        sample=int(r.samples[int(p)-1])
        max_count=sample//2
        snp_counts=np.where(gen_1p.sum(axis=1)>max_count,sample-gen_1p.sum(axis=1),gen_1p.sum(axis=1))
    else:
        sample=int(r.samples[0])
        max_count=sample//2
        print("Sample is",sample, type(sample),". Max count for fSFS is ", max_count)
        snp_counts=np.where(gen_1p.sum(axis=1)>max_count,sample-gen_1p.sum(axis=1),gen_1p.sum(axis=1))

    snp_counts=np.asarray(snp_counts, dtype=int)
    folded_counts=np.bincount(snp_counts, minlength=max_count+1)
    tons=np.arange(1, max_count+1)
    counts=folded_counts[1:max_count+1]

    return tons,counts

def FSC2_gen2VCF(gen,it,p=0,r=None):
    """
    Generates a vcf-like file from the fsc2 .gen table
    - gen: pd dataframe of gen table
    - p: population. If non specified or 0, all the samples are retrieved
    Returns: A tuple
    - [0]: path_to_vcf_file
    - [1]: list of individual samples
    """
    thisP=FSC2_gen_filter_1p(gen,p) # get only genotypes for 1 pop
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
    in_vcf["FILTER"]="PASS"
    in_vcf["INFO"]="DP=100"
    in_vcf["FORMAT"]="GT"
    ind_df=pd.DataFrame(thisP_mat,columns=inds)
    in_vcf.reset_index(drop=True, inplace=True) # reset indexes to concat
    ind_df.reset_index(drop=True,inplace=True)
    in_vcf=pd.concat([in_vcf,ind_df],axis=1)
    tsv_file=it.fullname+"_deme_"+str(p)+".tempvcf"
    vcf_file=it.fullname+"_deme_"+str(p)+".vcf"
    in_vcf.to_csv(tsv_file,index=False, sep="\t") # write tsv
    write_vcf(tsv_file, vcf_file)
    return(vcf_file,inds)

def FSC2_FST(gen, r, it, sampled_demes=[1,2]):
    print("Sampled demes are:", sampled_demes)
    frequencies=pd.DataFrame()
    deme_sizes=[r.samples[sampled_demes[0]-1], r.samples[sampled_demes[1]-1]]
    Nt=sum([r.samples[d-1]/2 for d in sampled_demes])
    print("Nt is", Nt)
    print("total sample size is: ", Nt)
    for deme in sampled_demes:
        print("Samples in this deme are:",r.samples[deme-1] /2)
        this_deme_gen=FSC2_gen_filter_1p(gen,deme)
        # 1. Get allele frequencies per population and overall populations (2 more columns per dataframe)
        p=this_deme_gen.sum(axis=1)/r.samples[deme-1]
        q=1-p
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
    # 6. Calculate mean frequencies (metapop frequencies)
    filter_col= [col for col in frequencies if col.startswith("pond_p")]
    ps=frequencies[filter_col]
    print("ps are", ps)
    frequencies["prev_mean_p"]=ps.sum(axis=1) ######### ONLY works if all demes have the same size
    frequencies["mean_p"]=frequencies["prev_mean_p"]/(int(Nt)*2) ######### ONLY works if all demes have the same size
    filter_col= [col for col in frequencies if col.startswith("pond_q")]
    ps=frequencies[filter_col]
    frequencies["prev_mean_q"]=ps.sum(axis=1) ######### ONLY works if all demes have the same size
    frequencies["mean_q"]=frequencies["prev_mean_q"]/(int(Nt)*2) ######### ONLY works if all demes have the same size

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

        # Copy it to result folder

def PSMC_estimate_sample_per_individual(individual,svalue, it, r, gen, p, psmc_pattern=DEFAULT_PSMC_PATTERN):
    consensus=PSMC_consensus_fa(it,r,gen,p, individual, psmc_pattern=psmc_pattern)
    consensus_fq=PSMC_convert_fastq(consensus,it, individual,p, psmc_pattern=psmc_pattern)
    svalue=str(svalue)
    psmcfa=PSMC_convert_input(consensus_fq,it,individual,p,svalue, psmc_pattern=psmc_pattern)
    PSMC_run(psmcfa,it,r,individual,p,svalue,psmc_pattern=psmc_pattern)
    lpsmc=PSMC_last_iteration(it,r,individual,p,svalue,psmc_pattern=psmc_pattern)
    this_sample_dict=PSMC_sample(psmc=lpsmc,svalue=svalue,mu=float(r.mu))
    return(this_sample_dict)

def PSMC_fullrun(j_name,i,p,s=DEFAULT_PSMC_S, psmc_pattern=DEFAULT_PSMC_PATTERN, two_bin_sizes=False,study_bin_size=False):

    '''
    Run PSMC for every sampled individual.
    - j_name: json file defining the model
    - i: repetition
    - p: population
    '''

    start_time = time.time()

    d=get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")
    if type_model=="StSi":
       r=StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=panmictic(d=d)
    elif type_model=="SST":
        r=SST(d=d) # parse the model
    elif type_model=="Free":
        r=Free(d=d) # parse the mode
    else:
        print("Model incorrect")
        sys.exit()

    it=Repetition_model(r,i,p) # parse the repetition
    gen=FSC2_read_gen(it.gen_path)

    print("Calling PSMC...",i)
    psmc_patterns = parse_psmc_patterns(psmc_pattern)
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

    for current_psmc_pattern in psmc_patterns:
        print("Running PSMC vector:", current_psmc_pattern)
        for svalue in svalues:
            results=[]
            for individual in range(1,individuals+1):
                results.append(PSMC_estimate_sample_per_individual(individual,svalue,it, r, gen , p, psmc_pattern=current_psmc_pattern ))

            sample_dicts = results
            PSMC_mean_inferences(sample_dicts=sample_dicts, svalue=svalue,r=r,it=it,p=p,psmc_pattern=current_psmc_pattern)

def PSMC_sample(psmc,svalue, mu, scheme="log"):
    print("Sampling PSMC...")
    svalue=int(svalue)
    sample_dict={}

    if scheme == "both":
        print("Do both sampling schemes")
        scheme = ["linear", "log"]
    else:
        scheme = [scheme]
    for sc in scheme:

        if sc == "linear":
            xvector=lin_vector
        elif sc == "log":
            xvector=log_vector
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
        sample_dict[sc]=this_psmc_sampling
    return sample_dict

def PSMC_mean_inferences(sample_dicts, svalue,r, it, save_tmp=True, p=None, psmc_pattern=DEFAULT_PSMC_PATTERN):
    svalue=str(svalue)
    pattern_suffix = psmc_pattern_suffix(psmc_pattern)
    print("Calculating mean of psmc inferences for model ", r.id)
    log_sampled=[]
    lin_sampled=[]
    params=["#"+par for par in r.params]
    for dic in sample_dicts:
        if "log" in dic:
            log_sampled.append(dic.get("log"))
        elif "linear" in dic:
            lin_sampled.append(dic.get("linear"))
        else:
            print("Not accepted sample schemes")

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
            o.write(line)
            o.close()

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
                listofill=[str(r.id),str(r.N[0]),str(r.islands),str(r.mig[0]),str(it.rep),str_log_time,str_log_iicr,str_log_psmc]
        elif r.mtype=="SST":
            listofill=[str(r.id),str(r.N[0]),str(r.mig[0]), str(r.M),str(r.L),str(p),str(it.rep),str_log_time,str_log_iicr,str_log_psmc]
        elif r.mtype == "Free":
            listofill=[r.id,  str(r.islands), str(r.n_mm), str(r.n_events), str(r.gr[0]), r.name,str(it.pop),str(it.rep),str_log_time,str_log_iicr,str_log_psmc]
        else:
            print("Model "+r.mtype+" still not implemented")

        with open("RESULTS/"+tyop, "a") as logf:
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
            o.write(line)
            o.close()

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
            listofill=[r.id, str(r.N[0]),str(r.islands),str(r.mig[0]), str(r.M),str(it.rep),str_lin_time,str_lin_iicr,str_lin_psmc]
        elif r.mtype == "SST":
            listofill=[r.id, str(r.N[0]),str(r.mig[0]), str(r.M),str(r.L),str(p),str(it.rep),str_lin_time,str_lin_iicr,str_lin_psmc]
        elif r.mtype == "Free":
            listofill=[r.id,  str(r.islands), str(r.n_mm), str(r.n_events), str(r.gr[0]), r.name,str(it.pop),str(it.rep),str_log_time,str_log_iicr,str_log_psmc]
        else:
            print("Model "+r.mtype+" still not implemented")

        with open("RESULTS/"+tyop, "a") as linf:
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
    pattern_suffix = psmc_pattern_suffix(psmc_pattern)
    consensus=it.fullname+"_deme_"+str(p)+"_ind_"+str(individual)+pattern_suffix+".consensus.fa"

    #Doing it for all the individuals sampled

   #     individual+=1

    for c in range(len(list(r.sizes))):
        chr=c+1
        size=int(r.sizes[c])
        tmp_fa=r.name+"/"+r.name+"_"+str(chr)+".fa"# temporary fa for chromosome
    # 1.2. Change the heterozygote positions
        this_chr=gen[gen["Chrom"]==chr]
        if int(p):
            # Sample the individual we want (first from the designed population)
            this_chr=FSC2_gen_filter_1i(this_chr,p,individual=individual)
                # call again FSC2 for this repetition
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
        position_count=0
        for snp in positions:
            # +2 and +3 to get the end (needs to be like this)
            init=snp+2
            end=snp+3
            m[init:end]=b"S" # b=binary
            position_count+=1
        print(position_count, "added positions in",chr)
            # +2 and +3 to get the end (needs to be like this)
        co.append(this_chr)
        os.close(f) # close
        print(consensus_tmp, "has been created")

    print("Analyses performed with individual", str(individual), "from population", str(p))

    time.sleep(180)

    if int(r.chr)==1: # only one chromosome. Fix the bug of non copying the consensus
        print("Only one chromosome in this model")
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

def PSMC_convert_fastq(consensus,it,individual,p,psmc_pattern=DEFAULT_PSMC_PATTERN):
    '''
    Convert the consensus .fa into a fastq.gz
    '''
    #it.fullname+"_ind_"+str(individual)+".consensus.fa"
    pattern_suffix = psmc_pattern_suffix(psmc_pattern)
    consensus_fq=it.fullname+"_deme_"+str(p)+"_ind_"+str(individual)+pattern_suffix+".consensus.fq.gz"
    sentence="cat "+consensus+" | "+TOOLS["seqtk"]+" seq -F 'I' | gzip > "+consensus_fq
    run_external(sentence)
    print(consensus_fq+" created")

    return consensus_fq

def PSMC_convert_input(consensus_fq,it, individual,p,s=100,psmc_pattern=DEFAULT_PSMC_PATTERN):
    '''
    Get proper psmc input
    '''
    s=str(s)
    pattern_suffix = psmc_pattern_suffix(psmc_pattern)
    psmcfa=it.fullname+"_s"+s+"_deme_"+str(p)+"_ind_"+str(individual)+pattern_suffix+".consensus.psmcfa"
    sentence=TOOLS["fq2psmcfa"]+" -q0 -s "+str(s)+" "+consensus_fq+" > "+psmcfa
    run_external(sentence)
    print(psmcfa+" created")
    return psmcfa

def PSMC_last_iteration(it,r,individual,p,s=100,psmc_pattern=DEFAULT_PSMC_PATTERN):

    '''
    Reads psmc output file (name.psmc) and extracts only the last iteration
    Creates a new file name.lpsmc
    Returns name.lpsmc path
    '''
    s=str(s)
    output_prefix = psmc_output_prefix(it, s, p, individual, psmc_pattern)
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
    pattern_suffix = psmc_pattern_suffix(psmc_pattern)
    sentence="rm "+it.fullname+"*"+pattern_suffix+".consensus.fatmp"
    os.system(sentence)
    sentence="rm "+it.fullname+"*"+pattern_suffix+".consensus.fa"
    os.system(sentence)
    s=str(s)
    start_time = time.time()
    output_prefix = psmc_output_prefix(it, s, p, individual, psmc_pattern)
    sentence=TOOLS["psmc"]+' -t15 -r1 -p "'+str(psmc_pattern)+'" -d -o '+output_prefix+'.psmc '+psmcfa
    run_external(sentence)
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
        print(r.name,"PSMC",str(runtime),str(s)+"/"+str(individual)+"/"+psmc_pattern_label(psmc_pattern), sep="\t", file=f)
    print("Remove psmcfa")
    sentence="rm "+psmcfa
    os.system(sentence)

    return psmc

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
    sentence=TOOLS["bgzip"]+" -c "+vcf+" > "+vcfgz
    run_external(sentence)
    # index the vcf
    print("Indexing vcf file")
    sentence=TOOLS["tabix"]+" "+vcfgz
    run_external(sentence)
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
            sentence=TOOLS["smcpp"]+" vcf2smc -d "+dis+" "+dis+" "+vcfgz+" "+smcgz+" "+str(c)+" "+undis+" --length "+str(r.sizes[i])
            run_external(sentence)

def SMC_estimate(it,r,dis,p,s=0):
    '''
    Runs smc++ estimation using all the smc.gz files generated for this repetition
    '''
    #smc++ estimate 1e-8 toy.*.smc.gz
    sentence=TOOLS["smcpp"]+" estimate "+str(r.mu)+" "+it.dir+"/smcpp/*_deme_"+str(p)+"_ind_"+dis+".*.smc.gz -o "+it.dir+"/smcpp --base _deme_"+str(p)+"_ind_"+str(dis)+"_s"+str(s)+" --timepoints "+str(s)+" 100000"

    print("Running SMC++ estimation")
    run_external(sentence)
    # Change the name with the starting point
    model_json = it.dir+"/smcpp/_deme_"+str(p)+"_ind_"+str(dis)+"_s"+str(s)+".final.json"
    return model_json

def SMC_arrange(model_json,it,dis,p,s=0):
    '''
    Calls the smc++ plot function to get the csv output (and a png)
    '''
    # smc++ plot plot.png --csv model.final.json
    sentence=TOOLS["smcpp"]+" plot "+it.dir+"/smcpp/"+it.name+"_deme_"+str(p)+"_ind_"+str(dis)+"_start_"+str(s)+".png"+" --csv "+model_json
    run_external(sentence)
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
    for sc in scheme:

        if sc == "linear":
            xvector=lin_vector
        elif sc == "log":
            xvector=log_vector
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
        sample_dicts[sc]=this_smcpp_sampling

    return sample_dicts

def SMC_mean_inferences(sample_dicts,r, it,p=None, save_tmp=True):
    print("Calculating mean of smcpp inferences for model ", r.id)
    log_sampled=[]
    lin_sampled=[]
    params=["#"+par for par in r.params]
    for dic in sample_dicts:
        if "log" in dic:
            log_sampled.append(dic.get("log"))
        if "linear" in dic:
            lin_sampled.append(dic.get("linear"))
        else:
            print("Not accepted sample schemes")

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
            o.write(line)
            o.close()

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
                listofill=[str(r.id),str(r.N[0]),str(r.islands),str(r.mig[0]),str(it.rep),str_log_time,str_log_iicr,str_log_smcpp]
        elif r.mtype=="SST":
            listofill=[str(r.id),str(r.N[0]),str(r.mig[0]), str(r.Nm),str(r.L),str(p),str(it.rep),str_log_time,str_log_iicr,str_log_smcpp]
        elif r.mtype == "Free":
            listofill=[r.id,  str(r.islands), str(r.n_mm), str(r.n_events), str(r.gr[0]), r.name,str(it.pop),str(it.rep),str_log_time,str_log_iicr,str_log_smcpp]
        else:
            print("Model "+r.mtype+" still not implemented")

        with open("RESULTS/"+tyop, "a") as logf:
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
            o.write(line)
            o.close()

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
            listofill=[r.id, str(r.N[0]),str(r.islands),str(r.mig[0]), str(r.M),str(it.rep),str_lin_time,str_lin_iicr,str_lin_smcpp]
        elif r.mtype == "SST":
            listofill=[r.id, str(r.N[0]),str(r.mig[0]), str(r.Nm),str(r.L),str(p),str(it.rep),str_lin_time,str_lin_iicr,str_lin_smcpp]
        elif r.mtype == "Free":
            listofill=[r.id,  str(r.islands), str(r.n_mm), str(r.n_events), str(r.gr[0]), r.name,str(it.pop),str(it.rep),str_lin_time,str_lin_iicr,str_lin_smcpp]  ###############################################################################################################
        else:
            print("Model "+r.mtype+" still not implemented")

        with open("RESULTS/"+tyop, "a") as linf:
            line="\t".join(listofill)+"\n"
            linf.write(line)

def SMC_estimate_sample_per_individual(dis, it, r, p,s=0):
    # Prepare to run each individual in a parallel process
    # This function should be created inside the main SMCPP function
    # it, r and i should be local variables
    print("Starting smcpp with individual %s" % dis)
    start_inf_time=time.time()
    print("Estimating demographic model in SMC++")
    model_json=SMC_estimate(it,r,dis,p) # RUN SMCPP estimation
    print("Arrange and plot results for SMC++")
    csv_smcpp=SMC_arrange(model_json,it,dis,p) # GET THE CSV FROM SMCPP ESTIMATION
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
    this_sample_dicts=SMC_sample(csv_smcpp) # SAMPLE THE VECTOR OF TIMES OUT OF THE CSV
    return this_sample_dicts  # RETURN THE VECTOR OF SAMPLED TIMES

def SMC_run(i,j_name,p):
    start_time=time.time()
    d=get_from_json(j_name) # get the model in a dict
    type_model=d.get("mtype")

    if type_model=="StSi":
       r=StSi(d=d) # parse the model
    elif type_model=="panmictic":
        r=panmictic(d=d)
    elif type_model=="SST":
        r=SST(d=d) # parse the model
    elif type_model=="Free":
        r=Free(d=d) # parse the mode
    else:
        print("Model incorrect")
        sys.exit()


    it=Repetition_model(r,i,p) # parse the repetition
    gen=FSC2_read_gen(it.gen_path)

    print("Starting smc++ run...",str(it.rep))
    vcf,inds=FSC2_gen2VCF(gen,it,p,r=r)
    SMC_vcf2smc(vcf,inds,it,r,p)
    finish_time_vcf=time.time()
    time_vcf=finish_time_vcf-start_time
    # Do for each sampled individual
    sample_dicts=[]
    start_time=time.time()

    results=[]

    for dis in inds:
        results.append(SMC_estimate_sample_per_individual(dis,it,r, p))

    sample_dicts = results
    print("Now we have %d entries (individuals) in the dict list" % len(sample_dicts))
    # Call mean and write
    SMC_mean_inferences(sample_dicts,r,it,p)

    with open("gyarados_time_check.log","a") as f:
        print(r.name,"SMCPP",str(time.time()-start_time),"ALL_RUN", sep="\t", file=f)

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
    theta_P=theta_pi(sfs)/total_size
    print("theta_P: ", theta_P)
    theta_S=float(theta_s(sfs)/total_size)
    print("theta_S: ",theta_S)
    sample_size=2*len(sfs) # as it is folded
    S=segregating_sites(sfs)
    deno=(((calc_e1(sample_size)*S)+(calc_e2(sample_size)*S*(S-1)))**0.5)
    print("deno:",deno)
    if deno == 0:
        print("Tajima's D is undefined for this sample/SFS; returning NaN.")
        return np.nan
    TD=(theta_pi(sfs)-theta_s(sfs))/deno
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
    ss=pd.DataFrame({"theta_P":theta_pi(sfs)/total_size, "theta_S":theta_s(sfs)/total_size, "S":segregating_sites(sfs), "TD":TajimaD(sfs,sz)}, index=[0])
    sst=pd.concat([sst,ss])
    sst.to_csv(it_fullname+"_deme_"+str(p)+"_stats.tsv", index=False, sep="\t")
    print("Stats from SFS are in",it_fullname+"_deme_"+str(p)+"_stats.tsv")

    # Copy to results
    sentence="cp "+it_fullname+"_deme_"+str(p)+"_stats.tsv RESULTS/"+r.name+"/"
    print(sentence)
    os.system(sentence)

    res = {tons[i]: counts[i] for i in range(len(tons))}
    columns=list(tons)
    columns.append("Type")
    fsfs_df = pd.DataFrame(columns=columns, index=[0,1])
    fill_counts=list(counts)
    fill_counts.append("Counts")
    fill_normalized=normalize_sfs(counts)
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

def IICR_fullrun(r,p=1, fn=10000000, fs=1):
    '''
    Run FSC2 to get the distribution of T2 for an r model. Calculate the IICR.
    - Input: r magikarp model
    - Output: IICR dataframe (that is also saved, along with a plot)
    '''
    print("Start IICR with fn " + str(fn)+" and fs "+str(fs))
    start_time=time.time()
    par_name=IICR_create_par(r,p, fn, fs)  ##population is specified
    # IF SST or FREE
    # IF StSi or panmictic
    mrca_folder=IICR_FSC2_run_tMRCA(r, par_name)
    fsc_file=mrca_folder+par_name+"_mrca.txt"
    print("IICR file is " + str(fsc_file))
    sentence="mv -f "+par_name+" "+r.name

    os.system(sentence)

    tmrca=IICR_FSC2_read_tMRCA(fsc_file,r.N[p-1]) #### POPULATION SPECIFIED
    sentence="mkdir -p "+r.name+"/"+r.name+"_IICR" ### if it doesnt exist
    os.system(sentence)
    sentence= "cp "+fsc_file+" "+r.name+"/"+r.name+"_IICR/"
    os.system(sentence)

    IICR_lin_df=IICR_calculate(tmrca,r,mrca_folder,type="linear", island=p, par_name=par_name)
    IICR_log_df=IICR_calculate(tmrca,r,mrca_folder,type="log", island=p, par_name=par_name)

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
    if r.mtype == "SST":
        # Create array full of 0
        mrca_N=["0\n"]*(r.islands)
        # Change the value from the population per 2
        mrca_N[pop-1]="2\n"

    # Build the chromosome sizes
    # We will sample 100Millions chunks of 1000

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

    # Eliminate the chromosomes part, add the new one
    erase_from=headers_index[headers.index(ev_header)+1]#included
    p=p[:erase_from]
    p.append(str_chromosomes)

    # Create new par file
    mrca_par="".join(p)
    if r.mtype=="StSi" or r.mtype=="panmictic":
        par_name=r.name+"_fn"+str(fn)+"_fs"+str(fs)+"_IICR"
    elif r.mtype=="SST" or r.mtype=="Free": ############# CONSIDERING FREE MODELS BC SAMPLING CAN BE DIFFERENT TOO
        par_name=r.name+"_fn"+str(fn)+"_fs"+str(fs)+"_deme_"+str(pop)+"_IICR"

    mrca_par_path=r.name+"/"+par_name+".par"
    print("Par too long?????????")
    print(mrca_par_path)
    with open(mrca_par_path,"w") as npar:
        npar.write(mrca_par)

    return par_name

def IICR_FSC2_run_tMRCA(r, par_name):
    print("Running IICR_FSC2_run_tMRCA with par file: " + par_name)
    par_path=r.name+"/"+par_name+".par"
    sentence=TOOLS["fastsimcoal2"]+" -i "+par_path+" -n1 -I -x -k 100000000 -q --recordMRCA"
    run_external(sentence)
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
            path_IICR_df=r.iicr_lin_path
        if r.mtype=="SST" or r.mtype=="Free":
            path_IICR_df=r.iicr_lin_path[island]
    if type=="log":
        x=np.array(log_vector)/(2*Nd)
        this_path_IICR_df=mrca_folder+par_name+"_log.tsv"
        if r.mtype=="StSi" or r.mtype=="panmictic":
            path_IICR_df=r.iicr_log_path
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
        theor_IICR=compute_IICR_n_islands(t_k, params)
        IICR_theor_df=pd.DataFrame({"theor_IICR":r.N[island]*theor_IICR,"theor_time":2*r.N[island]*t_k,"M":M})
        IICR_theor_df.to_csv(path_to_save_theor, sep="\t", header=True, index=False)
        sentence="cp "+path_to_save_theor+" RESULTS/"+r.name+"/"
        os.system(sentence)

    return(IICR_df)
