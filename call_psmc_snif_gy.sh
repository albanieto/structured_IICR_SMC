#!/bin/bash
#SBATCH -p workq
#SBATCH --output=R-%x.%j.out
#SBATCH --mem=5000
#SBATCH -t 2-00:00:00

### MODULES

# GENOTOUL-GENOLOGIN
#module load bioinfo/psmc-0.6.5 # psmc software
#module load bioinfo/fsc2702 # fsc2 software
#module load bioinfo/seqtk-1.3 # to get fastq from a fasta (and sequences processing)
#module load system/Python-3.7.4 # python3
#module load bioinfo/bcftools-1.9

# GENOTOUL-BIOINFO
module load bioinfo/Bcftools/1.9
module load bioinfo/psmc/0.6.5
module load bioinfo/fastsimcoal2/2709
module load bioinfo/Seqtk/1.3
module load devel/python/Python-3.7.9



### Parse

i=$1
j_name=$2
p=$3
s=$4
psmc_pattern=${5:-4+25*2+4+6}

### Export
export i
export j_name
export p
export s
export psmc_pattern

### Call

python -c 'import gyarados as gy, os  # gyarados ;
i=os.environ["i"];
j_name=os.environ["j_name"];
p=os.environ["p"];
s=os.environ["s"];
psmc_pattern=os.environ["psmc_pattern"];
gy.PSMC_SNIF_fullrun(j_name,i,p,s,psmc_pattern=psmc_pattern)'

echo "Called for repetition ${i}"
