#!/bin/bash
#SBATCH -p workq
#SBATCH -n 1
#SBATCH --output=R5d-%x.%j.out
#SBATCH --error=R5d-%x.%j.err
#SBATCH --mem=70000

### GENOTOUL - GENOLOGIN
#module load bioinfo/psmc-0.6.5 # psmc software
#module load bioinfo/fsc2702 # fsc2 software
#module load bioinfo/seqtk-1.3 # to get fastq from a fasta (and sequencing processing)
#module load system/Python-3.7.4 # python3
#module load bioinfo/bcftools-1.9

### GENOTOUL - BIOINFO
module load bioinfo/fastsimcoal2/2709 # fsc2 software
module load devel/python/Python-3.7.9

### Parse

i=$1
j_name=$2
p=$3

### Export
export i
export j_name
export p

### Call

python -c 'import gyarados as gy, os  # gyarados ;
i=os.environ["i"];
j_name=os.environ["j_name"];
p=os.environ["p"];
gy.GYARADOS_WORK(i, j_name, p)'

echo "Called for repetition $i in population $p"
