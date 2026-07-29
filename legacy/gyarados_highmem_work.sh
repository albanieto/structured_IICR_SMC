#!/bin/bash
#SBATCH -p workq
#SBATCH -n 2
#SBATCH --output=R-5d%x.%j.out
#SBATCH --error=R-5d%x.%j.err
#SBATCH --mem=120000

### GENOTOUL - GENOLOGIN
#module load bioinfo/psmc-0.6.5 # psmc software
#module load bioinfo/fsc2702 # fsc2 software
#module load bioinfo/seqtk-1.3 # to get fastq from a fasta (and sequencing processing)
#module load system/Python-3.7.4 # python3
#module load bioinfo/bcftools-1.9

### GENOTOUL - BIOINFO
module bioinfo/fastsimcoal2/2709 # fsc2 software
module load devel/python/Python-3.7.9

### Parse

i=$1
j_name=$2
p=$3
mode=${4:-iicr,simulate,stats,psmc}
psmc_s=${5:-100}
psmc_patterns=${6:-4+25*2+4+6}

### Export
export i
export j_name
export p
export mode
export psmc_s
export psmc_patterns

### Call

python -c 'import gyarados as gy, os  # gyarados ;
i=os.environ["i"];
j_name=os.environ["j_name"];
p=os.environ["p"];
mode=os.environ["mode"];
psmc_s=os.environ["psmc_s"];
psmc_patterns=os.environ["psmc_patterns"];
gy.GYARADOS_WORK(i, j_name, p, mode=mode, psmc_s=psmc_s, psmc_patterns=psmc_patterns)'

echo "Called for repetition $i in population $p"
