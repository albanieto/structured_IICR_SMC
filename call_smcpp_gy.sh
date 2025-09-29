#!/bin/bash
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --threads=1
#SBATCH --output=R-%x.%j.out
#SBATCH --mem=30000


### MODULES

#GENOTOUL-GENOLOGIN
#module load system/Miniconda3-4.4.10
#module load bioinfo/smcpp-v1.15.2
#module load bioinfo/bcftools-1.9
#module load system/Python-3.7.4 # python3

#GENOTOUL-GENOBIOINFO
module load devel/Miniconda/Miniconda3
module load bioinfo/SMC++/1.15.5
module load bioinfo/Bcftools/1.9
module load devel/python/Python-3.7.9
################################################################

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
gy.SMC_run(i, j_name, p)'

echo 'Called for repetition $i'

