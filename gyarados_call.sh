#!/bin/bash
#SBATCH -p workq
#SBATCH --exclude n025
#SBATCH --job-name=5d_complete_gyarados
#SBATCH --mem=10000

### GENOTOUL - GENOLOGIN
#module load bioinfo/psmc-0.6.5 # psmc software
#module load bioinfo/fsc2702 # fsc2 software
#module load bioinfo/seqtk-1.3 # to get fastq from a fasta (and sequencing processing)
#module load system/Python-3.7.4 # python3


### GENOTOUL - BIOINFO
module load bioinfo/fastsimcoal2/2709 # fsc2 software
module load devel/python/Python-3.7.9


################################################################

### PARSE PARAMETERS

## default parameters


## parser

## help

# a=eq_sim_nislands(simn_n_islands=6,simn_N=100000, simn_samples=10, simn_sizes=[10000]*4,simn_mig=0.0005)

help()
{
    echo "Usage: magikarp -n <nislands> -N <comma,separated,island,sizes> -s <comma,separated,island,sample,sizes>
                        -m <migration frequency>, -b <comma,separated,chrom,sizes>
                [ -n | --nislands ] : number of islands 
                [ -N | --Ndeme ] deme sizes (haploid). Comma separated or 1 single value if all the same
                [ -m | --migration ] migration rate
                [-s | --samples] sample sizes (haploid). Comma separated or 1 single value if all the same
                [-b | --sizes] size for each chromosome to simmulate. 
                [-t | --type] Type of demographic model (1:eq_sim_nislands)
                [-p | --population] Deme to sample
                [-i | --iterations] Number of iterations
                [-o | --mode] Comma-separated steps to run: iicr,simulate,stats,psmc,smcpp or none
                [-q | --psmc-patterns] Comma-separated PSMC -p vectors
                [-B | --psmc-s] fq2psmcfa -s bin size for PSMC
                [ -h | --help  ]"
    exit 2
}

SHORT=n:N:m:s:b:t:p:i:c:d:g:e:a:M:P:L:o:q:B:h
#nislands,Ndeme,migration,samples,size,type_model,pop,iterations,mutationrate,growrate,events
LONG=nislands:Ndeme:migration:samples:sizes:type:population:iterations:parfile:,mode:,psmc-patterns:,psmc-s:,help
OPTS=$(getopt -a -n magikarp --options $SHORT --longoptions $LONG -- "$@")

VALID_ARGUMENTS=$# # Returns the count of arguments that are in short or long options

## loop

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
    help
fi


### DEFAULT ###
MU=1e-08
GR=0
EVENTS=0
PAR=0
NMD=0
MODE=iicr,simulate,stats,psmc
PSMC_PATTERNS=4+25*2+4+6
PSMC_S=100

eval set -- "$OPTS"

while :
do
    case "$1" in
        -n | --nislands )
            n_ISLANDS="$2"
            shift 2
            ;;
        -N | --Ndeme )
            N_DEME="$2"
            shift 2
            ;;
        -m | --migration )
            MIG="$2"
            shift 2
            ;;
        -s | --samples )
            SAMPLES="$2"
            shift 2
            ;;
        -b | --sizes )
            SIZES=$2
            shift 2
            ;;
        -t | --type )
            TYPE=$2
            shift 2
            ;;
        -p | --population ) 
            POP=$2
            shift 2
            ;;
        -i | --iterations ) 
            ITER=$2
            shift 2
            ;;
	-c | --chromosomes )
            CHR=$2
            shift 2
            ;;
	-d | --divergence )
            MU=$2
            shift 2
            ;;
	-g | --growrate )
            GR=$2
            shift 2
            ;;
	-e | --events )
            EVENTS=$2
            shift 2
            ;;
    	-a | --parfile )
            PAR=$2
            shift 2
            ;;
	-M | --M )
            NMD=$2
            shift 2
            ;;

	-P | --P )
            P=$2
            shift 2
    
            ;;
    -L | --L )
            L=$2
            shift 2
    
            ;;
    -o | --mode )
            MODE=$2
            shift 2
            ;;
    -q | --psmc-patterns )
            PSMC_PATTERNS=$2
            shift 2
            ;;
    -B | --psmc-s )
            PSMC_S=$2
            shift 2
            ;;


        -h | --help)
            help
            ;;
        --)
            shift;
            break
            ;;
        *)
            echo "Unexpected option: $1"
            help
            ;;
    esac
done 


if [ ! $TYPE ]; then
echo "No model type specified"
exit 1
fi


if [ ! $ITER ]; then
echo "No iterations specified"
exit 1
fi




if  [ $TYPE -eq 2 ]; then
echo "Calling panmitic model"

echo "Parameters set to:
N = $N_DEME
sample = $SAMPLES
sizes = $SIZES
type = $TYPE
niter = $ITER
chr = $CHR
mu = $MU 
gr= $GR
events=$EVENTS"
NMD=$NMD
P=$P
mode=$MODE
psmc_patterns=$PSMC_PATTERNS
psmc_s=$PSMC_S

# Export
export N_DEME
export SAMPLES
export SIZES
export TYPE
export ITER
export CHR
export MU
export GR
export EVENTS
export NMD
export P
export mode
export psmc_patterns
export psmc_s

# Call panmitic model
python -c 'import gyarados as gy, os # import packages;
N=os.environ["N_DEME"] # import bash variable $N_DEME ;
sample=os.environ["SAMPLES"] # import bash variable $SAMPLES ;
sizes=os.environ["SIZES"] # import bash variable $SIZES ;
type=os.environ["TYPE"] # import bash variable $TYPE ;
niter=os.environ["ITER"] # import bash variable $ITER ;
chr=os.environ["CHR"] # import bash variable $CHR;
mu=os.environ["MU"] # import bash variable $MU
gr=os.environ["GR"];
e=os.environ["EVENTS"];
mode=os.environ["mode"];
psmc_patterns=os.environ["psmc_patterns"];
psmc_s=os.environ["psmc_s"];
gy.GYARADOS_PAR(type=type,N=N,sizes=sizes,sample=sample,niter=niter, chr=chr, mu=mu, gr=gr, evs=e, mode=mode, psmc_patterns=psmc_patterns, psmc_s=psmc_s) # run'



fi 

if [ $TYPE -eq 1 ]; then
echo "Equilibrium simmetric n-island model"

if [ ! $n_ISLANDS ]; then
echo "No number of demes specified"
exit 1
fi

if [ ! $MIG ]; then
echo "No number migration specified"
exit 1
fi


if [ ! $POP ]; then
echo "No population specified"
exit 1
fi

    echo "Calling equilibrium simmetric n-island model"

    ## checks





echo "Parameters set to:
nislands = $n_ISLANDS
Ndeme = $N_DEME
mig = $MIG
sample = $SAMPLES
sizes = $SIZES
type = $TYPE
niter = $ITER
p = $POP
chr = $CHR
mu=$MU
gr=$GR
events=$EVENTS
NMD=$NMD
P=$P
mode=$MODE
psmc_patterns=$PSMC_PATTERNS
psmc_s=$PSMC_S
"


## EXPORT

export n_ISLANDS
export N_DEME
export MIG
export SAMPLES
export SIZES
export TYPE
export POP
export ITER
export CHR
export MU
export GR
export EVENTS
export NMD
export P
export mode
export psmc_patterns
export psmc_s

## run the model 



    # 0: script name
    # 1: number of islands
    # 2: N for each deme
    # 3: Migration frequency
    # 4: Sample size 
    # 5: Chromosome Sizes
    # 6: Type of model
    # 7: Number of iterations 
    # 8: population


    python -c 'import gyarados as gy, os # import packages; 
nislands=os.environ["n_ISLANDS"] # import bash variable $n_ISLANDS ;
Ndeme=os.environ["N_DEME"] # import bash variable $N_DEME ;
mig=os.environ["MIG"] # import bash variable $MIG ;
sample=os.environ["SAMPLES"] # import bash variable $SAMPLES ;
sizes=os.environ["SIZES"] # import bash variable $SIZES ;
type=os.environ["TYPE"] # import bash variable $TYPE ;
niter=os.environ["ITER"] # import bash variable $ITER ;
pop=os.environ["POP"] # import bash variable $POP ;
chr=os.environ["CHR"] # import bash variable $CHR;
mu=os.environ["MU"] # import bash variable $MU;
gr=os.environ["GR"] # import bash $GR;
e=os.environ["EVENTS"] # import bash $EVENTS;
NMD=os.environ["NMD"] # import bash $NMD;
P=os.environ["P"] # import bash $NMD;
mode=os.environ["mode"];
psmc_patterns=os.environ["psmc_patterns"];
psmc_s=os.environ["psmc_s"];
gy.GYARADOS_PAR(type=type,N=Ndeme,sizes=sizes,sample=sample,niter=niter,chr=chr,nislands=nislands,mig=mig,p=pop, mu=mu, gr=gr, evs=e, M=NMD, Nt=P, mode=mode, psmc_patterns=psmc_patterns, psmc_s=psmc_s) # run'


fi

if  [ $TYPE -eq 3 ]; then
echo "Model type: Free"

if [ ! $PAR ]; then
echo "No population specified"
exit 1
fi

#if [ ! $POP ]; then
#echo "No population specified"
#exit 1
#fi 

export PAR
#export POP
export TYPE
export ITER
export MODE
export PSMC_PATTERNS
export PSMC_S

python -c 'import gyarados as gy, os # import packages;
#pop=os.environ["POP"];
par=os.environ["PAR"];
type=os.environ["TYPE"];
niter=os.environ["ITER"] # import bash variable $ITER;
mode=os.environ["MODE"];
psmc_patterns=os.environ["PSMC_PATTERNS"];
psmc_s=os.environ["PSMC_S"];
gy.GYARADOS_PAR(type=type, par=par, niter=niter, mode=mode, psmc_patterns=psmc_patterns, psmc_s=psmc_s)'

fi



if [ $TYPE -eq 4 ]; then
echo "2D Stepping stone"

if [ ! $L ]; then
echo "No grid size specified"
exit 1
fi

if [ ! $MIG ]; then
echo "No number migration specified"
exit 1
fi


if [ ! $POP ]; then
echo "No populations specified"
exit 1
fi

    echo "Calling stepping stone model..."

    ## checks

NM=$NMD



echo "Parameters set to:
L = $L
Ndeme = $N_DEME
mig = $MIG
sample = $SAMPLES
sizes = $SIZES
type = $TYPE
niter = $ITER
p = $POP
chr = $CHR
mu=$MU
gr=$GR
events=$EVENTS
NM=$NM
mode=$MODE
psmc_patterns=$PSMC_PATTERNS
psmc_s=$PSMC_S
"


## EXPORT

export L
export N_DEME
export MIG
export SAMPLES
export SIZES
export TYPE
export POP
export ITER
export CHR
export MU
export GR
export EVENTS
export NM
export mode
export psmc_patterns
export psmc_s


## run the model 



    # 0: script name
    # 1: number of islands
    # 2: N for each deme
    # 3: Migration frequency
    # 4: Sample size 
    # 5: Chromosome Sizes
    # 6: Type of model
    # 7: Number of iterations 
    # 8: population


    python -c 'import gyarados as gy, os # import packages; 
L=os.environ["L"] # import bash variable $n_ISLANDS ;
Ndeme=os.environ["N_DEME"] # import bash variable $N_DEME ;
mig=os.environ["MIG"] # import bash variable $MIG ;
sample=os.environ["SAMPLES"] # import bash variable $SAMPLES ;
sizes=os.environ["SIZES"] # import bash variable $SIZES ;
type=os.environ["TYPE"] # import bash variable $TYPE ;
niter=os.environ["ITER"] # import bash variable $ITER ;
pop=os.environ["POP"] # import bash variable $POP ;
chr=os.environ["CHR"] # import bash variable $CHR;
mu=os.environ["MU"] # import bash variable $MU;
gr=os.environ["GR"] # import bash $GR;
e=os.environ["EVENTS"] # import bash $EVENTS;
NM=os.environ["NM"] # import bash $NM;
mode=os.environ["mode"];
psmc_patterns=os.environ["psmc_patterns"];
psmc_s=os.environ["psmc_s"];
gy.GYARADOS_PAR(type=type,L=L, N=Ndeme,sizes=sizes,sample=sample,niter=niter,chr=chr,mig=mig,p=pop,mu=mu,rho=1e-8, evs=e, gr=gr,M=NM, mode=mode, psmc_patterns=psmc_patterns, psmc_s=psmc_s) # run'


fi
