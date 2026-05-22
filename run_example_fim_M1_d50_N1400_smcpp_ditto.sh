#!/bin/bash

# Example FIM/StSi + Ditto run:
# d = 50 demes
# M = 1
# N = 1400 haploids per deme
# Nt = d * N = 70000
#
# Runs in the original structured model:
#   IICRsim + fastsimcoal2 simulation + summary statistics + SMC++
#
# Also builds the Ditto panmictic mimic from the simulated IICR and runs:
#   IICRsim + fastsimcoal2 simulation + summary statistics + SMC++

set -euo pipefail

D=50
N=1400
M=1
NT=70000
SAMPLES=20
ITERATIONS=1
POP=1
SIZES=100000000
CHROMOSOMES=5
MU=1e-8
MODE='iicr,simulate,stats,smcpp'

python run_psmc_fim_vector.py \
  --d-vector "${D}" \
  --n-vector "${N}" \
  --m-vector "${M}" \
  --samples "${SAMPLES}" \
  --sizes "${SIZES}" \
  --chromosomes "${CHROMOSOMES}" \
  --iterations "${ITERATIONS}" \
  --population "${POP}" \
  --mu "${MU}" \
  --mode "${MODE}" \
  --ditto
