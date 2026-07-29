#!/bin/bash

# Example FIM/StSi run:
# d = 50 demes
# M = 5
# N = 1400 haploids per deme
# Nt = d * N = 70000
#
# Runs: IICRsim + fastsimcoal2 simulation + summary statistics + PSMC.
# Set DITTO=true below to also run the panmictic Ditto mimic.

set -euo pipefail

D=50
N=1400
M=5
NT=70000
SAMPLES=20
ITERATIONS=1
POP=1
SIZES=100000000
CHROMOSOMES=5
MU=1e-8
PSMC_S=100
PSMC_PATTERN='4+25*2+4+6'
MODE='iicr,simulate,stats,psmc'
DITTO=false

CMD=(python run_psmc_fim_vector.py \
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
  --psmc-s "${PSMC_S}" \
  --psmc-pattern "${PSMC_PATTERN}")

if [ "${DITTO}" = true ]; then
  CMD+=(--ditto)
fi

"${CMD[@]}"
